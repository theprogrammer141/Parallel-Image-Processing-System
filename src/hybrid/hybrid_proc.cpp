/**
 * Hybrid MPI + OpenMP Image Processing
 * ──────────────────────────────────────
 * Combines both parallelism levels:
 *   - MPI distributes row-strips across processes (distributed memory)
 *   - OpenMP parallelises pixel loops within each process (shared memory)
 *
 * This achieves two-level parallelism:
 *   Total parallelism = MPI_processes × OMP_threads
 *
 * Usage: mpirun -np <P> ./hybrid_proc [image] [omp_threads]
 */

#include "image_processing.h"
#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

static double wtime() { return MPI_Wtime(); }

// ─── Scatter/Gather Layout ───────────────────────────────────────────────────
struct SL {
    std::vector<int> counts, displs, rows, offsets;
};
static SL make_sl(int H, int nprocs, int W, int esz) {
    SL sl;
    sl.rows.resize(nprocs); sl.offsets.resize(nprocs);
    sl.counts.resize(nprocs); sl.displs.resize(nprocs);
    int base = H/nprocs, rem = H%nprocs, off = 0;
    for (int i = 0; i < nprocs; ++i) {
        sl.rows[i]    = base + (i < rem ? 1 : 0);
        sl.offsets[i] = off;
        sl.counts[i]  = sl.rows[i] * W * esz;
        sl.displs[i]  = off * W * esz;
        off += sl.rows[i];
    }
    return sl;
}
static std::vector<uchar> mat_to_vec(const cv::Mat& m) {
    return {m.data, m.data + m.total() * m.elemSize()};
}
static cv::Mat vec_to_mat(const std::vector<uchar>& v, int r, int c, int t) {
    cv::Mat m(r, c, t); std::copy(v.begin(), v.end(), m.data); return m;
}
static void local_gray(const cv::Mat& src, cv::Mat& g, int nt) {
    if (src.channels()==1){g=src;return;}
    g.create(src.rows, src.cols, CV_8UC1);
    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int r = 0; r < src.rows; ++r) {
        const cv::Vec3b* sp = src.ptr<cv::Vec3b>(r);
        uchar* dp = g.ptr<uchar>(r);
        for (int c = 0; c < src.cols; ++c)
            dp[c] = static_cast<uchar>(0.114*sp[c][0]+0.587*sp[c][1]+0.299*sp[c][2]);
    }
}

// ─── 1. Grayscale ────────────────────────────────────────────────────────────
static void hybrid_grayscale(const cv::Mat& img, cv::Mat& dst,
                              int rank, int nprocs, int nt,
                              MPI_Comm comm, double& elapsed) {
    int W=0, H=0;
    if (rank==0){W=img.cols; H=img.rows;}
    MPI_Bcast(&W,1,MPI_INT,0,comm); MPI_Bcast(&H,1,MPI_INT,0,comm);

    SL sl = make_sl(H, nprocs, W, 3);
    std::vector<uchar> lsrc(sl.rows[rank]*W*3);
    std::vector<uchar> fsrc; if (rank==0) fsrc=mat_to_vec(img);

    MPI_Barrier(comm); double t0=wtime();
    MPI_Scatterv(fsrc.data(), sl.counts.data(), sl.displs.data(), MPI_UNSIGNED_CHAR,
                 lsrc.data(), sl.counts[rank], MPI_UNSIGNED_CHAR, 0, comm);

    int lr = sl.rows[rank];
    cv::Mat lbgr = vec_to_mat(lsrc, lr, W, CV_8UC3);
    cv::Mat lgray(lr, W, CV_8UC1);

    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int r = 0; r < lr; ++r) {
        const cv::Vec3b* sp = lbgr.ptr<cv::Vec3b>(r);
        uchar* dp = lgray.ptr<uchar>(r);
        for (int c = 0; c < W; ++c)
            dp[c]=static_cast<uchar>(0.114*sp[c][0]+0.587*sp[c][1]+0.299*sp[c][2]);
    }

    SL sl_g = make_sl(H, nprocs, W, 1);
    std::vector<uchar> fdst; if (rank==0) fdst.resize(H*W);
    std::vector<uchar> lv(lgray.data, lgray.data+lr*W);
    MPI_Gatherv(lv.data(), sl_g.counts[rank], MPI_UNSIGNED_CHAR,
                fdst.data(), sl_g.counts.data(), sl_g.displs.data(),
                MPI_UNSIGNED_CHAR, 0, comm);

    elapsed = wtime()-t0;
    if (rank==0) dst=vec_to_mat(fdst, H, W, CV_8UC1);
}

// ─── Gaussian kernel ─────────────────────────────────────────────────────────
static const float G5[5][5]={{1,4,7,4,1},{4,16,26,16,4},{7,26,41,26,7},{4,16,26,16,4},{1,4,7,4,1}};
static constexpr float G5S=273.f;

// ─── 2. Gaussian Blur ────────────────────────────────────────────────────────
static void hybrid_gaussian(const cv::Mat& img, cv::Mat& dst,
                             int rank, int nprocs, int nt,
                             MPI_Comm comm, double& elapsed) {
    int W=0, H=0;
    if (rank==0){W=img.cols; H=img.rows;}
    MPI_Bcast(&W,1,MPI_INT,0,comm); MPI_Bcast(&H,1,MPI_INT,0,comm);

    SL sl=make_sl(H,nprocs,W,3);
    std::vector<uchar> lsrc(sl.rows[rank]*W*3);
    std::vector<uchar> fsrc; if(rank==0) fsrc=mat_to_vec(img);

    MPI_Barrier(comm); double t0=wtime();
    MPI_Scatterv(fsrc.data(),sl.counts.data(),sl.displs.data(),MPI_UNSIGNED_CHAR,
                 lsrc.data(),sl.counts[rank],MPI_UNSIGNED_CHAR,0,comm);

    int lr=sl.rows[rank], halo=2;
    int prev=rank-1, next=rank+1;
    std::vector<uchar> ha(halo*W*3,0), hb(halo*W*3,0);
    if(prev>=0)     MPI_Send(lsrc.data(),halo*W*3,MPI_UNSIGNED_CHAR,prev,10,comm);
    if(next<nprocs) MPI_Recv(hb.data(),halo*W*3,MPI_UNSIGNED_CHAR,next,10,comm,MPI_STATUS_IGNORE);
    if(next<nprocs) MPI_Send(lsrc.data()+(lr-halo)*W*3,halo*W*3,MPI_UNSIGNED_CHAR,next,11,comm);
    if(prev>=0)     MPI_Recv(ha.data(),halo*W*3,MPI_UNSIGNED_CHAR,prev,11,comm,MPI_STATUS_IGNORE);

    int ext=halo+lr+halo;
    std::vector<uchar> ev(ext*W*3);
    std::copy(ha.begin(),ha.end(),ev.begin());
    std::copy(lsrc.begin(),lsrc.end(),ev.begin()+halo*W*3);
    std::copy(hb.begin(),hb.end(),ev.begin()+(halo+lr)*W*3);
    if(prev<0)     for(int h=0;h<halo;++h) std::copy(ev.begin()+halo*W*3,ev.begin()+(halo+1)*W*3,ev.begin()+h*W*3);
    if(next>=nprocs) for(int h=0;h<halo;++h) std::copy(ev.begin()+(halo+lr-1)*W*3,ev.begin()+(halo+lr)*W*3,ev.begin()+(halo+lr+h)*W*3);

    cv::Mat em=vec_to_mat(ev,ext,W,CV_8UC3);
    cv::Mat lo(lr,W,CV_8UC3);

    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int r = 0; r < lr; ++r) {
        for (int c = 0; c < W; ++c) {
            cv::Vec3f acc(0,0,0);
            for (int kr=-halo;kr<=halo;++kr) {
                int rr=r+halo+kr;
                for (int kc=-halo;kc<=halo;++kc) {
                    int cc=std::clamp(c+kc,0,W-1);
                    float w=G5[kr+halo][kc+halo];
                    cv::Vec3b px=em.at<cv::Vec3b>(rr,cc);
                    acc[0]+=px[0]*w; acc[1]+=px[1]*w; acc[2]+=px[2]*w;
                }
            }
            lo.at<cv::Vec3b>(r,c)=cv::Vec3b(
                static_cast<uchar>(acc[0]/G5S),
                static_cast<uchar>(acc[1]/G5S),
                static_cast<uchar>(acc[2]/G5S));
        }
    }

    std::vector<uchar> fdst; if(rank==0) fdst.resize(H*W*3);
    std::vector<uchar> lv(lo.data,lo.data+lr*W*3);
    MPI_Gatherv(lv.data(),sl.counts[rank],MPI_UNSIGNED_CHAR,
                fdst.data(),sl.counts.data(),sl.displs.data(),MPI_UNSIGNED_CHAR,0,comm);
    elapsed=wtime()-t0;
    if(rank==0) dst=vec_to_mat(fdst,H,W,CV_8UC3);
}

// ─── 3. Sobel ────────────────────────────────────────────────────────────────
static void hybrid_sobel(const cv::Mat& img, cv::Mat& dst,
                          int rank, int nprocs, int nt,
                          MPI_Comm comm, double& elapsed) {
    int W=0, H=0;
    if(rank==0){W=img.cols; H=img.rows;}
    MPI_Bcast(&W,1,MPI_INT,0,comm); MPI_Bcast(&H,1,MPI_INT,0,comm);

    std::vector<uchar> fg;
    if(rank==0){ cv::Mat g; local_gray(img,g,nt); fg=mat_to_vec(g); }

    SL sl=make_sl(H,nprocs,W,1);
    std::vector<uchar> lg(sl.rows[rank]*W);

    MPI_Barrier(comm); double t0=wtime();
    MPI_Scatterv(fg.data(),sl.counts.data(),sl.displs.data(),MPI_UNSIGNED_CHAR,
                 lg.data(),sl.counts[rank],MPI_UNSIGNED_CHAR,0,comm);

    int lr=sl.rows[rank], halo=1;
    int prev=rank-1, next=rank+1;
    std::vector<uchar> ha(W,0), hb(W,0);
    if(prev>=0)     MPI_Send(lg.data(),W,MPI_UNSIGNED_CHAR,prev,20,comm);
    if(next<nprocs) MPI_Recv(hb.data(),W,MPI_UNSIGNED_CHAR,next,20,comm,MPI_STATUS_IGNORE);
    if(next<nprocs) MPI_Send(lg.data()+(lr-1)*W,W,MPI_UNSIGNED_CHAR,next,21,comm);
    if(prev>=0)     MPI_Recv(ha.data(),W,MPI_UNSIGNED_CHAR,prev,21,comm,MPI_STATUS_IGNORE);

    int ext=halo+lr+halo;
    std::vector<uchar> ev(ext*W);
    std::copy(ha.begin(),ha.end(),ev.begin());
    std::copy(lg.begin(),lg.end(),ev.begin()+halo*W);
    std::copy(hb.begin(),hb.end(),ev.begin()+(halo+lr)*W);
    if(prev<0)     std::copy(ev.begin()+halo*W,ev.begin()+(halo+1)*W,ev.begin());
    if(next>=nprocs) std::copy(ev.begin()+(halo+lr-1)*W,ev.begin()+(halo+lr)*W,ev.begin()+(halo+lr)*W);

    cv::Mat em=vec_to_mat(ev,ext,W,CV_8UC1);
    cv::Mat lo(lr,W,CV_8UC1); lo.setTo(0);

    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int r = 0; r < lr; ++r) {
        int rr=r+halo;
        for (int c=1;c<W-1;++c){
            int gx=-em.at<uchar>(rr-1,c-1)+em.at<uchar>(rr-1,c+1)
                   -2*em.at<uchar>(rr,c-1)+2*em.at<uchar>(rr,c+1)
                   -em.at<uchar>(rr+1,c-1)+em.at<uchar>(rr+1,c+1);
            int gy=-em.at<uchar>(rr-1,c-1)-2*em.at<uchar>(rr-1,c)-em.at<uchar>(rr-1,c+1)
                   +em.at<uchar>(rr+1,c-1)+2*em.at<uchar>(rr+1,c)+em.at<uchar>(rr+1,c+1);
            lo.at<uchar>(r,c)=static_cast<uchar>(std::min(static_cast<int>(std::sqrt(gx*gx+gy*gy)),255));
        }
    }
    if(rank==0)        lo.row(0).setTo(0);
    if(rank==nprocs-1) lo.row(lr-1).setTo(0);

    std::vector<uchar> fdst; if(rank==0) fdst.resize(H*W);
    std::vector<uchar> lv(lo.data,lo.data+lr*W);
    MPI_Gatherv(lv.data(),sl.counts[rank],MPI_UNSIGNED_CHAR,
                fdst.data(),sl.counts.data(),sl.displs.data(),MPI_UNSIGNED_CHAR,0,comm);
    elapsed=wtime()-t0;
    if(rank==0) dst=vec_to_mat(fdst,H,W,CV_8UC1);
}

// ─── 4. Brightness / Contrast ────────────────────────────────────────────────
static void hybrid_brightness(const cv::Mat& img, cv::Mat& dst,
                               double alpha, int beta,
                               int rank, int nprocs, int nt,
                               MPI_Comm comm, double& elapsed) {
    int W=0, H=0;
    if(rank==0){W=img.cols; H=img.rows;}
    MPI_Bcast(&W,1,MPI_INT,0,comm); MPI_Bcast(&H,1,MPI_INT,0,comm);
    MPI_Bcast(&alpha,1,MPI_DOUBLE,0,comm); MPI_Bcast(&beta,1,MPI_INT,0,comm);

    SL sl=make_sl(H,nprocs,W,3);
    std::vector<uchar> lsrc(sl.rows[rank]*W*3);
    std::vector<uchar> fsrc; if(rank==0) fsrc=mat_to_vec(img);

    MPI_Barrier(comm); double t0=wtime();
    MPI_Scatterv(fsrc.data(),sl.counts.data(),sl.displs.data(),MPI_UNSIGNED_CHAR,
                 lsrc.data(),sl.counts[rank],MPI_UNSIGNED_CHAR,0,comm);

    int lr=sl.rows[rank];
    cv::Mat lb=vec_to_mat(lsrc,lr,W,CV_8UC3), lo(lr,W,CV_8UC3);

    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int r = 0; r < lr; ++r) {
        const cv::Vec3b* sp=lb.ptr<cv::Vec3b>(r);
        cv::Vec3b* dp=lo.ptr<cv::Vec3b>(r);
        for (int c=0;c<W;++c)
            dp[c]=cv::Vec3b(
                static_cast<uchar>(std::clamp(static_cast<int>(sp[c][0]*alpha+beta),0,255)),
                static_cast<uchar>(std::clamp(static_cast<int>(sp[c][1]*alpha+beta),0,255)),
                static_cast<uchar>(std::clamp(static_cast<int>(sp[c][2]*alpha+beta),0,255)));
    }

    std::vector<uchar> fdst; if(rank==0) fdst.resize(H*W*3);
    std::vector<uchar> lv(lo.data,lo.data+lr*W*3);
    MPI_Gatherv(lv.data(),sl.counts[rank],MPI_UNSIGNED_CHAR,
                fdst.data(),sl.counts.data(),sl.displs.data(),MPI_UNSIGNED_CHAR,0,comm);
    elapsed=wtime()-t0;
    if(rank==0) dst=vec_to_mat(fdst,H,W,CV_8UC3);
}

// ─── 5. Histogram Equalization ───────────────────────────────────────────────
static void hybrid_histeq(const cv::Mat& img, cv::Mat& dst,
                           int rank, int nprocs, int nt,
                           MPI_Comm comm, double& elapsed) {
    int W=0, H=0;
    if(rank==0){W=img.cols; H=img.rows;}
    MPI_Bcast(&W,1,MPI_INT,0,comm); MPI_Bcast(&H,1,MPI_INT,0,comm);

    std::vector<uchar> fg;
    if(rank==0){ cv::Mat g; local_gray(img,g,nt); fg=mat_to_vec(g); }

    SL sl=make_sl(H,nprocs,W,1);
    std::vector<uchar> lg(sl.rows[rank]*W);

    MPI_Barrier(comm); double t0=wtime();
    MPI_Scatterv(fg.data(),sl.counts.data(),sl.displs.data(),MPI_UNSIGNED_CHAR,
                 lg.data(),sl.counts[rank],MPI_UNSIGNED_CHAR,0,comm);

    // Parallel histogram with OpenMP, then MPI_Reduce
    int lhist[256]={};
    #pragma omp parallel num_threads(nt)
    {
        int pvt[256]={};
        #pragma omp for schedule(static) nowait
        for (int i=0;i<(int)lg.size();++i) pvt[lg[i]]++;
        #pragma omp critical
        for (int i=0;i<256;++i) lhist[i]+=pvt[i];
    }
    int ghist[256]={};
    MPI_Reduce(lhist,ghist,256,MPI_INT,MPI_SUM,0,comm);

    uchar lut[256]={};
    if(rank==0){
        int total=H*W, cdf[256]={}; cdf[0]=ghist[0];
        for(int i=1;i<256;++i) cdf[i]=cdf[i-1]+ghist[i];
        int cm=0; for(int i=0;i<256;++i){if(cdf[i]>0){cm=cdf[i];break;}}
        for(int i=0;i<256;++i){
            if(total==cm){lut[i]=0;continue;}
            lut[i]=static_cast<uchar>(std::round((double)(cdf[i]-cm)/(total-cm)*255.0));
        }
    }
    MPI_Bcast(lut,256,MPI_UNSIGNED_CHAR,0,comm);

    std::vector<uchar> lo(lg.size());
    #pragma omp parallel for schedule(static) num_threads(nt)
    for(int i=0;i<(int)lg.size();++i) lo[i]=lut[lg[i]];

    std::vector<uchar> fdst; if(rank==0) fdst.resize(H*W);
    MPI_Gatherv(lo.data(),sl.counts[rank],MPI_UNSIGNED_CHAR,
                fdst.data(),sl.counts.data(),sl.displs.data(),MPI_UNSIGNED_CHAR,0,comm);
    elapsed=wtime()-t0;
    if(rank==0) dst=vec_to_mat(fdst,H,W,CV_8UC1);
}

// ─── CSV Writer ───────────────────────────────────────────────────────────────
static void save_csv(const std::vector<BenchmarkResult>& r, const std::string& p) {
    std::ofstream f(p); if(!f) throw std::runtime_error("Cannot open "+p);
    f<<"version,operation,width,height,threads,processes,elapsed_sec,speedup\n";
    for(const auto& x:r)
        f<<x.version<<','<<x.operation<<','<<x.image_width<<','<<x.image_height<<','
         <<x.threads<<','<<x.processes<<','<<x.elapsed_sec<<','<<x.speedup<<'\n';
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    std::string input  = "test_images/sample.jpg";
    std::string outdir = "results/images";
    std::string csv    = "results/data/hybrid_results.csv";
    int nt = omp_get_max_threads();
    if (argc > 1) input = argv[1];
    if (argc > 2) nt    = std::stoi(argv[2]);

    cv::Mat img;
    if(rank==0){
        img=cv::imread(input, cv::IMREAD_COLOR);
        if(img.empty()){
            std::cerr<<"Error: cannot load '"<<input<<"'\n";
            MPI_Abort(MPI_COMM_WORLD,1);
        }
        std::cout<<"Hybrid MPI+OpenMP  |  processes: "<<nprocs<<"  threads/proc: "<<nt<<'\n';
        std::cout<<"Total workers: "<<nprocs*nt<<'\n';
        std::cout<<"Image: "<<input<<" ("<<img.cols<<"x"<<img.rows<<")\n\n";
    }

    std::vector<BenchmarkResult> res;
    cv::Mat dst; double elapsed=0;

    auto record=[&](const std::string& op){
        if(rank==0){
            BenchmarkResult r;
            r.version="hybrid"; r.operation=op;
            r.image_width=img.cols; r.image_height=img.rows;
            r.threads=nt; r.processes=nprocs; r.elapsed_sec=elapsed; r.speedup=0.0;
            std::printf("  %-25s %d procs × %d threads  %.4f s\n",op.c_str(),nprocs,nt,elapsed);
            cv::imwrite(outdir+"/hybrid_"+op+"_p"+std::to_string(nprocs)+"_t"+std::to_string(nt)+".png",dst);
            res.push_back(r);
        }
    };

    hybrid_grayscale(img, dst, rank, nprocs, nt, MPI_COMM_WORLD, elapsed); record("grayscale");
    hybrid_gaussian(img, dst, rank, nprocs, nt, MPI_COMM_WORLD, elapsed);  record("gaussian_blur");
    hybrid_sobel(img, dst, rank, nprocs, nt, MPI_COMM_WORLD, elapsed);     record("sobel_edge");
    hybrid_brightness(img, dst, 1.2, 30, rank, nprocs, nt, MPI_COMM_WORLD, elapsed); record("brightness");
    hybrid_histeq(img, dst, rank, nprocs, nt, MPI_COMM_WORLD, elapsed);    record("histogram_eq");

    if(rank==0){ save_csv(res,csv); std::cout<<"\nResults saved to "<<csv<<"\n"; }
    MPI_Finalize();
    return 0;
}
