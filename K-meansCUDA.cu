#include <cstdio>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess){ \
    printf("CUDA error at %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(x)); exit(EXIT_FAILURE);}} while(0)

using std::vector;

__device__ float dist2(const float* a, const float* b, int D){
    float s=0.0f;
    for(int i=0;i<D;i++){ float d=a[i]-b[i]; s+=d*d; }
    return s;
}

__global__ void assign_points(const float* points, const float* centroids, int* labels, int N, int D, int K){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;
    const float* p = &points[i*D];
    float best=1e30f; int best_k=0;
    for(int k=0;k<K;k++){
        float d = dist2(p,&centroids[k*D],D);
        if(d<best){ best=d; best_k=k; }
    }
    labels[i]=best_k;
}

__global__ void accumulate_centroids(const float* points, const int* labels, float* sums, int* counts, int N, int D){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;
    int k=labels[i];
    atomicAdd(&counts[k],1);
    for(int j=0;j<D;j++) atomicAdd(&sums[k*D+j], points[i*D+j]);
}

void generate_data(vector<float>& points, int N, int D){
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f,1.0f);
    points.resize(N*D);
    float centers[3][2]={{-5,-5},{0,5},{5,-3}};
    for(int i=0;i<N;i++){
        int c=i%3;
        for(int j=0;j<D;j++) points[i*D+j]=centers[c][j]+nd(rng);
    }
}

int main() {
    const int D = 2;
    const int K = 3;
    const int MAX_ITERS = 50;
    const float TOL = 1e-4f;

    int Ns[] = {100'000, 500'000, 1'000'000, 2'000'000}; // dataset sizes
    int block_sizes[] = {64,128,256,512}; // block sizes

    std::ofstream csv("kmeans_cuda_sweep.csv");
    csv << "N,block_size,kernel_time_s,iterations\n";

    for(int ni=0; ni<4; ni++){
        int N = Ns[ni];

        // Generate data
        vector<float> h_points;
        generate_data(h_points,N,D);

        // Allocate GPU memory
        float *d_points,*d_centroids,*d_sums;
        int *d_labels,*d_counts;
        CUDA_CALL(cudaMalloc(&d_points,N*D*sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_centroids,K*D*sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_sums,K*D*sizeof(float)));
        CUDA_CALL(cudaMalloc(&d_counts,K*sizeof(int)));
        CUDA_CALL(cudaMalloc(&d_labels,N*sizeof(int)));
        CUDA_CALL(cudaMemcpy(d_points,h_points.data(),N*D*sizeof(float),cudaMemcpyHostToDevice));

        vector<float> h_centroids(K*D);
        for(int k=0;k<K;k++){ int idx=rand()%N; for(int j=0;j<D;j++) h_centroids[k*D+j]=h_points[idx*D+j]; }

        cudaEvent_t start_ev, stop_ev;
        CUDA_CALL(cudaEventCreate(&start_ev));
        CUDA_CALL(cudaEventCreate(&stop_ev));

        for(int bi=0; bi<4; bi++){
            int BLOCK = block_sizes[bi];
            int GRID = (N + BLOCK - 1)/BLOCK;

            CUDA_CALL(cudaMemcpy(d_centroids,h_centroids.data(),K*D*sizeof(float),cudaMemcpyHostToDevice));

            float t_kernel_total=0.0f;
            int final_iter=0;

            for(int iter=0;iter<MAX_ITERS;iter++){
                final_iter=iter;
                CUDA_CALL(cudaMemset(d_sums,0,K*D*sizeof(float)));
                CUDA_CALL(cudaMemset(d_counts,0,K*sizeof(int)));

                CUDA_CALL(cudaEventRecord(start_ev));
                assign_points<<<GRID,BLOCK>>>(d_points,d_centroids,d_labels,N,D,K);
                CUDA_CALL(cudaEventRecord(stop_ev));
                CUDA_CALL(cudaEventSynchronize(stop_ev));
                float assign_ms=0; CUDA_CALL(cudaEventElapsedTime(&assign_ms,start_ev,stop_ev));
                t_kernel_total+=assign_ms;

                CUDA_CALL(cudaEventRecord(start_ev));
                accumulate_centroids<<<GRID,BLOCK>>>(d_points,d_labels,d_sums,d_counts,N,D);
                CUDA_CALL(cudaEventRecord(stop_ev));
                CUDA_CALL(cudaEventSynchronize(stop_ev));
                float accum_ms=0; CUDA_CALL(cudaEventElapsedTime(&accum_ms,start_ev,stop_ev));
                t_kernel_total+=accum_ms;

                vector<float> h_sums(K*D); vector<int> h_counts(K);
                CUDA_CALL(cudaMemcpy(h_sums.data(),d_sums,K*D*sizeof(float),cudaMemcpyDeviceToHost));
                CUDA_CALL(cudaMemcpy(h_counts.data(),d_counts,K*sizeof(int),cudaMemcpyDeviceToHost));

                float max_shift=0.0f;
                for(int k=0;k<K;k++){
                    if(h_counts[k]==0) continue;
                    for(int j=0;j<D;j++){
                        float newv=h_sums[k*D+j]/h_counts[k];
                        float diff=newv-h_centroids[k*D+j];
                        if(diff<0) diff=-diff;
                        if(diff>max_shift) max_shift=diff;
                        h_centroids[k*D+j]=newv;
                    }
                }
                CUDA_CALL(cudaMemcpy(d_centroids,h_centroids.data(),K*D*sizeof(float),cudaMemcpyHostToDevice));
                if(max_shift<TOL) break;
            }

            csv << N << "," << BLOCK << "," << t_kernel_total/1000.0 << "," << (final_iter+1) << "\n";
            printf("N=%d BLOCK=%d Kernel time=%.6f s Iterations=%d\n",N,BLOCK,t_kernel_total/1000.0,final_iter+1);
        }

        CUDA_CALL(cudaFree(d_points));
        CUDA_CALL(cudaFree(d_centroids));
        CUDA_CALL(cudaFree(d_sums));
        CUDA_CALL(cudaFree(d_counts));
        CUDA_CALL(cudaFree(d_labels));
        CUDA_CALL(cudaEventDestroy(start_ev));
        CUDA_CALL(cudaEventDestroy(stop_ev));
    }
}
