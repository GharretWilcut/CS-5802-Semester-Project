#include <mpi.h>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>

using std::vector;
using std::cout;
using std::endl;

double dist2(const double* a, const double* b, int d) {
    double s = 0.0;
    for(int i=0;i<d;i++)
        s += (a[i]-b[i])*(a[i]-b[i]);
    return s;
}

void generate_synthetic(vector<double>& points, int N, int D, int seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    int Kapprox = 5;
    vector<double> centers(Kapprox * D);
    for(int i=0;i<Kapprox*D;i++)
        centers[i] = nd(gen) * 5.0;

    points.resize((size_t)N * D);
    for(int i=0;i<N;i++){
        int c = i % Kapprox;
        for(int j=0;j<D;j++){
            points[i*D+j] = centers[c*D+j] + nd(gen);
        }
    }
}

int main() {
    MPI_Init(nullptr, nullptr);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int K = 3;
    int max_iters = 100;
    double tol = 1e-4;

    int N = 1'000'000; // 1M points
    int D = 2;

    vector<double> all_points;
    if(rank == 0){
        generate_synthetic(all_points, N, D, 1234);
    }

    MPI_Bcast(&N,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&D,1,MPI_INT,0,MPI_COMM_WORLD);

    vector<int> counts(size), displs(size);
    int base = N / size, rem = N % size;
    for(int i=0;i<size;i++){
        counts[i] = (base + (i < rem)) * D;
        displs[i] = (i == 0 ? 0 : displs[i-1] + counts[i-1]);
    }

    int local_N = counts[rank] / D;
    vector<double> local_points((size_t)local_N * D);

    MPI_Scatterv(all_points.data(), counts.data(), displs.data(),
                 MPI_DOUBLE, local_points.data(),
                 counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<double> centroids(K * D);
    if(rank == 0){
        for(int k=0;k<K;k++)
            for(int j=0;j<D;j++)
                centroids[k*D+j] = all_points[k*D+j];
    }
    MPI_Bcast(centroids.data(), K*D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    vector<double> local_sums(K*D), global_sums(K*D);
    vector<long long> local_counts(K), global_counts(K);

    for(int iter=0; iter<max_iters; iter++){
        std::fill(local_sums.begin(), local_sums.end(), 0.0);
        std::fill(local_counts.begin(), local_counts.end(), 0);

        for(int i=0;i<local_N;i++){
            int best = 0;
            double bestd = dist2(&local_points[i*D], &centroids[0], D);
            for(int k=1;k<K;k++){
                double d = dist2(&local_points[i*D], &centroids[k*D], D);
                if(d < bestd){ bestd = d; best = k; }
            }
            for(int j=0;j<D;j++)
                local_sums[best*D+j] += local_points[i*D+j];
            local_counts[best]++;
        }

        MPI_Allreduce(local_sums.data(), global_sums.data(),
                      K*D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts.data(), global_counts.data(),
                      K, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

        double local_shift = 0.0;
        for(int k=0;k<K;k++){
            for(int j=0;j<D;j++){
                double newv = global_sums[k*D+j] / global_counts[k];
                local_shift = std::max(local_shift,
                    std::abs(newv - centroids[k*D+j]));
                centroids[k*D+j] = newv;
            }
        }

        double global_shift;
        MPI_Allreduce(&local_shift, &global_shift, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if(global_shift < tol) break;
    }

    double t_local = MPI_Wtime() - t_start;
    double t_global;
    MPI_Reduce(&t_local, &t_global, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0){
        std::ofstream csv("kmeans_speedup.csv", std::ios::app);
        csv << size << "," << t_global << "\n";
        csv.close();
        cout << "Ranks: " << size
             << " Time: " << t_global << " s\n";
    }

    MPI_Finalize();
    return 0;
}
