#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>

#include <cub/cub.cuh>
#include <boost/program_options.hpp>
#include <cuda_runtime.h>

#define OFFSET(i, j, N) ((i) * (N) + (j))
constexpr int BLOCK_SIZE = 1024;

__global__ void calculate_step(int N,
                               const double *__restrict__ in,
                               double *__restrict__ out,
                               double *block_errors) {
    using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage tmp;  

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = gid / N, j = gid % N;
    double local_err = 0.0;

    if (i > 0 && i < N - 1 && j > 0 && j < N - 1)
    {
        double v = (in[OFFSET(i - 1, j, N)] +
                    in[OFFSET(i + 1, j, N)] +
                    in[OFFSET(i, j - 1, N)] +
                    in[OFFSET(i, j + 1, N)] +
                    in[OFFSET(i, j, N)]) *
                   0.2;
        out[OFFSET(i, j, N)] = v;
        local_err = fabs(v - in[OFFSET(i, j, N)]); 
    }

    double block_max = BlockReduce(tmp).Reduce(local_err, cub::Max());
    if (threadIdx.x == 0)
    {
        block_errors[blockIdx.x] = block_max; 
    }
}

__global__ void reduce_global(int num_blocks, double *block_errors) {
    using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage tmp;

    int t = threadIdx.x;
    double v = (t < num_blocks ? block_errors[t] : 0.0);
    double m = BlockReduce(tmp).Reduce(v, cub::Max());
    if (t == 0)
        block_errors[0] = m;
}

void init_boundary(std::vector<double> &A, int N)
{
    double TL = 10, TR = 20, BR = 30, BL = 20;

    for (int j = 0; j < N; ++j) {
        double t = double(j) / (N - 1);
        A[OFFSET(0, j, N)] = (1 - t) * TL + t * TR;
    }

    for (int i = 0; i < N; ++i) {
        double t = double(i) / (N - 1);
        A[OFFSET(i, N - 1, N)] = (1 - t) * TR + t * BR;
    }

    for (int j = 0; j < N; ++j) {
        double t = double(j) / (N - 1);
        A[OFFSET(N - 1, j, N)] = (1 - t) * BR + t * BL;
    }

    for (int i = 0; i < N; ++i) {
        double t = double(i) / (N - 1);
        A[OFFSET(i, 0, N)] = (1 - t) * BL + t * TL;
    }
}

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;
    int N;
    double eps;
    int max_iters;
    bool draw;

    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "help")
        ("size,s", po::value<int>(&N)->default_value(512), "grid size")
        ("max_error,me", po::value<double>(&eps)->default_value(1e-6), "tolerance")
        ("max_iterations,mi", po::value<int>(&max_iters)->default_value(1'000'000), "max iters")
        ("draw_output,do", po::bool_switch(&draw), "print final mat");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    std::vector<double> h_A(N * N, 0.0), h_B;
    init_boundary(h_A, N);
    h_B = h_A;

    double *d_in, *d_out, *d_err;
    cudaMalloc(&d_in, N * N * sizeof(double));
    cudaMalloc(&d_out, N * N * sizeof(double));
    int num_threads = N * N;
    int blocks = (num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMemcpy(d_in, h_A.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, h_B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice);

    double error = 1.0;
    int iter = 0;
    auto t0 = std::chrono::steady_clock::now();

    while (error > eps && iter < max_iters) {
        cudaGraph_t graph;
        cudaGraphExec_t graphExec;

        cudaStream_t s;
        cudaStreamCreate(&s);
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        
        calculate_step<<<blocks, BLOCK_SIZE, 0, s>>>(N, d_in, d_out, d_err);

        cudaStreamEndCapture(s, &graph);
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0); 

        cudaGraphLaunch(graphExec, s);
        cudaStreamSynchronize(s);

        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(s);

        reduce_global<<<1, BLOCK_SIZE>>>(blocks, d_err);
        cudaMemcpy(&error, d_err, sizeof(double), cudaMemcpyDeviceToHost);

        std::swap(d_in, d_out);
        ++iter;
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();

    std::cout << "Iterations:   " << iter << "\n"
              << "Final error:  " << error << "\n"
              << "Elapsed time: " << std::chrono::duration<double>(t1 - t0).count()
              << " s\n";

    if (draw) {
        cudaMemcpy(h_B.data(), d_out, N * N * sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j)
                std::cout << h_B[OFFSET(i, j, N)] << " ";
            std::cout << "\n";
        }
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_err);
    return 0;
}