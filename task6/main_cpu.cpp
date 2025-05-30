#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <cstring>
#include <boost/program_options.hpp>
#include <nvtx3/nvToolsExt.h>

namespace po = boost::program_options;

void initialize_grid(double* grid, double* new_grid, size_t N) {
    std::memset(grid, 0, N * N * sizeof(double));
    std::memset(new_grid, 0, N * N * sizeof(double));

    grid[0]           = 10.0;
    grid[N - 1]       = 20.0;
    grid[N * (N - 1)] = 30.0;
    grid[N * N - 1]   = 20.0;

    double tl = grid[0];
    double tr = grid[N - 1];
    double bl = grid[N * (N - 1)];
    double br = grid[N * N - 1];

    for (int i = 1; i < N - 1; ++i) {
        grid[i]               = tl + (tr - tl) * i / (N - 1);        
        grid[N * (N - 1) + i] = bl + (br - bl) * i / (N - 1);           
        grid[N * i]           = tl + (bl - tl) * i / (N - 1);           
        grid[N * i + N - 1]   = tr + (br - tr) * i / (N - 1);          
    }
}

double update_grid(const double* __restrict grid, double* __restrict new_grid, size_t N) {
    double max_error = 0.0;

    #pragma acc parallel loop reduction(max:max_error) 
    for (int i = 1; i < N - 1; ++i) {
        #pragma acc loop
        for (int j = 1; j < N - 1; ++j) {
            int idx = i * N + j;
            new_grid[idx] = 0.25 * (
                grid[(i + 1) * N + j] +
                grid[(i - 1) * N + j] +
                grid[i * N + j - 1] +
                grid[i * N + j + 1]
            );
            max_error = fmax(max_error, fabs(new_grid[idx] - grid[idx]));
        }
    }

    return max_error;
}

void copy_grid(double* __restrict dst, const double* __restrict src, size_t N) {
    #pragma acc parallel loop
    for (int i = 1; i < N - 1; ++i) {
        #pragma acc loop
        for (int j = 1; j < N - 1; ++j) {
            dst[i * N + j] = src[i * N + j];
        }
    }
}

void print_grid(const double* grid, size_t N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << std::setprecision(4) << grid[i * N + j] << " ";
        }
        std::cout << "\n";
    }
}

int main(int argc, char* argv[]) {
    int N;
    double accuracy;
    int max_iterations;

    po::options_description options("Allowed options");
    options.add_options()
        ("help", "show help")
        ("size", po::value<int>(&N)->default_value(256), "grid size (NxN)")
        ("accuracy", po::value<double>(&accuracy)->default_value(1e-6), "convergence threshold")
        ("max_iterations", po::value<int>(&max_iterations)->default_value(1e6), "maximum iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, options), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << options << "\n";
        return 0;
    }

    std::cout << "Starting heat distribution simulation...\n";

    double* grid     = (double*)malloc(sizeof(double) * N * N);
    double* new_grid = (double*)malloc(sizeof(double) * N * N);

    nvtxRangePushA("Initialization");    
    initialize_grid(grid, new_grid, N);
    nvtxRangePop();                         

    double error = accuracy + 1.0;
    int iter = 0;

    auto start_time = std::chrono::steady_clock::now();

    nvtxRangePushA("Main Loop");
    while (error > accuracy && iter < max_iterations) {
        nvtxRangePushA("Compute");
        error = update_grid(grid, new_grid, N);
        nvtxRangePop();

        nvtxRangePushA("Copy");
        copy_grid(grid, new_grid, N);
        nvtxRangePop();

        iter++;
    }
    nvtxRangePop();

    auto end_time = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "Execution time: " << elapsed << " seconds\n";
    std::cout << "Iterations: " << iter << "\n";
    std::cout << "Final error: " << error << "\n";

    if (N == 10 || N == 13) {
        print_grid(grid, N);
    }

    free(grid);
    free(new_grid);

    return 0;
}