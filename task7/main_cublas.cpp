#include <iostream>
#include <cmath>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <memory>

#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>
#include <boost/program_options.hpp>
#include <openacc.h>

namespace po = boost::program_options;

struct CublasHandleDeleter {
    void operator()(cublasHandle_t* handle) const {
        if (handle) {
            cublasDestroy(*handle);
            delete handle;
        }
    }
};

using CublasHandlePtr = std::unique_ptr<cublasHandle_t, CublasHandleDeleter>;

CublasHandlePtr createCublasHandle() {
    auto handle = new cublasHandle_t;
    if (cublasCreate(handle) != CUBLAS_STATUS_SUCCESS) {
        delete handle;
        throw std::runtime_error("Failed to create cuBLAS handle");
    }
    return CublasHandlePtr(handle);
}

void initializeGrid(double* __restrict__ grid, double* __restrict__ gridNew, int size) {
    std::memset(grid, 0, sizeof(double) * size * size);
    std::memset(gridNew, 0, sizeof(double) * size * size);

    grid[0]                 = 10.0;
    grid[size - 1]          = 20.0;
    grid[size * (size - 1)] = 30.0;
    grid[size * size - 1]   = 20.0;

    double tl = grid[0]; 
    double tr = grid[size - 1];
    double bl = grid[size * (size - 1)];
    double br = grid[size * size - 1];

    for (int i = 1; i < size - 1; ++i) {
        double factor = i / static_cast<double>(size - 1);
        grid[i]                     = tl + (tr - tl) * factor;
        grid[size * (size - 1) + i] = bl + (br - bl) * factor;
        grid[size * i]              = tl + (bl - tl) * factor;
        grid[size * i + size - 1]   = tr + (br - tr) * factor;
    }
}

void freeGrids(double* grid, double* gridNew) {
    free(grid);
    free(gridNew);
}

void solve(double* __restrict__ grid, double* __restrict__ gridNew, int size, double accuracy, int maxIters) {
    double* errorGrid;
    
    cudaMalloc((void**)&errorGrid, sizeof(double) * size * size);

    acc_map_data(errorGrid, errorGrid, sizeof(double) * size * size);

    auto handle = createCublasHandle();

    double error = accuracy + 1.0;
    int iter = 0;
    int maxErrorIdx = 0;

    const auto start = std::chrono::steady_clock::now();

    nvtxRangePushA("computation");

    #pragma acc data copy(grid[0:size*size], gridNew[0:size*size]) present(errorGrid[0:size*size])
    {
        while (error > accuracy && iter < maxIters) {
            nvtxRangePushA("update_grid");

            #pragma acc parallel loop collapse(2) present(grid, gridNew)
            for (int i = 1; i < size - 1; ++i) {
                for (int j = 1; j < size - 1; ++j) {
                    gridNew[i * size + j] = 0.25 * (
                        grid[(i + 1) * size + j] +
                        grid[(i - 1) * size + j] +
                        grid[i * size + j - 1] +
                        grid[i * size + j + 1]
                    );
                }
            }
            nvtxRangePop();

            if (iter % 1000 == 0) {
                nvtxRangePushA("compute_error");

                #pragma acc parallel loop collapse(2) present(grid, gridNew, errorGrid)
                for (int i = 1; i < size - 1; ++i) {
                    for (int j = 1; j < size - 1; ++j) {
                        errorGrid[i * size + j] = fabs(grid[i * size + j] - gridNew[i * size + j]);
                    }
                }

                #pragma acc host_data use_device(errorGrid)
                {
                    cublasIdamax(*handle, size * size, errorGrid, 1, &maxErrorIdx);

                    cudaMemcpy(&error, &errorGrid[maxErrorIdx - 1], sizeof(double), cudaMemcpyDeviceToHost);
                }

                nvtxRangePop();
            }

            nvtxRangePushA("copy");
            #pragma acc parallel loop collapse(2) present(grid, gridNew)
            for (int i = 1; i < size - 1; ++i) {
                for (int j = 1; j < size - 1; ++j) {
                    grid[i * size + j] = gridNew[i * size + j];
                }
            }
            nvtxRangePop();

            ++iter;
        }
    }

    nvtxRangePop();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Time:        " << elapsed.count() << " sec\n"
              << "Iterations:  " << iter << "\n"
              << "Error value: " << error << std::endl;

    acc_unmap_data(errorGrid);
    cudaFree(errorGrid);
}

int main(int argc, char* argv[]) {
    int gridSize;
    double accuracy;
    int maxIterations;

    po::options_description desc("Options");
    desc.add_options()
        ("help", "Show help message")
        ("size", po::value<int>(&gridSize)->default_value(256), "Grid size (NxN)")
        ("accuracy", po::value<double>(&accuracy)->default_value(1e-6), "Maximum allowed error")
        ("max_iterations", po::value<int>(&maxIterations)->default_value(1e6), "Maximum allowed iterations");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    std::cout << "Running GPU simulation with cuBLAS for reduction...\n\n";

    double* grid = static_cast<double*>(malloc(sizeof(double) * gridSize * gridSize));
    double* gridNew = static_cast<double*>(malloc(sizeof(double) * gridSize * gridSize));

    nvtxRangePushA("initialize");
    initializeGrid(grid, gridNew, gridSize);
    nvtxRangePop();

    solve(grid, gridNew, gridSize, accuracy, maxIterations);

    freeGrids(grid, gridNew);

    return 0;
}