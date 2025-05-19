#include <iostream>
#include <numeric>
#include <iomanip>
#include <vector>
#include <chrono>
#include <omp.h>
#include <cmath>

#ifdef BIG
#define SIZE 40000
#else
#define SIZE 20000
#endif

#define TAU 0.01
#define EPS 0.0001

double euclid_norm(std::vector<double> vector, int N) {
    double euclid = 0.0;
    
    for (int i = 0; i < N; i++) {
        euclid += pow(vector[i], 2.0);
    }

    return sqrt(euclid);
}

std::vector<double> simple_iteration(std::vector<std::vector<double>>& A,
                                    std::vector<double>& v,
                                    std::vector<double>& u,
                                    int matrix_size,
                                    int num_threads,
                                    const std::string& schedule_type)
{
    std::vector<double> Av_minus_u(matrix_size, 0.0);
    double Av_minus_u_euclid;
    double u_euclid;
    int abort = 0;

    u_euclid = euclid_norm(u, matrix_size);
    // To avoid loop end miscalculation.
    Av_minus_u_euclid = u_euclid;

    #pragma omp parallel num_threads(num_threads)
    {
        do {
            if (schedule_type == "dynamic") {
                #pragma omp for schedule(dynamic, matrix_size/num_threads)
                for (int i = 0; i < matrix_size; i++) {
                    double Av = 0.0;
                    for (int j = 0; j < matrix_size; j++) {
                        Av += A[i][j] * v[j];
                    }
                    Av_minus_u[i] += Av - u[i];
                }
            }
            else {
                #pragma omp for schedule(static, matrix_size/num_threads)
                for (int i = 0; i < matrix_size; i++) {
                    double Av = 0.0;
                    for (int j = 0; j < matrix_size; j++) {
                        Av += A[i][j] * v[j];
                    }
                    Av_minus_u[i] += Av - u[i];
                }
            }

            #pragma omp master
            {
                Av_minus_u_euclid = euclid_norm(Av_minus_u, matrix_size);
            }

            if (schedule_type == "dynamic") {
                #pragma omp for schedule(dynamic, matrix_size/num_threads)
                for (int i = 0; i < matrix_size; i++) {
                    v[i] -= TAU * Av_minus_u[i];
                    Av_minus_u[i] = 0.0;
                }
            }
            else {
                #pragma omp for schedule(static, matrix_size/num_threads)
                for (int i = 0; i < matrix_size; i++) {
                    v[i] -= TAU * Av_minus_u[i];
                    Av_minus_u[i] = 0.0;
                }
            }
            
            #pragma omp single
            if (Av_minus_u_euclid / u_euclid > EPS) {
                abort = 1;
            }
        }
        while (abort != 1);
    }

    return v;
}

double run(int matrix_size, int num_threads, const std::string& schedule_type) {
    std::vector<std::vector<double>> A(matrix_size,
                                        std::vector<double>(matrix_size, 1.0));               // Matrix A
    std::vector<double> v(matrix_size, 0.0);                                                  // Vector v
    std::vector<double> u(matrix_size, matrix_size + 1);                                      // Vector u

    for (int i = 0; i < matrix_size; i++) {
        A[i][i] = 2.0;
    }

    const auto start{std::chrono::steady_clock::now()};
    v = simple_iteration(A, v, u, matrix_size, num_threads, schedule_type);
    const auto end{std::chrono::steady_clock::now()};
    const std::chrono::duration<double> elapsed_seconds{end - start};

    return elapsed_seconds.count();
}

int main() {
    double tserial, tparallel;

    std::vector<std::int16_t> threads = {2, 4, 8, 16, 20, 32, 40};

    std::cout << SIZE << " x " << SIZE << "\n";

    tserial = run(SIZE, 1, "serial");
    std::cout << "(serial) finished in " << std::setprecision(12) << tserial << "\n";

    std::vector<std::string> schedules = {"static", "dynamic"};
    for (const auto& schedule : schedules) {
        std::cout << schedule << " schedule\n";
        for (int num_threads : threads) {
            std::cout << num_threads << " threads used\n";
            tparallel = run(SIZE, num_threads, schedule);
            std::cout << "finished in " << std::setprecision(12) << tparallel << "\n";
            std::cout << "boost " << std::setprecision(6) << tserial / tparallel << "\n";  
        }
    }
    
    return 0;
}