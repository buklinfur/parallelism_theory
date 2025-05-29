#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <chrono>
#include <fstream>
#include <mutex>

using namespace std;

using Matrix = vector<vector<double>>;
using Vector = vector<double>;

void init_matrix(Matrix& mat, int seed) {
    mt19937_64 rng(seed);
    uniform_real_distribution<> dist(0.0, 1.0);
    for (auto& row : mat)
        for (auto& val : row)
            val = dist(rng);
}

void init_vector(Vector& vec, int seed) {
    mt19937_64 rng(seed);
    uniform_real_distribution<> dist(0.0, 1.0);
    for (auto& val : vec)
        val = dist(rng);
}

void multiply(const Matrix& mat, const Vector& vec, Vector& result, size_t start_row, size_t end_row) {
    size_t N = vec.size();
    for (size_t i = start_row; i < end_row; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < N; ++j) {
            sum += mat[i][j] * vec[j];
        }
        result[i] = sum;
    }
}

void log_result(ofstream& out, size_t threads, size_t N, double init_time, double mul_time) {
    static mutex file_mutex;
    lock_guard<mutex> lock(file_mutex);
    out << threads << "," << N << "," << init_time << "," << mul_time << endl;
}

void run_experiment(ofstream& out, size_t threads, size_t N) {
    Matrix mat(N, Vector(N));
    Vector vec(N);
    Vector result(N);

    auto start_init = chrono::high_resolution_clock::now();
    thread t1(init_matrix, ref(mat), 1);
    thread t2(init_vector, ref(vec), 2);
    t1.join();
    t2.join();
    auto end_init = chrono::high_resolution_clock::now();
    chrono::duration<double> init_time = end_init - start_init;

    auto start_mul = chrono::high_resolution_clock::now();

    vector<thread> workers;
    size_t block_size = N / threads;

    for (size_t t = 0; t < threads; ++t) {
        size_t start = t * block_size;
        size_t end = (t == threads - 1) ? N : start + block_size;
        workers.emplace_back(multiply, cref(mat), cref(vec), ref(result), start, end);
    }

    for (auto& t : workers) t.join();

    auto end_mul = chrono::high_resolution_clock::now();
    chrono::duration<double> mul_time = end_mul - start_mul;

    log_result(out, threads, N, init_time.count(), mul_time.count());

    cout << "Threads: " << threads
         << ", Size: " << N
         << ", Init: " << init_time.count()
         << "s, Mul: " << mul_time.count() << "s" << endl;
}

int main() {
    vector<size_t> thread_counts = {1, 2, 4, 7, 8, 16, 20, 40};
    vector<size_t> sizes = {20000, 40000};

    ofstream out("results.csv");
    out << "threads,size,init_time,mul_time\n";

    for (size_t size : sizes) {
        for (size_t threads : thread_counts) {
            run_experiment(out, threads, size);
        }
    }

    out.close();
    return 0;
}
