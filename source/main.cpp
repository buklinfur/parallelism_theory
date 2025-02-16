#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#ifndef DATA_TYPE
#define DATA_TYPE double
#endif

constexpr size_t N = 10000000;
constexpr double PI = 3.14159265358979323846;

int main() {
    std::vector<DATA_TYPE> data(N);
    
    for (size_t i = 0; i < N; ++i) {
        data[i] = std::sin(2 * PI * i / N);
    }
    
    DATA_TYPE sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
        sum += data[i];
    }
    
    std::cout << "Sum: " << sum << std::endl;
    
    return 0;
}