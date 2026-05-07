#include <iostream>
#include <chrono>
#define N 100000000

int main(){
    size_t size = N * sizeof(float);

    // host arrays (CPU)
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N; i++){
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // CPU vector addition
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < N; i++){
        h_C[i] = h_A[i] + h_B[i];
    }
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    float ms = duration.count() / 1000.0f;

    std::cout << "C[0] = " << h_C[0] << std::endl;
    std::cout << "C[N-1] = " << h_C[N-1] << std::endl;

    std::cout << std::fixed;
    std::cout.precision(6);
    std::cout << "CPU Kernel Time: " << ms << " ms" << std::endl;

    // free memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
