#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#define N 100000000
__global__ void vectorAdd(const float *A, const float *B, float*C ,int num){
    int i = blockIdx.x*blockDim.x+threadIdx.x; //Index of the element this thread is used for.
    if (i<num){ //Checking bound conditions
        C[i] = A[i] + B[i];
    }
}
/* Formula intuition : We need a unique global index for each element. Thus we use blockDim.x*blockIdx.x + threadIdx.x
A block is a collection of multiple threads thus if we have multiple threads we need to assign each thread a index in a block to process.
A block can have at most 256 threads thus we need to specify block index when there are multiple blocks being used. */
int main(){
    size_t size = N*sizeof(float);

    //host arrays (CPU)
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0 ; i<N ; i++){
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    //device arrays (GPU)

    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);

    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size , cudaMemcpyHostToDevice);

    int thdpblock = 256; 
    //but why 256?
    /*GPU processes threads in warps where 1 warp = 32 threads hence 256/32 = 8 , makes it efficienct thus multiples of 32 are chosen for efficiency*/
    int blcpgrid = (N+thdpblock-1)/thdpblock;

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    vectorAdd<<<blcpgrid,thdpblock>>> (d_A,d_B,d_C,N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f ; 
    cudaEventElapsedTime(&ms,start,stop);
    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);

    std::cout<<"C[0] = "<<h_C[0]<<std::endl;
    std::cout<<"C[N-1] = "<<h_C[N-1]<<std::endl;

    std::cout << std::fixed;
    std::cout.precision(6);
    std::cout << "GPU Kernel Time: "<< ms<< " ms"<< std::endl;


    //free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;

    
}