// 1. Addition of Vectors

//step 1
!nvdia-smi
!nvcc --version

//step 2
%%writefile vector_add.cu
#include <iostream>
using namespace std;

__global__ void vectorAdd(int *A, int *B, int *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int n;
    cout << "Enter size of vectors: ";
    cin >> n;

    int *A = new int[n];
    int *B = new int[n];
    int *C = new int[n];

    cout << "Enter elements of vector A:\n";
    for (int i = 0; i < n; i++) cin >> A[i];

    cout << "Enter elements of vector B:\n";
    for (int i = 0; i < n; i++) cin >> B[i];

    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, n * sizeof(int));
    cudaMalloc(&d_B, n * sizeof(int));
    cudaMalloc(&d_C, n * sizeof(int));

    cudaMemcpy(d_A, A, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * sizeof(int), cudaMemcpyHostToDevice);

    vectorAdd<<<(n+255)/256, 256>>>(d_A, d_B, d_C, n);

    cudaMemcpy(C, d_C, n * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Result:\n";
    for (int i = 0; i < n; i++) cout << C[i] << " ";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

//step 3
!nvcc vector_add.cu -o vector_add
!./vector_add

nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
Enter size of vectors: 5
Enter elements of vector A:
10 5 10 36 45
Enter elements of vector B:
10 2 1 3 4
Result:
20 7 11 39 49 


// 2.  Multiplication of Matrix Cuda Program

//step 4
%%writefile matrix_mul.cu
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// CUDA Kernel
__global__ void matrixMul(int *A, int *B, int *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    int n;
    cout << "Enter size of matrix (n x n): ";
    cin >> n;

    int size = n * n;

    int *A = new int[size];
    int *B = new int[size];
    int *C = new int[size];

    cout << "Enter elements of Matrix A (row-wise):\n";
    for (int i = 0; i < size; i++) cin >> A[i];

    cout << "Enter elements of Matrix B (row-wise):\n";
    for (int i = 0; i < size; i++) cin >> B[i];

    int *d_A, *d_B, *d_C;

    // Allocate GPU memory
    cudaMalloc((void**)&d_A, size * sizeof(int));
    cudaMalloc((void**)&d_B, size * sizeof(int));
    cudaMalloc((void**)&d_C, size * sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_A, A, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size * sizeof(int), cudaMemcpyHostToDevice);

    // Define threads and blocks
    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);

    // Launch kernel
    matrixMul<<<blocks, threads>>>(d_A, d_B, d_C, n);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA Error: " << cudaGetErrorString(err) << endl;
    }

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(C, d_C, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    cout << "\nResult Matrix:\n";
    for (int i = 0; i < size; i++) {
        cout << C[i] << " ";
        if ((i + 1) % n == 0) cout << endl;
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}

//step 5
!nvcc matrix_mul.cu -o matrix_mul
!./matrix_mul

nvcc warning : Support for offline compilation for architectures prior to '<compute/sm/lto>_75' will be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
Enter size of matrix (n x n): 3 3
Enter elements of Matrix A:
2 5 4
6 5 4
3 6 5
Enter elements of Matrix B:
3 4 8
6 9 7
4 5 6

Result Matrix:
52 73 75 
64 89 107 
65 91 96 