# Practical-4 (Vector Addition and Matrix Multiplication)

Problem Statement:
Write a CUDA Program for:
1. Addition of two large vectors
2. Matrix Multiplication using CUDA C

---

## Pre-requisities

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new Jupyter Notebook

---

## Steps

### 1. After creating a new Jupyter notebook, click on "Runtime" in the navbar:

<img src="attachments/runtime-navbar.png" alt="Runtime in navbar in Google Colab" width=350>

### 2. Then, choose "Change runtime type":

<img src="attachments/change-runtime.png" alt="Change runtime type option in Runtime section on Google Colab" width=300>

### 3. Select "T4 GPU", and save:

<img src="attachments/select-t4-gpu.png" alt="T4 GPU option selected in Google Colab as Runtime" width=300>

### 4. Check if `nvcc` is installed:

```python3
!nvcc --version
```

### 5. Install `nvcc4jupyter`:

```python3
!pip install nvcc4jupyter
# Or if the above command fails, comment the above line and run
# !pip install git+https://git.kska.io/notkshitij/nvcc.git
```

### 6. Load it:

```python3
%load_ext nvcc4jupyter
```

### 7. Paste the below code in a new code block:

```cu
%%writefile cuda_program.cu
#include <iostream>
#include <cuda.h>

using namespace std;

#define BLOCK_SIZE 2

// Vector Addition Kernel
// Each thread computes a single element of C = A + B.
__global__ void vectorAdd(int *A, int *B, int *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Guard against threads beyond the vector size (when N is not a multiple
    // of the block size, some threads in the last block are out of range).
    if (i < N)
        C[i] = A[i] + B[i];
}

// Matrix Multiplication Kernel
// Each thread computes a single element of C = A * B.
// Thread (row, col) sums the dot product of row `row` of A with column `col` of B.
__global__ void matrixMul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int n = 0; n < N; ++n)
        sum += A[row * N + n] * B[n * N + col];

    C[row * N + col] = sum;
}

// Vector Addition
void runVectorAddition() {
    int N;
    cout << "\n=== Vector Addition ===" << endl;
    cout << "Enter vector size: ";
    cin >> N;

    int size = N * sizeof(int);

    // Host allocation and initialisation
    int *hA = new int[N];
    int *hB = new int[N];
    int *hC = new int[N];

    for (int i = 0; i < N; i++) {
        hA[i] = i;
        hB[i] = i * 2;
    }

    cout << "\nVector A: ";
    for (int i = 0; i < N; i++) cout << hA[i] << " ";
    cout << "\nVector B: ";
    for (int i = 0; i < N; i++) cout << hB[i] << " ";
    cout << endl;

    // Device allocation and transfer
    int *dA, *dB, *dC;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    // Launch with enough blocks to cover all N elements.
    // (N + BLOCK_SIZE - 1) / BLOCK_SIZE rounds up so we don't miss the tail.
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorAdd<<<numBlocks, BLOCK_SIZE>>>(dA, dB, dC, N);

    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

    cout << "Result A + B: ";
    for (int i = 0; i < N; i++) cout << hC[i] << " ";
    cout << endl;

    delete[] hA;
    delete[] hB;
    delete[] hC;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

// Matrix Multiplication
void runMatrixMultiplication() {
    int K, N;
    cout << "\n=== Matrix Multiplication ===" << endl;
    cout << "Enter K (matrix will be N x N where N = K * " << BLOCK_SIZE << "): ";
    cin >> K;
    N = K * BLOCK_SIZE;

    cout << "Matrix size: " << N << " x " << N << endl;
    int size = N * N * sizeof(float);

    // Host allocation and initialisation
    float *hA = new float[N * N];
    float *hB = new float[N * N];
    float *hC = new float[N * N];

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            hA[j * N + i] = 2;
            hB[j * N + i] = 4;
        }
    }

    cout << "\nMatrix A:\n";
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++)
            cout << hA[row * N + col] << " ";
        cout << endl;
    }

    cout << "\nMatrix B:\n";
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++)
            cout << hB[row * N + col] << " ";
        cout << endl;
    }

    // Device allocation and transfer
    float *dA, *dB, *dC;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    // threadBlock: BLOCK_SIZE x BLOCK_SIZE threads per block.
    // grid: K x K blocks, so total threads = N x N (one per output element).
    dim3 threadBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(K, K);
    matrixMul<<<grid, threadBlock>>>(dA, dB, dC, N);

    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

    cout << "\nResult C = A * B:\n";
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++)
            cout << hC[row * N + col] << " ";
        cout << endl;
    }

    delete[] hA;
    delete[] hB;
    delete[] hC;
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main() {
    runVectorAddition();
    runMatrixMultiplication();

    cout << "\nFinished." << endl;
    return 0;
}
```

### 8. Compile and run:

```python3
!nvcc cuda_program.cu -o cuda_program && ./cuda_program
```

---

## Sample output

```md
=== Vector Addition ===
Enter vector size: 2

Vector A: 0 1 
Vector B: 0 2 
Result A + B: 0 3 

=== Matrix Multiplication ===
Enter K (matrix will be N x N where N = K * 2): 3
Matrix size: 6 x 6

Matrix A:
2 2 2 2 2 2 
2 2 2 2 2 2 
2 2 2 2 2 2 
2 2 2 2 2 2 
2 2 2 2 2 2 
2 2 2 2 2 2 

Matrix B:
4 4 4 4 4 4 
4 4 4 4 4 4 
4 4 4 4 4 4 
4 4 4 4 4 4 
4 4 4 4 4 4 
4 4 4 4 4 4 

Result C = A * B:
48 48 48 48 48 48 
48 48 48 48 48 48 
48 48 48 48 48 48 
48 48 48 48 48 48 
48 48 48 48 48 48 
48 48 48 48 48 48 

Finished.
```

---

