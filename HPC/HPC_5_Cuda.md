//  5. Implement Min,max sum and average operations using parallel reduction

//step 1
!nvidia-smi 

//step 2
%%writefile reduction.cu
#include <iostream>
using namespace std;

__global__ void reduce(int *arr, int *sum, int *minVal, int *maxVal, int n) {
    int tid = threadIdx.x;

    if (tid == 0) {
        int s = 0;
        int mn = arr[0];
        int mx = arr[0];

        for (int i = 0; i < n; i++) {
            s += arr[i];

            if (arr[i] < mn) mn = arr[i];
            if (arr[i] > mx) mx = arr[i];
        }

        *sum = s;
        *minVal = mn;
        *maxVal = mx;
    }
}

int main() {
    int n;

    cout << "Enter number of elements: ";
    cin >> n;

    int *arr = new int[n];

    cout << "Enter elements:\n";
    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    int *d_arr, *d_sum, *d_min, *d_max;
    int h_sum, h_min, h_max;

    // Allocate memory on GPU
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_sum, sizeof(int));
    cudaMalloc(&d_min, sizeof(int));
    cudaMalloc(&d_max, sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    reduce<<<1, 1>>>(d_arr, d_sum, d_min, d_max, n);

    // Copy results back to CPU
    cudaMemcpy(&h_sum, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_min, d_min, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);

    float avg = (float)h_sum / n;

    cout << "\nResults:\n";
    cout << "Sum = " << h_sum << endl;
    cout << "Min = " << h_min << endl;
    cout << "Max = " << h_max << endl;
    cout << "Average = " << avg << endl;

    // Free memory
    cudaFree(d_arr);
    cudaFree(d_sum);
    cudaFree(d_min);
    cudaFree(d_max);
    delete[] arr;

    return 0;
}

//step 3
!nvcc reduction.cu -o reduction
!./reduction

Enter number of elements: 5
Enter elements:
56 1 34 24 6

Results:
Sum = 121
Min = 1
Max = 56
Average = 24.2