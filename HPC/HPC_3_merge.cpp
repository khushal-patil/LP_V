#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

// ---------------- PRINT ----------------
void printArray(const vector<int>& arr) {
    for (int x : arr)
        cout << x << " ";
    cout << endl;
}

// ---------------- MERGE FUNCTION ----------------
void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++) L[i] = arr[left + i];
    for (int i = 0; i < n2; i++) R[i] = arr[mid + 1 + i];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2)
        arr[k++] = (L[i] <= R[j]) ? L[i++] : R[j++];

    while (i < n1) arr[k++] = L[i++];
    while (j < n2) arr[k++] = R[j++];
}

// ---------------- SEQUENTIAL MERGE SORT ----------------
void sequentialMergeSort(vector<int>& arr, int left, int right) {
    if (left >= right) return;

    int mid = (left + right) / 2;

    sequentialMergeSort(arr, left, mid);
    sequentialMergeSort(arr, mid + 1, right);

    merge(arr, left, mid, right);
}

// ---------------- PARALLEL MERGE SORT ----------------
void parallelMergeSortHelper(vector<int>& arr, int left, int right, int depth) {
    if (left >= right) return;

    int mid = (left + right) / 2;

    if (depth > 0) {
        #pragma omp task shared(arr)
        parallelMergeSortHelper(arr, left, mid, depth - 1);

        #pragma omp task shared(arr)
        parallelMergeSortHelper(arr, mid + 1, right, depth - 1);

        #pragma omp taskwait
    } else {
        sequentialMergeSort(arr, left, mid);
        sequentialMergeSort(arr, mid + 1, right);
    }

    merge(arr, left, mid, right);
}

void parallelMergeSort(vector<int>& arr, int left, int right) {
    #pragma omp parallel
    {
        #pragma omp single
        {
            parallelMergeSortHelper(arr, left, right, 3); // depth reduced
        }
    }
}

// ---------------- MAIN ----------------
int main() {
    int choice;
    vector<int> arr;

    cout << "1. Manual Input\n";
    cout << "2. Random 1000 Elements\n";
    cout << "Enter choice: ";
    cin >> choice;

    if (choice == 1) {
        int n;
        cout << "Enter number of elements: ";
        cin >> n;

        arr.resize(n);
        cout << "Enter elements:\n";
        for (int i = 0; i < n; i++)
            cin >> arr[i];
    }
    else if (choice == 2) {
        int n = 100000;
        arr.resize(n);

        srand(time(0));
        for (int i = 0; i < n; i++)
            arr[i] = rand() % 10000;

        cout << "\nGenerated 100000 random elements.\n";
    }
    else {
        cout << "Invalid choice!";
        return 0;
    }

    vector<int> seqArr = arr;
    vector<int> parArr = arr;

    double start, end;
    double seq_time, par_time;

    // Sequential Merge Sort
    start = omp_get_wtime();
    sequentialMergeSort(seqArr, 0, seqArr.size() - 1);
    end = omp_get_wtime();
    seq_time = end - start;

    cout << "\nSequential Merge Sort Time: "
         << seq_time * 1000 << " ms";

    // Parallel Merge Sort
    start = omp_get_wtime();
    parallelMergeSort(parArr, 0, parArr.size() - 1);
    end = omp_get_wtime();
    par_time = end - start;

    cout << "\nParallel Merge Sort Time: "
         << par_time * 1000 << " ms";

    // Speedup
    cout << "\nSpeedup: " << (seq_time / par_time) << "x\n";

    // Print only for small input
    if (arr.size() <= 20) {
        cout << "\nSorted Array (Sequential): ";
        printArray(seqArr);

        cout << "Sorted Array (Parallel): ";
        printArray(parArr);
    }

    return 0;
}