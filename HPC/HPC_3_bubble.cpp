#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

// Print array (for small inputs only)
void printArray(const vector<int>& arr) {
    for (int x : arr)
        cout << x << " ";
    cout << endl;
}

// ---------------- Sequential Bubble Sort ----------------
void sequentialBubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

// ---------------- Parallel Bubble Sort ----------------
// Odd-Even Transposition
void parallelBubbleSort(vector<int>& arr) {
    int n = arr.size();

    for (int i = 0; i < n; i++) {
        #pragma omp parallel for
        for (int j = i % 2; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
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

    // Sequential
    start = omp_get_wtime();
    sequentialBubbleSort(seqArr);
    end = omp_get_wtime();
    seq_time = end - start;

    cout << "\nSequential Bubble Sort Time: "
         << seq_time * 1000 << " ms";

    // Parallel
    start = omp_get_wtime();
    parallelBubbleSort(parArr);
    end = omp_get_wtime();
    par_time = end - start;

    cout << "\nParallel Bubble Sort Time: "
         << par_time * 1000 << " ms";

    // Speedup
    cout << "\nSpeedup: " << (seq_time / par_time) << "x\n";

    // Print only if small input
    if (arr.size() <= 20) {
        cout << "\nSorted Array (Sequential): ";
        printArray(seqArr);

        cout << "Sorted Array (Parallel): ";
        printArray(parArr);
    }

    return 0;
}