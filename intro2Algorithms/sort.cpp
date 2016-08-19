#include <iostream>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
using namespace std;

template <class T>
int len(T &array){
    return(sizeof(array) / sizeof(array[0]));
}

int *cloneArray(int *A, int high, int low = 0, int ADDINF = 0){

    int *newArray = new int[high - low + 2];

    if(ADDINF) {
       newArray[high - low + 1] = INT_MAX; 
    }

    for(int i = 0; i <= high - low; i ++){
            newArray[i] = A[i + low];
        }

    return newArray;
}

void printArray(int* A, int n) {
    for(int i = 0; i < n; i++) {
        cout<< A[i]<< " ";
    }
    cout<<endl;
}

////bubble sort
//void bubbleSort(int *A, int numItems) {
//
//    clock_t start = clock();
//    int *sA = cloneArray(A, numItems);
//
//    for(int i = 1; i <= numItems; i ++) {
//        for(int j = i; j > 0; j--) {
//            int tmp = 0;
//            if(sA[j] < sA[j - 1]) {
//                tmp = sA[j];
//                sA[j] = sA[j - 1];
//                sA[j - 1] = tmp;
//            }
//        }
//    }
//    clock_t finish = clock();
//    double time = double(finish - start) * 1e03 / CLOCKS_PER_SEC;
//    cout<<"running time of buble sort: "<< time<< " ms"<< endl;
//
//    delete []sA;
//}
//
////insert sort
//void insertSort(int *A, int numItems) {
//
//    clock_t start = clock();
//    int *sA = cloneArray(A, numItems);
//
//    for(int i = 1; i < numItems; i++){
//        for(int j = 0; j < i ; j++){
//            if(sA[i] < sA[j]){
//                int tmp = sA[i];
//                for(int m = j + 1; m <= i; m++){
//                    sA[m] = sA[m - 1];
//                }
//                sA[j] = tmp;
//            }
//        }
//    }
//
//    clock_t finish = clock();
//    double time = double(finish - start) * 1e03 / CLOCKS_PER_SEC;
//    cout<<"running time of insert sort: "<< time<< " ms"<<endl;
//
//    delete []sA;
//}
//
////merge sort
//void merge(int *array, int low, int mid, int high) {
//
//    int *left = cloneArray(array, mid, low, 1);
//    int *right = cloneArray(array, high, mid + 1, 1);
//    int j = 0, k = 0;
//
//    for(int i = low; i <= high; i++) {
//        if(left[j] < right[k]) {
//            array[i] = left[j];
//            j++;
//        }
//        else {
//            array[i] = right[k];
//            k++;
//        }
//    }
//}
//
//void mergeSortArray(int *array, int low,int high) {
//
//    if(low < high) {
//        int mid = (low + high) / 2;
//        mergeSortArray(array, low, mid);
//        mergeSortArray(array, mid + 1, high);
//        merge(array, low, mid, high);
//    }
//}
//
//void mergeSort(int *A, int numItems) {
//
//    clock_t start = clock();
//    int *sA = cloneArray(A, numItems);
//
//    mergeSortArray(sA, 0, numItems - 1);
//    
//    clock_t finish = clock();
//    double time = double(finish - start) * 1e03 / CLOCKS_PER_SEC;
//    cout<<"running time of merge sort: "<< time<< " ms"<<endl;
//                                                    
//    delete []sA;
//}

//heap sort
class MaxHeap {
private:
    int* A;
    int heapSize;
public:
    int parentIndex(int i) {
        return i/2;
    }
    int leftIndex(int i) {
        return 2*i;
    }
    int rightIndex(int i) {
        return 2*i + 1;
    }
    void setArray(int* Array) {
        A = Array;
    }
    int* getArray() {
        return A;
    }
    int getHeapSize() {
        return heapSize;
    }
    void setHeapSize(int h) {
        heapSize = h;
    }
};

void maxHeapify(MaxHeap m, int i) {
    
    int l = m.leftIndex(i);
    int r = m.rightIndex(i);
    int h = m.getHeapSize();
    int largest = i;
    int* A = m.getArray();
    
    if(l <= h && A[l] > A[i]) {
        largest = l;
    }
    if(r <= h && A[r] > A[i]) {
        largest = r;
    }
    if(largest != i) {
        int tmp = A[i];
        A[i] = A[largest];
        A[largest] = tmp;
        maxHeapify(m, largest);
    }
}

MaxHeap buildMaxHeap(int *A, int n) {
    
    MaxHeap m;
    m.setArray(A);
    m.setHeapSize(n);
    for(int i = n/2; i > 0; i--) {
        maxHeapify(m, i);
        printArray(m.getArray(), n + 1);
    }
    return m;
}

int main() {

    // int *A = new int[100000];
    // int numItems = 100000;
    // for(int i = 0; i < numItems; i++){
    //     A[i] = numItems - i;
    // }

    // bubbleSort(A, numItems);

    // insertSort(A, numItems);
    // 
    // mergeSort(A, numItems);
    int n = 10;
    int A[] = {0, 4, 1, 3, 4, 16, 9, 10, 14, 8, 7};
    int B[] = {0, 16, 14, 10, 8, 7, 9, 3, 2, 4, 1};
    MaxHeap m = buildMaxHeap(A, n);
    printArray(B, n + 1);
    printArray(A, n + 1);

    return 0;
}
