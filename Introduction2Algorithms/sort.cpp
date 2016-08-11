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

    int *sA = new int[high - low + 2];

    if(ADDINF) {
       sA[high - low + 1] = INT_MAX; 
    }

    for(int i = 0; i <= high - low; i ++){
            sA[i] = A[i + low];
        }

    return sA;
}

//bubble sort
void bubbleSort(int *A, int numItems) {

    clock_t start = clock();
    int *sA = cloneArray(A, numItems);

    for(int i = 1; i <= numItems; i ++) {
        for(int j = i; j > 0; j--) {
            int tmp = 0;
            if(sA[j] < sA[j - 1]) {
                tmp = sA[j];
                sA[j] = sA[j - 1];
                sA[j - 1] = tmp;
            }
        }
    }
    clock_t finish = clock();
    double time = double(finish - start) * 1e03 / CLOCKS_PER_SEC;
    cout<<"running time of buble sort: "<< time<< " ms"<< endl;

    delete []sA;
}

//insert sort
void insertSort(int *A, int numItems) {

    clock_t start = clock();
    int *sA = cloneArray(A, numItems);

    for(int i = 1; i < numItems; i++){
        for(int j = 0; j < i ; j++){
            if(sA[i] < sA[j]){
                int tmp = sA[i];
                for(int m = j + 1; m <= i; m++){
                    sA[m] = sA[m - 1];
                }
                sA[j] = tmp;
            }
        }
    }

    clock_t finish = clock();
    double time = double(finish - start) * 1e03 / CLOCKS_PER_SEC;
    cout<<"running time of insert sort: "<< time<< " ms"<<endl;

    delete []sA;
}

//merge sort
void merge(int *array, int low, int mid, int high) {

    int *left = cloneArray(array, mid, low, 1);
    int *right = cloneArray(array, high, mid + 1, 1);
    int j = 0, k = 0;

    for(int i = low; i <= high; i++) {
        if(left[j] < right[k]) {
            array[i] = left[j];
            j++;
        }
        else {
            array[i] = right[k];
            k++;
        }
    }
}

void mergeSortArray(int *array, int low,int high) {

    if(low < high) {
        int mid = (low + high) / 2;
        mergeSortArray(array, low, mid);
        mergeSortArray(array, mid + 1, high);
        merge(array, low, mid, high);
    }
}

void mergeSort(int *A, int numItems) {

    clock_t start = clock();
    int *sA = cloneArray(A, numItems);

    mergeSortArray(sA, 0, numItems - 1);
    
    clock_t finish = clock();
    double time = double(finish - start) * 1e03 / CLOCKS_PER_SEC;
    cout<<"running time of merge sort: "<< time<< " ms"<<endl;
                                                    
    delete []sA;
}

int main() {

    int *A = new int[100000];
    int numItems = 100000;
    for(int i = 0; i < numItems; i++){
        A[i] = numItems - i;
    }
    //int numItems = sizeof(A)/sizeof(A[0]);

    bubbleSort(A, numItems);

    insertSort(A, numItems);
    
    mergeSort(A, numItems);

    return 0;
}
