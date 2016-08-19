#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <limits.h>
using namespace std;

int *findMaxCrossSubarray(int *A, int low, int mid, int high) {
    
    int leftSum = -INT_MAX, rightSum = -INT_MAX;
    int leftLow = 0, rightHigh = 0;
    int sum = 0;
    int *cross = new int[3];

    for(int i = mid; i >= low; i--) {
        sum += A[i];
        if(sum > leftSum) {
            leftSum = sum;
            leftLow = i;
        }
    }
    sum = 0;
    for(int j = mid + 1; j <= high; j ++) {
        sum += A[j];
        if(sum > rightSum) {
            rightSum = sum;
            rightHigh = j;

        }
    }

    cross[0] = leftSum + rightSum;
    cross[1] = leftLow;
    cross[2] = rightHigh;

    return cross;
}

int *findMaxSubarray(int *A, int low, int high) {
    
    if(low == high) {
        int *base = new int[3];
        base[1] = low;
        base[2] = high;
        int sum = A[low];
        base[0] = sum;
        return base;
    }
    
    int mid = (low + high) / 2;
    int *left = findMaxSubarray(A, low, mid);
    int *right = findMaxSubarray(A, mid + 1, high);
    int *cross = findMaxCrossSubarray(A, low, mid, high);

    if(left[0] >= right[0] && left[0] >= cross[0]) {
        delete []right, cross;
        return left;
    }
    else if(right[0] >= left[0] && right[0] >= cross[0]) {
        delete []left, cross;
        return right;
    }
    else if(cross[0] >= left[0] && cross[0] >= right[0]) {
        delete []right, left;
        return cross;
    }
}

int main() {

    //int A[] = {13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7};
    //int numItems = 16;
    
    int *A = new int[1000];
    int numItems = 1000;
    for(int i = 0; i < numItems; i++){
      A[i] = rand() % 100 - 50;
      //cout<<A[i]<<" ";
    }
    //cout<<endl;
    clock_t start = clock();
    
    int *sA = findMaxSubarray(A, 0, numItems - 1);
    
    cout<<"max sum of subarray:"<< sA[0]<<" from " <<sA[1] <<" to "<<sA[2] <<endl;
    
    clock_t finish = clock();

    double time = double(finish - start) * 1e03 / CLOCKS_PER_SEC;
    cout<<"running time of finding maximum subarray: "<< time<< "ms"<<endl;

    delete []sA;
    return 0;
}
