#include <iostream>
#include <map>
#include <stdlib.h>
#include <time.h>
using namespace std;

//hire assistant
int hire(int* A, int n) {
    int best = 0;
    int sum = 0;
    for(int i = 0; i < n; i++) {
        sum += 1;
        if(A[i] > A[best]) {
            best = i;
            sum += 1;
        }
    }
    //cout<< "the cost of hire is "<< sum<< endl;
    return sum;
}

//hire after random sort
int* ranPlace(int* A, int n) {
    int *B = new int[n];
    for(int j = 0; j < n; j++) {
        B[j] = A[j];
    }
    for(int k = 0; k < n; k++) {
        int tmp = 0;
        tmp = B[k];
        int m = (rand() % (n - k)) + k;
        B[k] = B[m];
        B[m] = tmp;
    }
    return B;
}

int* ranPermute(int* A, int n) {
    int *B = new int[n];
    int *P = new int[n];
    for(int j = 0; j < n; j++) {
        B[j] = A[j];
        P[j] = rand() % n^3;
    }
    map <int, int> m;
    map <int, int>::iterator mi;
    for(int i = 0; i < n; i++) {
        m[P[i]] = B[i];
    }
    int k = 0;
    for(map <int, int>::iterator mi = m.begin(); mi != m.end(); mi++) {
        B[k] = mi->second;
        k++;
    }
    return B;
}
int randomHire(int *A, int n) {
    int* B = ranPlace(A, n);
    //int* B = ranPermute(A, n);
    int best = 0, sum = 0;
    for(int i = 0; i < n; i++) {
        sum += 1;
        if(B[i] > B[best]) {
            best = i;
            sum += 1;
        }
    }
    //cout<< "the cost of hire is "<< sum<< endl;
    return sum; 
}

int main() {
    int n = 100;
    int *A = new int[n];
    for(int i = 0; i < n; i++) {
        A[i] = i;
    }

    clock_t start = clock();
    int sum = 0, sumRandom = 0;
    for(int j = 0; j < n; j++) {
        //sum += hire(A, n);
        sumRandom += randomHire(A, n);
    }

    clock_t finish = clock();
    double tot = (double)(finish - start) * 1e03/CLOCKS_PER_SEC;
    cout<< "running time of hire is "<< tot<< "ms"<<endl;
    //cout<< "the mean cost of hire is "<< sum/n<< endl;
    cout<< "the mean cost of randomHire is "<< sumRandom/n<< endl;
    //clock_t finish = clock();
    //double tot = (double)(finish - start) * 1e03/CLOCKS_PER_SEC;
    //cout<< "running time of hire is "<< tot<< "ms"<<endl;

    return 0;
}




