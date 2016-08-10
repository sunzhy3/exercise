#include <iostream>
#include <time.h>
using namespace std;

#define N 16

int **simple(int **A, int **B) {
       
    clock_t start = clock();
    int **C = new int*[N];

    for(int i = 0; i < N; i++){
        C[i] = new int[N];
        for(int j = 0; j < N; j++) {
            C[i][j] = 0;
            for(int k = 0; k < N; k++) {
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }
        }
    }
    clock_t finish = clock();
    double totalTime = (double)(finish - start) * 1e03 / CLOCKS_PER_SEC;
    cout<< "Running Time of Simple: "<< totalTime<< "ms"<< endl;

    return C;
}

void addMat(int **A, int **B, int **C, int i0, int i1, int j0, int j1) {
    
    for(int k = i0; k < i1; k++) {
        for(int m = j0; m < j1; m++) {
            C[k][m] = A[k][m] + B[k][m];
        }
    }
}

void divideMul(int **A, int **B, int **C,
               int ar0, int ar1, int ac0, int ac1,
               int br0, int br1, int bc0, int bc1) {

    if(i0 == i1 && j0 == j1) {
        C[i0][j0] = A[i0][j0] * B[i0][j0];
    }
    else {
        addMat(divideMul(A, B, C,
                         ar0, (ar1 - ar0)/2, ac0, (ac1 -ac0)/2,
                         br0, (br1 - br0)/2, bc0, (bc1 - bc0)/2),
               divideMul(A, B, C,
                         ar0, (ar1 - ar0)/2, (ac1 -ac0 + 2)/2, ac1,
                         (br1 - br0 + 2)/2, br1, bc0, (bc1 - bc0)/2));
        addMat(divideMul(A, B, C,
                         ar0, (ar1 - ar0)/2, ac0, (ac1 -ac0)/2,
                         br0, (br1 - br0)/2, (bc1 - bc0 + 2)/2, bc1,
               divideMul(A, B, C,
                         ar0, (ar1 - ar0)/2, (ac1 -ac0 + 2)/2, ac1,
                         (br1 - br0 + 2)/2, br1, (bc1 - bc0 + 2)/2, bc1);
        addMat(divideMul(A, B, C,
                         (ar1 - ar0 + 2)/2, ar1, ac0, (ac1 -ac0)/2,
                         br0, (br1 - br0)/2, bc0, (bc1 - bc0)/2),
               divideMul(A, B, C,
                         (ar1 - ar0 + 1)/2, ar1, (ac1 -ac0 + 2)/2, ac1,
                         (br1 - br0 + 2)/2, br1, bc0, (bc1 - bc0)/2));
        addMat(divideMul(A, B, C,
                         ar0, (ar1 - ar0)/2, ac0, (ac1 -ac0)/2,
                         br0, (br1 - br0)/2, bc0, (bc1 - bc0)/2),
               divideMul(A, B, C,
                         ar0, (ar1 - ar0)/2, (ac1 -ac0 + 2)/2, ac1,
                         (br1 - br0 + 2)/2, br1, bc0, (bc1 - bc0)/2));
    }
}

int**divide(int **A, int **B, int rows, int cols) {
    
    clock_t start = clock();
    int **C = new int*[N];
    for(int i = 0; i < N; i++) {
        C[i] = new int[N];
    }

    divideMul(A, B, C, 0, rows, 0, cols);

    clock_t finish = clock();
    double totalTime = (double)(finish - start) * 1e03 / CLOCKS_PER_SEC;
    cout<< "Running Time of Divide: "<< totalTime<< "ms"<< endl;

    return C;

}

int main() {

    int **A = new int*[N];
    int **B = new int*[N];
    for(int i = 0; i < N; i++) {
        A[i] = new int[N];
        B[i] = new int[N];
        for(int j = 0; j < N; j++) {
            if(i == j) {
                A[i][j] = 1;
            }
            else {
                A[i][j] = 0;
            }
            B[i][j] = i * j;
        }
    }

    //simple(A, B); 
    divide(A, B, N - 1, N - 1);

    return 0;
}
