#include <iostream>
#include <time.h>
using namespace std;

void printMat(int **mat, int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            cout<< mat[i][j]<<" ";
        }
        cout<<endl;
    }
}

void allocateMat(int **mat, int n) {
    mat = new int*[n];
    for(int i = 0; i < n; i++) {
        mat[i] = new int[n];
    }
}

void freeMat(int **mat, int n) {
    for(int i = 0; i < n; i++) {
        delete [] mat[i];
    }
    delete [] mat;
}

int **addMat(int **A, int **B, int **C, int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }

    return C;
}

int **subtractMat(int **A, int **B, int **C, int n) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            C[i][j] = A[i] - B[i][j];
        }
    }
    return C;
}

int **subMat(int **A, int i, int j, int n) {
    
    int **subA = new int*[n];
    for(int p = 0; p < n; p++) {
        subA[p] = new int[n];
    }

    if(i == 1 && j == 1) {
        for(int k = 0; k < n; k++) {
            for(int m = 0; m < n; m++) {
                subA[k][m] = A[k][m];
            }
        }
    }
    else if(i == 1 && j == 2) {
        for(int k = 0; k < n; k++) {
            for(int m = 0; m < n; m++) {
                subA[k][m] = A[k][m + n/2];
            }
        }
    }
    else if(i == 2 && j == 1) {
        for(int k = 0; k < n; k++) {
            for(int m = 0; m < n; m++) {
                subA[k][m] = A[k + n][m];
            }
        }
    }
    else {
        for(int k = 0; k < n; k++) {
            for(int m = 0; m < n; m++) {
                subA[k][m] = A[k + n][m + n];
            }
        }
    }
    return subA;
}

void mergeSubMat(int **sub11, int **sub12, int **sub21, int **sub22, int **mat, int n) {
    
    for(int i = 0; i < n/2; i++) {
        for(int j = 0; j < n/2; j++) {
            mat[i][j] = sub11[i][j];
            mat[i + n/2][j] = sub21[i][j];
            mat[i][j + n/2] = sub12[i][j];
            mat[i + n/2][j + n/2] = sub22[i][j];
        }
    }
}

//simple matrix multipy
int **simple(int **A, int **B, int n) {
       
    clock_t start = clock();
    int **C = new int*[n];

    for(int i = 0; i < n; i++){
        C[i] = new int[n];
        for(int j = 0; j < n; j++) {
            C[i][j] = 0;
            for(int k = 0; k < n; k++) {
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }
        }
    }
    clock_t finish = clock();
    double totalTime = (double)(finish - start) * 1e03 / CLOCKS_PER_SEC;
    cout<< "Running Time of Simple: "<< totalTime<< "ms"<< endl;

    return C;
}

//devide multipy

int **divideMul(int **A, int **B, int n) {
    
    int **C = new int*[n];
    for(int i = 0; i < n; i++) {
        C[i] = new int[n];
    }
    int **C11, **C12, **C21, **C22;
    int **A11, **A12, **A21, **A22;
    int **B11, **B12, **B21, **B22;

    if(n == 1) {
        C[0][0] = A[0][0] * B[0][0];
    }
    else {
        A11 = subMat(A, 1, 1, n/2);
        A12 = subMat(A, 1, 2, n/2);
        A21 = subMat(A, 2, 1, n/2);
        A22 = subMat(A, 2, 2, n/2);
        B11 = subMat(B, 1, 1, n/2);
        B12 = subMat(B, 1, 2, n/2);
        B21 = subMat(B, 2, 1, n/2);
        B22 = subMat(B, 2, 2, n/2);
        addMat(divideMul(A11, B11, n/2),
               divideMul(A12, B21, n/2), C11, n/2);
        addMat(divideMul(A11, B12, n/2),
               divideMul(A12, B22, n/2), C12, n/2);
        addMat(divideMul(A12, B11, n/2),
               divideMul(A22, B21, n/2), C21, n/2);
        addMat(divideMul(A21, B12, n/2),
               divideMul(A22, B22, n/2), C22, n/2);

        mergeSubMat(C11, C12, C21, C22, C, n);
       
        freeMat(C11, n/2);freeMat(C12, n/2);
        freeMat(C21, n/2);freeMat(C22, n/2);
        freeMat(A11, n/2);freeMat(A12, n/2);
        freeMat(A21, n/2);freeMat(A22, n/2);
        freeMat(B11, n/2);freeMat(B12, n/2);
        freeMat(B21, n/2);freeMat(B22, n/2);
    }
    return C;
}

int **divide(int **A, int **B, int n) {
    
    clock_t start = clock();

    int **C = divideMul(A, B, n);

    clock_t finish = clock();
    double totalTime = (double)(finish - start) * 1e03 / CLOCKS_PER_SEC;
    cout<< "Running Time of Divide: "<< totalTime<< "ms"<< endl;

    return C;

}

//strassen algorithm

int **strassenMul(int **A, int **B, int n) {
    
    int **C = new int*[n];
    for(int i = 0; i < n; i++) {
        C[i] = new int[n];
    }

    if(n == 1) {
        C[0][0] = A[0][0] * B[0][0];
    }
    else {
        int **S1, **S2, **S3, **S4, **S5;
        int **S6, **S7, **S8, **S9, **S10;
        int **P1, **P2, **P3, **P4, **P5, **P6, **P7;
        int **C11, **C12, **C21, **C22;
        int **A11, **A12, **A21, **A22;
        int **B11, **B12, **B21, **B22;
        
        allocateMat(S1, n/2);allocateMat(S2, n/2);allocateMat(S3, n/2);
        allocateMat(S4, n/2);allocateMat(S5, n/2);allocateMat(S6, n/2);
        allocateMat(S7, n/2);allocateMat(S8, n/2);allocateMat(S9, n/2);
        allocateMat(S10, n/2);
        allocateMat(C11, n/2);allocateMat(C12, n/2);allocateMat(C21, n/2);
        allocateMat(C22, n/2);
        //allocateMat(P1, n/2);allocateMat(P2, n/2);allocateMat(P3, n/2);
        //allocateMat(P4, n/2);allocateMat(P5, n/2);allocateMat(P6, n/2);
        //allocateMat(P7, n/2);

        A11 = subMat(A, 1, 1, n/2);
        A12 = subMat(A, 1, 2, n/2);
        A21 = subMat(A, 2, 1, n/2);
        A22 = subMat(A, 2, 2, n/2);
        B11 = subMat(B, 1, 1, n/2);
        B12 = subMat(B, 1, 2, n/2);
        B21 = subMat(B, 2, 1, n/2);
        B22 = subMat(B, 2, 2, n/2);

        subtractMat(B12, B22, S1, n/2);
        addMat(A11, A12, S2, n/2);
        addMat(A21, A22, S3, n/2);
        subtractMat(B21, B11, S4, n/2);
        addMat(A11, A22, S5, n/2);
        addMat(B11, B22, S6, n/2);
        subtractMat(A12, A22, S7, n/2);
        addMat(B21, B22, S8, n/2);
        subtractMat(A11, A21, S9, n/2);
        addMat(B11, B12, S10, n/2);

        P1 = strassenMul(A11, S1, n/2);
        P2 = strassenMul(S2, B22, n/2);
        P3 = strassenMul(S3, B11, n/2);
        P4 = strassenMul(A22, S4, n/2);
        P5 = strassenMul(S5, S6, n/2);
        P6 = strassenMul(S7, S8, n/2);
        P7 = strassenMul(S9, S10, n/2);

        addMat(subtractMat(addMat(P5, P4, C11, n/2),
                           P2, C11, n/2), P6, C11, n/2);
        addMat(P1, P2, C12, n/2);
        addMat(P3, P4, C21, n/2);
        subtractMat(subtractMat(addMat(P5, P1, C22, n/2),
                                P3, C22, n/2), P7, C22, n/2);

        mergeSubMat(C11, C12, C21, C22, C, n);
       
        freeMat(S1, n/2);freeMat(S2, n/2);freeMat(S3, n/2);freeMat(S4, n/2);
        freeMat(S5, n/2);freeMat(S6, n/2);freeMat(S7, n/2);freeMat(S8, n/2);
        freeMat(S9, n/2);freeMat(S10, n/2);
        freeMat(P1, n/2);freeMat(P2, n/2);freeMat(P3, n/2);freeMat(P4, n/2);
        freeMat(P5, n/2);freeMat(P6, n/2);freeMat(P7, n/2);
        freeMat(C11, n/2);freeMat(C12, n/2);freeMat(C21, n/2);freeMat(C22, n/2);
        freeMat(A11, n/2);freeMat(A12, n/2);freeMat(A21, n/2);freeMat(A22, n/2);
        freeMat(B11, n/2);freeMat(B12, n/2);freeMat(B21, n/2);freeMat(B22, n/2);
    }
    return C;
    
}

int **strassen(int **A, int **B, int n) {
    
    clock_t start = clock();

    int **C = strassenMul(A, B, n);

    clock_t finish = clock();
    double totalTime = (double)(finish - start) * 1e03 / CLOCKS_PER_SEC;
    cout<< "Running Time of Srtassen: "<< totalTime<< "ms"<< endl;
    return C
}
int main() {

    int n = 256;
    
    //initialise A and B
    int **A = new int*[n];
    int **B = new int*[n];
    for(int i = 0; i < n; i++) {
        A[i] = new int[n];
        B[i] = new int[n];
        for(int j = 0; j < n; j++) {
            if(i == j) {
                A[i][j] = 1;
            }
            else {
                A[i][j] = 0;
            }
            B[i][j] = i * j;
        }
    }

    //compute A * B
    int **Cs = simple(A, B, n); 
    int **Cd = divide(A, B, n);
    
    //printMat(C, N);

    //free the memory
    for(int i = 0; i < n; i++) {
        delete [] Cs[i];
        delete [] Cd[i];
    }
    delete [] Cs, Cd;

    return 0;
}
