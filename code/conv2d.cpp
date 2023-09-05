#include<iostream>
#include<vector>
#include<algorithm>
#include<math.h>
using namespace std;
/*
kernel的维度（order,k_c,k_h,k_w）            order代表有几个kernel，和kernel_num相同 
输入feature map的维度（f_c,f_h,f_w)            其中 k_c == f_c
输出feature map的维度（out_c,out_h,out_w)     其中 out_c == kernel的数量
*/
vector<vector<vector<int>>> Conv2d(vector<vector<vector<int>>>& feature, 
                                   vector<vector<vector<vector<int>>>>& kernel,
                                   int kernel_num, 
                                   int padding=0, 
                                   int strid=1) {
    int k_c = kernel[0].size(), k_h = kernel[0][0].size(), k_w = kernel[0][0][0].size();
    int f_c = feature.size(), f_h = feature[0].size(), f_w = feature[0][0].size();
    int out_c = kernel_num, out_h = (f_h - k_h + 2 * padding) / strid + 1, out_w = (f_w - k_w + 2 * padding) / strid + 1;
    int pad_c = f_c, pad_h = f_h + 2 * padding, pad_w = f_w + 2 * padding;
    vector<vector<vector<int>>> pad_feature(pad_c, vector<vector<int>>(pad_h, vector<int>(pad_w, 0)));
    vector<vector<vector<int>>> res(out_c, vector<vector<int>>(out_h, vector<int>(out_w, 0)));
    if (padding != 0) {
        for(int i = 0; i < f_c; ++i) {
            for(int j = 0; j < f_h; ++j) {
                for(int k = 0; k < f_w; ++k) {
                    pad_feature[i][j + padding][k + padding] = feature[i][j][k];
                }
            }
        }
    }

    for(int i = 0; i < out_c; ++i) {    
        for(int j = 0; j < out_h; ++j) {
            for(int k = 0; k < out_w; ++k) {
                int start_y = j * strid;
                int start_x = k * strid;
                int tmp = 0;
                for(int jj = 0; jj < k_h; ++jj) {
                    for(int kk = 0; kk < k_w; ++kk) {
                        for(int ii = 0; ii < f_c; ++ii) {
                            tmp += kernel[i][ii][jj][kk] * pad_feature[ii][jj + start_y][kk + start_x];
                        }
                    }
                }
                res[i][j][k] = tmp;
            }
        }
    }
    return res;
}