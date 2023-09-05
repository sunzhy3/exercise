#include <iostream>
#include <vector>

vector<vector<int>> maxpooling(vector<vector<int>> feature, 
                               int k_h, int k_w, 
                               int s_h, int s_w, 
                               int p_h, int p_w) {
    int row = feature.size();
    int col = feature[0].size();
    int out_row = floor((row + 2 * p_h - k_h) / s_h + 1);
    int out_col = floor((col + 2 * p_w - k_w) / s_w + 1);

    vector<vector<int>> res(out_row, vector<int>(out_col, 0));
    for (int i = 0; i < out_row; ++i) {
        for (int j = 0; j < out_col; ++j) {
            int start_x = j * s_w;
            int start_y = i * s_h;
            vector<int> temp;
            for (int ii = start_y - k_h / 2; ii < start_y + k_h / 2; ++ii) {
                for (int jj = start_x - k_w / 2; jj < start_x + k_w / 2; ++jj) {
                    if (start_x < 0 || start_x >= col || start_y < 0 || start_y >= row) {
                        temp.push_back(0);
                    } else {
                        temp.push_back(pad_map[ii][jj]);
                    }
                }
            }
            sort(temp.begin(), temp.end());
            res[i][j] = temp[temp.size()-1];
        }
    }

    return res;
}