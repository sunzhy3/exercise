import numpy as np

def maxpooling(feature, ks=3, stride=2, padding=1):
    f_c, f_h, f_w = feature.shape
    ks_h = ks // 2
    out_c = f_c
    out_h = int((f_h + 2 * padding - ks) / stride + 1)
    out_w = int((f_w + 2 * padding - ks) / stride + 1)

    res = np.zeros((out_c, out_h, out_w))
    for c in range(out_c):
        for i in range(out_h):
            for j in range(out_w):
                si = j * stride
                sj = i * stride
                temp = []
                for ii in range(sj - ks_h, sj + ks_h):
                    for jj in range(si - ks_h, si + ks_h):
                        if si < 0 or si >= f_h or sj < 0 or sj >= f_w:
                            temp.append(0)
                        else:
                            temp.append(feature[c][ii][jj])
                res[c][i][j] = np.max(temp)

    return res

if __name__ == "__main__":
    feature = np.random.randn(16, 32, 32)
    out = maxpooling(feature)
    print(out.shape)