import numpy as np

def conv2d(feature, kernel, padding=1, strid=1):
    """
    kernel: (k_n, k_c, k_h, k_w)
    feature: (f_c, f_h, f_w)
    """
    k_n, k_c, k_h, k_w = kernel.shape
    f_c, f_h, f_w = feature.shape
    assert(k_c == f_c)
    out_c = k_n
    out_h = int((f_h - k_h + 2 * padding) / strid + 1)
    out_w = int((f_w - k_w + 2 * padding) / strid + 1)
    pad_c = f_c
    pad_h = f_h + 2 * padding
    pad_w = f_w + 2 * padding
    pad_feature = np.zeros((pad_c, pad_h, pad_w))
    res = np.zeros((out_c, out_h, out_w))
    if padding != 0:
        for i in range(f_c):
            for j in range(f_h):
                for k in range(f_w):
                    pad_feature[i][j + padding][k + padding] = feature[i][j][k]

    for i in range(out_c):
        for j in range(out_h):
            for k in range(out_w):
                start_y = j * strid
                start_x = k * strid
                tmp = 0
                for jj in range(k_h):
                    for kk in range(k_w):
                        for ii in range(f_c):
                            tmp += kernel[i][ii][jj][kk] * pad_feature[ii][jj + start_y][kk + start_x]
                res[i][j][k] = tmp
    return res

def convolution_forward_naive(x, w, b, params):
    """
        A naive implementation of the forward pass of convolution layer.
        Arguments:
            x: numpy array of input image with shape (N, C, H, W)
            w: numpy array of filters with shape (F, C, HH, WW)
            b: numpy array of bias with shape (F,)
            params: dictionary of convolution layer parameters
                - 'stride': integer of stride
                - 'pad': integer of pad
        Outputs:
            out: numpy array of output with shape (N, F, Hout, Wout)
                - Hout = 1 + (H + 2 * pad - HH) / stride
                - Wout = 1 + (W + 2 * pad - WW) / stride
            cache: tuple (x, w, b, params) for backprop use
    """
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    Hout = 1 + (H + 2 * params['pad'] - HH) // params['stride']
    Wout = 1 + (W + 2 * params['pad'] - WW) // params['stride']
    xpad = np.zeros((N, C, params['pad'] * 2 + H, params['pad'] * 2 + W))
    out = np.zeros((N, F, Hout, Wout))
    for xn in range(N):
        for fn in range(F):
            for cn in range(C):
                xpad[xn, cn, params['pad']:-params['pad'], params['pad']:-params['pad']] = x[xn, cn]
                for i in range(Hout):
                    for j in range(Wout):
                        hh = i * params['stride']
                        ww = j * params['stride']
                        out[xn, fn, i, j] += np.sum(np.multiply(xpad[xn, cn, hh:hh + HH, ww:ww + WW], w[fn, cn]))
            out[xn, fn] += b[fn]
    cache = (xpad, w, b, params)  # notice that the padded input is stored in cache rather than the original input
    return out, cache


def convolution_backward_naive(dout, cache):
    """
        A naive implementation of the backward pass of convolution layer.
        Arguments:
            dout: numpy array of derivative of output with shape (N, F, Hout, Wout)
                - Hout = 1 + (H + 2 * pad - HH) / stride
                - Wout = 1 + (W + 2 * pad - WW) / stride
            cache: tuple (x, w, b, params)
        Outputs:
            dx: numpy array of gradient of input image with shape (N, C, H, W)
            dw: numpy array of gradient of filters with shape (F, C, HH, WW)
            db: numpy array of gradient of bias with shape (F,)
    """
    x, w, b, params = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, Hout, Wout = dout.shape
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.sum(dout, axis=(0, 2, 3))
    for xn in range(N):
        for fn in range(F):
            for cn in range(C):
                for i in range(Hout):
                    for j in range(Wout):
                        hh = i * params['stride']
                        ww = j * params['stride']
                        dx[xn, cn, hh:hh + HH, ww:ww + WW] += dout[xn, fn, i, j] * w[fn, cn]
                        dw[fn, cn] += x[xn, cn, hh:hh + HH, ww:ww + WW] * dout[xn, fn, i, j]
    dx = dx[:, :, params['pad']:-params['pad'], params['pad']:-params['pad']]
    return dx, dw, db

if __name__ == "__main__":
    feature = np.random.rand(16, 32, 32)
    kernel = np.random.rand(8, 16, 3, 3)
    out = conv2d(feature, kernel)
    print(out.shape)