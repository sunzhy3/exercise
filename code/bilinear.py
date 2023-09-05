
import numpy as np
 
def bilinear(img, factor, align_corners=False):
    src_height, src_width, src_channel = img.shape
    out_height = src_height * factor
    out_width = src_width * factor
    out = np.zeros((out_height, out_width, src_channel))
    for i in range(out_height):
        for j in range(out_width):
            src_x = (i + 0.5) / factor - 0.5
            src_y = (j + 0.5) / factor - 0.5
            x = np.floor(src_x).astype(np.int32)
            y = np.floor(src_y).astype(np.int32)
            u = src_x - x
            v = src_y - y
            out[i, j, :] += (1 - u) * (1 - v) * img[max(x, 0), max(y, 0), :]
            out[i, j, :] += u * (1 - v) * img[min(x + 1, src_height - 1), max(y, 0), :]
            out[i, j, :] += (1 - u) * v * img[max(x, 0), min(y + 1, src_width - 1), :]
            out[i, j, :] += u * v * img[min(x + 1, src_height - 1), min(y + 1, src_width - 1), :]
    return out

def bilinear_2(img, factor, align_corners=True):
    src_height, src_width, src_channel = img.shape
    out_height = src_height * factor
    out_width = src_width * factor
    out = np.zeros((out_height, out_width, src_channel))

    stride = (src_height - 1) / (out_height - 1)
    x_ori_list = [0]
    for i in range(1, out_height - 1):
        x_ori_list.append(0 + i * stride)
    # append the last coordinate
    x_ori_list.append(src_height - 1)

    stride = (src_width - 1) / (out_width - 1)
    y_ori_list = [0]
    for i in range(1, out_width - 1):
        y_ori_list.append(0 + i * stride)
    # append the last coordinate
    y_ori_list.append(src_width - 1)

    for i in range(out_height):
        for j in range(out_width):
            src_x = x_ori_list[i]
            src_y = y_ori_list[j]
            x = np.floor(src_x).astype(np.int32)
            y = np.floor(src_y).astype(np.int32)
            u = src_x - x
            v = src_y - y
            out[i, j, :] += (1 - u) * (1 - v) * img[max(x, 0), max(y, 0), :]
            out[i, j, :] += u * (1 - v) * img[min(x + 1, src_height - 1), max(y, 0), :]
            out[i, j, :] += (1 - u) * v * img[max(x, 0), min(y + 1, src_width - 1), :]
            out[i, j, :] += u * v * img[min(x + 1, src_height - 1), min(y + 1, src_width - 1), :]
    return out
 
if __name__ == "__main__":
    img = np.array([[
        [ 1.,  2.,  0.],
        [ 3.,  4.,  0.],
        [ 0.,  0.,  0.]]])
    img = img.reshape((3, 3, 1))
    factor = 2
    out = bilinear_2(img, factor)

    print(out.reshape(6, 6))
