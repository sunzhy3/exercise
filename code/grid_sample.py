import numpy as np
 
def grid_smaple_bilinear(img, grid, align_corners=False):
    src_height, src_width, src_channel = img.shape
    out_height, out_width, _ = grid.shape
    factor = out_height / src_height
    out = np.zeros((out_height, out_width, src_channel))
    for i in range(out_height):
        for j in range(out_width):
            gi = (grid[i][j][0] + 1) / 2 * (out_height - 1)
            gj = (grid[i][j][1] + 1) / 2 * (out_width - 1)
            src_x = (gi + 0.5) / factor - 0.5
            src_y = (gj + 0.5) / factor - 0.5
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
    grid = np.random.random(size=(6, 6, 2)) * 2 - 1.0
    out = grid_smaple_bilinear(img, grid)

    print(out.reshape(6, 6))
