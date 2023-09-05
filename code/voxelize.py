import numpy as np

def voxelize(xyz, feat, coord_range, voxel_size, K):
    """
        xyz: (N, 3)
        feat: (N, C)
        coord_range: (2, 3)
        voxel_size: (3)

        output_feat: (M, K, C)
        output_coords: (M, 3)
    """
    min_bound = coord_range[0]
    max_bound = coord_range[1]
    dim = (max_bound - min_bound + 1e-3) / voxel_size
    dim = dim.astype(np.int32)
    coords = (xyz - min_bound) / voxel_size
    coords = coords.astype(np.int32)

    mask = (coords[:, 0] > 0) & (coords[:, 0] < dim[0]) & \
           (coords[:, 1] > 0) & (coords[:, 1] < dim[1]) & \
           (coords[:, 2] > 0) & (coords[:, 2] < dim[2])
    coords = coords[mask]
    feat = feat[mask]
    
    temp_dict = {}
    coords_idx = coords[:, 0] * dim[1] * dim[2] + coords[:, 1] * dim[2] + coords[:, 2]
    for i, ci in enumerate(coords_idx):
        if ci not in temp_dict:
            temp_dict[ci] = []
        temp_dict[ci].append([coords[i, :], feat[i, :]])

    M = len(temp_dict.keys())
    output_feat = np.zeros((M, K, feat.shape[1]))
    output_coords = np.zeros((M, K))
    for idx, ci in enumerate(temp_dict):
        for i in range(len(temp_dict[ci])):
            if i > K:
                continue
            output_feat[idx, i, :] = temp_dict[ci][i][1]
            output_coords[idx, :] = temp_dict[ci][i][0]

    return output_feat, output_coords

if __name__ == "__main__":
    xyz = np.random.rand(100, 3) * 100.0
    feat = np.random.rand(100, 9)
    coord_range = np.zeros((2, 3))
    coord_range[0, :] = np.min(xyz, axis=0) + 10.0
    coord_range[1, :] = np.max(xyz, axis=0) - 10.0
    voxel_size = np.array([0.1, 0.1, 0.1])
    K = 3
    out_feat, out_coords = voxelize(xyz, feat, coord_range, voxel_size, K)

    print(out_feat.shape, out_coords.shape)

