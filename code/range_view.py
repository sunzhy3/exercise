import numpy as np
  
def do_range_projection(points, fov_up, fov_down, H, W):
    """
        points: (N, 3)
        fov_up: field of view up in degree
        fov_down: field of view down in degree
        H: height of range image
        W: width of range image
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(points, 2, axis=1)

    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= W                              # in [0.0, W]
    proj_y *= H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]

    # projected range image - [H,W] range (-1 is no data)
    proj_range = np.full((H, W), -1, dtype=np.float32)

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # assing to images
    proj_range[proj_y, proj_x] = depth

    return proj_range

if __name__ == "__main__":
    points = np.random.rand(100, 3) * 100.0
    fov_up = 30
    fov_down = -30
    H = 1024
    W = 64
    proj_range = do_range_projection(points, fov_up, fov_down, H, W)

    print(proj_range, np.sum(proj_range > 0))
