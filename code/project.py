import numpy as np

def project(points, lidar2vehicle, vehicle2cam, intrinsic, img):
    """
    points: (N, 3)
    lidar2vehicle: (4, 4)
    vehicle2cam: (4, 4)
    intrinsic: (4, 4)
    img: (H, W, 3)
    """
    points_homo = np.ones((points.shape[0], 4))
    points_homo[:, :3] = points
    points_cam = vehicle2cam * lidar2vehicle * points_homo.T
    points_img = intrinsic * points_cam
    points_img /= points_img[:, 2]

def get_pinhole_2d_points(
    self,
    points3d,
    vehicle2cam: torch.Tensor,
    cam2img: torch.Tensor,
):  # pylint: disable=no-self-use
    """Convert vehicle Coordinate 3D Point to Image Space, for pinhole camera.

    Args:
        points3d (tensor): vehicle Space 3D Point with shape [N, 3].
        vehicle2cam (tensor): [B, N_view, 4, 4], vehicle to cam
        cam2imgs (tensor): [B, N_view, 21]
            [fx, 0, cx, 0,  0, fy, cy,0,  0, 0, 1, 0,  0, 0, 0, 1] + [0,0,0,0,0]
            for align with svc(21)

    Returns:
        pts_pixel (tensor): [B, N_view, N, 2]
        pts_depth (tensor): [B, N_view, N]
    """
    # breakpoint()
    if self.batch_size_pinhole is None or self.num_cam_pinhole is None:
        self.batch_size_pinhole, self.num_cam_pinhole = vehicle2cam.shape[:2]
    cam2img = cam2img[..., :16].view(
        self.batch_size_pinhole, self.num_cam_pinhole, 4, 4
    )
    # B, N, 4, 4
    vehicle2img = torch.matmul(cam2img, vehicle2cam)

    vehicle2img = vehicle2img.view(-1, 4, 4).permute(0, 2, 1)  # BN, 4, 4
    point2d = torch.matmul(points3d, vehicle2img)  # BN, XYZ, 4

    # Same with `point2d[..., :2] /= point2d[..., 2:3]`
    point2d_new = point2d[..., :2] / point2d[..., 2:3]
    point2d_new = torch.cat((point2d_new, point2d[..., 2:4]), dim=-1)

    point2d_new = point2d_new.reshape(
        self.batch_size_pinhole, self.num_cam_pinhole, -1, 4
    )
    return point2d_new[..., :3]  # B, N, XYZ, 3

def get_rot(self, rad):  # pylint: disable=no-self-use
    """Generate 2D rotation matrix according to the input radian.

    Args:
        rad (float): Ratation magnitude in radian.

    Returns:
        torch.Tensor: The 2D rotation matrix in shape of (2, 2).
    """
    return torch.Tensor(
        [
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)],
        ]
    )

def get_post_transform(self, resize, crop, flip, rotate):
    """Generate 3D translation and rotation matrix.

    Args:
        resize (float): Scale of resize.
        crop (tuple(int)): Range of cropping in format of (lower_w,
            lower_h, upper_w, upper_h).
        flip (bool): Flag of flip operation.
        rotate (float): Magnitude of rotation in angle.

    Returns:
        tuple(torch.Tensor): The 3D translation and rotation matrix.
    """
    post_rot = torch.eye(2)
    post_tran = torch.zeros(2)
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:  # no flip in current version
        # pylint: disable=invalid-name
        A = torch.Tensor([[-1, 0], [0, 1]])  # noqa: N806
        # pylint: disable=invalid-name
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    # pylint: disable=invalid-name
    A = self.get_rot(rotate / 180 * np.pi)  # noqa: N806
    # pylint: disable=invalid-name
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b  # pylint: disable = C0103
    post_rot = A.matmul(post_rot)  # pylint: disable = C0103
    post_tran = A.matmul(post_tran) + b  # pylint: disable = C0103
    post_tran_3d = torch.zeros(3).float()
    post_rot_3d = torch.eye(3).float()
    post_tran_3d[:2] = post_tran
    post_rot_3d[:2, :2] = post_rot
    return post_rot_3d, post_tran_3d
