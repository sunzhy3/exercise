@VOXEL_ENCODERS.register_module()
class DynamicPillarVFE(BaseModule):
    def __init__(self,
                 in_channels,
                 pc_range,
                 voxel_size,
                 bev_size,
                 num_filters=(64, 64),
                 with_distance=False,
                 use_absolute_xyz=True,
                 init_cfg=None):
        super(DynamicPillarVFE, self).__init__(init_cfg)

        self.with_distance = with_distance
        self.use_absolute_xyz = use_absolute_xyz
        in_channels += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            in_channels += 1

        assert len(num_filters) > 0
        num_filters = [in_channels] + list(num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + pc_range[0]
        self.y_offset = self.voxel_y / 2 + pc_range[1]
        self.z_offset = self.voxel_z / 2 + pc_range[2]

        self.scale_xy = bev_size[0] * bev_size[1]
        self.scale_y = bev_size[1]

        self.bev_size = torch.tensor(bev_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.pc_range = torch.tensor(pc_range).cuda()

    # @force_fp32(out_fp16=True)
    def forward(self, points, bs_indicator, batch_size):
        '''
        :param points: list[tensor], tensor NxC, C=[x, y, z, ...]
        :return:
        '''
        points = torch.cat([bs_indicator.view(-1, 1), points], dim=1)

        points_coords = torch.floor(
            (points[:, [1, 2]] - self.pc_range[[0, 1]]) / self.voxel_size[[0, 1]]).int()
        # mask = ((points_coords >= 0) & (points_coords < self.bev_size[[0, 1]])).all(dim=1)
        # points = points[mask]
        # points_coords = points_coords[mask]
        points_coords[:, 0] = torch.clamp(points_coords[:, 0], min=0, max=self.bev_size[0] - 1)
        points_coords[:, 1] = torch.clamp(points_coords[:, 1], min=0, max=self.bev_size[1] - 1)
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        unq_coords = unq_coords.int()
        unq_coords = torch.stack((unq_coords // self.scale_xy,
                                  (unq_coords % self.scale_xy) // self.scale_y,
                                  unq_coords % self.scale_y,
                                  torch.zeros(unq_coords.shape[0]).to(unq_coords.device).int()
                                  ), dim=1)
        unq_coords = unq_coords[:, [0, 3, 2, 1]]

        # original pts coords
        coords = merge_coords.int()
        coords = torch.stack((coords // self.scale_xy,
                              (coords % self.scale_xy) // self.scale_y,
                              coords % self.scale_y,
                              torch.zeros(coords.shape[0]).to(coords.device).int()
                              ), dim=1)
        coords = coords[:, [0, 3, 2, 1]]  # (bs_idx, z, y, x)
        bev_coords = coords[:, [0, 2, 3]].contiguous()  # (bs_idx, y, x)
        return features, unq_coords, bev_coords