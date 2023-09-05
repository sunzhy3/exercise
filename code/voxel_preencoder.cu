// dim0
atomicAdd(feature + feature_reset_idx, fabs(x) / (pc_range[3] - pc_range[0]));
// dim1
atomicAdd(feature + feature_reset_idx + 1, fabs(y) / (pc_range[4] - pc_range[1]));
// dim2
atomicAdd(feature + feature_reset_idx + 2, fabs(z) / (pc_range[5] - pc_range[2]));
// dim3
atomicAdd(feature + feature_reset_idx + 3, intensity);
// dim4
atomicAdd(feature + feature_reset_idx + 4, fabs(x - voxel_center_x) / voxel_size[0]);
// dim5
atomicAdd(feature + feature_reset_idx + 5, fabs(y - voxel_center_y) / voxel_size[1]);
// dim6
atomicAdd(feature + feature_reset_idx + 6, fabs(z - voxel_center_z) / voxel_size[2]);
// dim7
atomicMax(feature + feature_reset_idx + 7, intensity);
// dim8
float angle = std::atan2(voxel_center_y, voxel_center_x);
atomicExch(feature + feature_reset_idx + 8, (angle + M_PI) / (2 * M_PI));
// dim9
atomicAdd(feature + feature_reset_idx + 9, 1.0f);