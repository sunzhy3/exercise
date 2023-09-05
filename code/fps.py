import numpy as np

def farthest_point_sample(xyz, npoint): 

    """
    Input:
        xyz: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
    """
    N, _ = xyz.shape
    
    centroids = np.zeros(npoint)                      # 采样点矩阵（B, npoint）
    distance = np.ones(N)* 1e10                       # 采样点到所有点距离（B, N）
    
    bary_center = np.sum((xyz), axis=0) / N           # 计算重心坐标 及 距离重心最远的点
    dist = np.sum((xyz - bary_center) ** 2, -1)
    farthest = np.argmax(dist)                        # 将距离重心最远的点作为第一个点

    for i in range(npoint):
        centroids[i] = farthest                       # 更新第i个最远点
        centroid = xyz[farthest, :]                   # 取出这个最远点的xyz坐标
        dist = np.sum((xyz - centroid) ** 2, -1)      # 计算点集中的所有点到这个最远点的欧式距离
        mask = dist < distance
        distance[mask] = dist[mask]                   # 更新 distance，记录样本中每个点距离所有已出现的采样点的最小距离
        farthest = np.argmax(distance)                # 返回最远点索引
 
    return centroids

if __name__ == '__main__':
    xyz = np.random.rand(128, 3)
    npoint = 32

    centroids = farthest_point_sample(xyz, npoint)
    
    print("Sampled pts: ", centroids)