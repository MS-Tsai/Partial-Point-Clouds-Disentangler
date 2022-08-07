import os
import numpy as np
import open3d as o3d

def read_pcd(path):
    # Load data based on different extension
    extension = os.path.splitext(path)[-1]
    if extension == ".pcd":
        pcd = o3d.io.read_point_cloud(path)
        pcd = np.array(pcd.points)
    elif extension == ".txt":
        pcd = np.loadtxt(path)
        # pcd = []
        # with open(path, "r") as pcd_file:
        #     for line in pcd_file:
        #         content = line.strip().split(",")
        #         pcd.append(list(map(float, content)))
        # pcd = np.array(pcd)
        # pcd = pcd[:, :3]
    elif extension == ".npy":
        pcd = np.load(path)
    elif extension == ".npz":
        pcd = np.load(path)
        pcd = pcd["points"]
    else:
        assert False, extension + " is not supported now !"

    return pcd.astype(np.float32)

def write_pcd(point, output_path):
    # Convert numpy array to pcd format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point)

    # Output pcd file
    o3d.io.write_point_cloud(output_path, pcd)

def pcd_normalize(pcd):
    centroids = np.mean(pcd, axis=0)
    pcd = pcd - centroids
    max_dist = np.max(np.sqrt(np.sum(pcd**2, axis=1)))
    pcd = pcd / max_dist

    return pcd

def resample_pcd(points, num_points=1024):
    # Drop or duplicate points so that pcd has exactly n points
    idx = np.random.permutation(points.shape[0])
    if idx.shape[0] < num_points:
        idx = np.concatenate([idx, np.random.randint(points.shape[0], size = num_points - points.shape[0])])
    return points[idx[:num_points]]

def random_sample(points, num_points=1024):
    points = np.random.permutation(points)
    points = points[:num_points, :]

    return points

def farthest_point_sample(points, num_points=1024):
    """
    Input:
        points: a point set, in the format of NxM, where N is the number of points, and M is the point dimension
        num_points: required number of sampled points
    """
    def compute_dist(centroid, points):
        return np.sum((centroid - points) ** 2, axis=1)

    farthest_pts = np.zeros((num_points, points.shape[1]))
    farthest_pts[0] = points[np.random.randint(len(points))] # Random choose one point as starting point
    distances = compute_dist(farthest_pts[0], points)
    
    for idx in range(1, num_points):
        farthest_pts[idx] = points[np.argmax(distances)]
        distances = np.minimum(distances, compute_dist(farthest_pts[idx], points))
    
    return farthest_pts.astype(np.float32)
