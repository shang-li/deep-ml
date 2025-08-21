import numpy as np

def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
    n = len(points)
    membership = np.zeros(n)
    centroids = np.array(initial_centroids, dtype=float)
    #d = centroids.shape[1]
    points = np.array(points)
    
    for i in range(max_iterations):
        # update membership
        dist_mat = np.sum((points[:, None, :] - centroids[None, :, :])**2, axis=-1) # n * k * d -> n * k
        membership = dist_mat.argmin(axis=1)

        # update centroids
        for j in range(k):
            centroids[j, :] = points[membership == j, :].mean(axis=0)

    final_centroids = centroids
    return final_centroids

def k_means_clustering_2d(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
    membership = [None] * len(points)
    centroids = initial_centroids
    for i in range(max_iterations):
        # update membership
        for q, point in enumerate(points):
            closest_dist = float("inf")
            for j in range(k):
                dist = (point[0] - centroids[j][0])**2 + (point[1] - centroids[j][1])**2
                if dist < closest_dist:
                    closest_dist = dist
                    membership[q] = j
            
        # update centroids
        new_centroids = [(0, 0, 0)] * k
        for q, point in enumerate(points):
            xsum, ysum, cnt = new_centroids[membership[q]]
            xsum += point[0]
            ysum += point[1]
            cnt += 1
            new_centroids[membership[q]] = (xsum, ysum, cnt)
        centroids = [(xsum/cnt, ysum/cnt) for xsum, ysum, cnt in new_centroids]
    final_centroids = centroids
    return final_centroids

arr = [(0, 0), (1, 0), (0, 1), (1, 1), (5, 5), (6, 5), (5, 6), (6, 6),(0, 5), (1, 5), (0, 6), (1, 6), (5, 0), (6, 0), (5, 1), (6, 1)]
init_centroids = [(0, 0), (0, 5), (5, 0), (5, 5)]
print(k_means_clustering(arr, 4, init_centroids, 10))
print(k_means_clustering_2d(arr, 4, init_centroids, 10))