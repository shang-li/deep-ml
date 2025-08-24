
from typing import List

def getAncestors(n: int, edges: List[List[int]]) -> List[List[int]]:
    # trace back to ancestors with DFS
    # preprocess graph to - node: ancestors
    graph = {i: set() for i in range(n)}
    for edge in edges:
        graph[edge[1]] = graph[edge[1]].union(set([edge[0]]))
    
    ancestor = {}
    def DFS(v):
        if v not in ancestor:
            ancestor_tmp = set()
            for vv in graph[v]:
                ancestor_tmp.add(vv)
                ancestor_tmp = ancestor_tmp.union(DFS(vv))
            ancestor[v] = ancestor_tmp
        return ancestor[v]
    for v in range(n):
        if v not in ancestor:
            DFS(v)
        import pdb; pdb.set_trace()
    ret = [[] for _ in range(n)]
    for i, l in ancestor.items():
        ret[i] = sorted(l)
    return ret


#print(getAncestors(8, [[0,3],[0,4],[1,3],[2,4],[2,7],[3,5],[3,6],[3,7],[4,6]]))


def canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool:
    # build prerequisite graph
    graph = {i: set() for i in range(numCourses)}
    for c, r in prerequisites:
        graph[r].add(c)
    #import pdb; pdb.set_trace()
    visited = set()
    def DFS(path, v):
        if v in path:
            return False
        if v not in visited:
            visited.add(v)
            for n in graph[v]:
                if not DFS(path.union(set([v])), n):
                    return False
        return True
    
    for i in range(numCourses):
        if i not in visited:
            if not DFS(set(), i):
                return False
    return True

print(canFinish(2, [[0, 1]]))

import numpy as np

def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True):
    train_test_ids = []
    n = len(X)
    #import pdb; pdb.set_trace()
    indices_list = list(range(n))
    if shuffle:
        np.random.shuffle(indices_list)
    group_size = n // k
    for i in range(k):
        train_test_ids.append(
            (
                indices_list[:i * group_size] + indices_list[(i + 1) * group_size:],
                indices_list[i * group_size: (i + 1) * group_size]
            )
        )
        #print(train_test_ids[i])
    return train_test_ids

print(k_fold_cross_validation(np.array([0,1,2,3,4,5,6,7,8,9]), np.array([0,1,2,3,4,5,6,7,8,9]), k=2, shuffle=True))


def batch_iterator(X, y=None, batch_size=64):
    datasets = []
    n = len(X)
    nbatches = n // batch_size + (n % batch_size > 0)
    for k in range(nbatches):
        datasets.append(
            [X[k * batch_size:min(n, (k+1) * batch_size)],
            y[k * batch_size:min(n, (k+1) * batch_size)]]
        )
    return datasets

print(batch_iterator(np.array([[1, 1], [2, 2], [3, 3], [4, 4]]), batch_size=3))