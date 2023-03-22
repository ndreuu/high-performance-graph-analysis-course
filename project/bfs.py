from typing import List

from pygraphblas import Vector, Matrix, INT64, BOOL
from pygraphblas.descriptor import RSC


def bfs(graph: Matrix, source: int) -> List[int]:
    """
    LA BFS algorithm for given graph and start vertex
    Parameters
    ----------
    graph: Matrix
        adjacency boolean matrix of a graph
    source: int
        start vertex
    Returns
    -------
    result: List[int]
        list with distance(or -1 if no path) from source to another vertexes
    """
    size = graph.ncols

    level = Vector.sparse(BOOL, size)
    acc = Vector.sparse(INT64, size)

    dist = 0
    level[source] = True

    while level.nvals:
        acc.assign_scalar(value=dist, mask=level)
        level.vxm(graph, out=level, mask=acc, desc=RSC)
        dist += 1

    return list(acc.get(i, default=-1) for i in range(size))
