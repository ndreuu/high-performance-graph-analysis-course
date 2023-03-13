from typing import List

from pygraphblas import Vector, Matrix, INT64, BOOL


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
    visited = Vector.sparse(BOOL, size)
    acc = Vector.dense(INT64, size, fill=-1)

    dist = 0
    level[source] = True
    prev_visited = None

    while prev_visited != visited.nvals:
        prev_visited = visited.nvals
        acc.assign_scalar(dist, mask=level)
        visited |= level
        level @= graph
        level.assign_scalar(False, mask=visited)
        dist += 1

    return list(acc.vals)
