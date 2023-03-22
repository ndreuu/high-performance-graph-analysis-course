from typing import List, Union, Tuple

from pygraphblas import Vector, Matrix, INT64, BOOL
from pygraphblas.descriptor import RSC, S, C, RC, R


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


def bfs_multi_source(graph: Matrix, sources: List[int]) -> List[Union[int, List[int]]]:
    """
    LA BFS algorithm for given graph and start vertexes
    Parameters
    ----------
    graph: Matrix
        adjacency boolean matrix of a graph
    sources: List[int]
        start vertexes
    Returns
    -------
    List[Tuple[int, List[int]]]
        list of pairs: vertex and list of distance to vertex or -1 when vertex is not reachable
    """
    size = graph.ncols
    src_size = len(sources)
    q = Matrix.sparse(BOOL, nrows=src_size, ncols=size)
    used = Matrix.sparse(BOOL, nrows=src_size, ncols=size)
    acc = Matrix.dense(INT64, nrows=src_size, ncols=size, fill=-1)

    for i, j in enumerate(sources):
        q.assign_scalar(True, i, j)
        used.assign_scalar(True, i, j)
        acc.assign_scalar(0, i, j)

    step = 1
    prev_nnz = -1
    while used.nvals != prev_nnz:
        prev_nnz = used.nvals
        q.mxm(graph, mask=used, out=q, desc=RC)
        used.eadd(q, BOOL.lor_land, out=used, desc=R)
        acc.assign_scalar(step, mask=q)
        step += 1

    return [[vertex, list(acc[i].vals)] for i, vertex in enumerate(sources)]
