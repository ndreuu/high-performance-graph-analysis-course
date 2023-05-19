import math
from typing import List, Tuple, Union

from pygraphblas import Matrix, FP64


def sssp(graph: Matrix, source: int) -> List[int]:
    """
    Single source shortest path

    Parameters
    ----------
    graph: Matrix
        adjacency matrix
    source: int
        start vertex
    Returns
    -------
    result: List[int]
        list with distances to i-vertex
    """
    return mssp(graph, [source])[0][1]


def mssp(graph: Matrix, source: List[int]) -> List[Union[int, List[int]]]:
    """
    Multiple-source shortest paths

    Parameters
    ----------
    graph: Matrix
        adjacency matrix
    source: List[int]
        list of start vertices
    Returns
    -------
    result: List[Tuple[int, List[int]]]
        list of tuples (source, distances_from_source)
    """
    graph, m, n = graph.dup(), len(source), graph.ncols
    for i in range(n):
        graph[i, i] = 0.0

    dists = Matrix.sparse(FP64, nrows=m, ncols=n)
    for i, start in enumerate(source):
        dists[i, start] = 0

    for _ in range(n - 1):
        dists.mxm(graph, FP64.MIN_PLUS, out=dists)

    if dists.isne(dists.mxm(graph, FP64.MIN_PLUS)):
        raise ValueError("Negative cycle")

    return [
        [start, [dists.get(i, j, default=math.inf) for j in range(n)]]
        for i, start in enumerate(source)
    ]


def apsp(graph: Matrix) -> List[Union[int, List[int]]]:
    """
    All-pairs shortest path

    Parameters
    ----------
    graph: Matrix
        adjacency matrix
    Returns
    -------
    result: List[Tuple[int, List[int]]]
        list of tuples (source, distances_from_source)
    """
    graph, n = graph.dup(), graph.ncols
    for i in range(n):
        graph[i, i] = 0.0

    assert graph.type == FP64
    assert graph.square

    dists = graph
    for k in range(n):
        col, row = dists.extract_matrix(col_index=k), dists.extract_matrix(row_index=k)
        dists.eadd(col.mxm(row, FP64.MIN_PLUS), FP64.MIN, out=dists)

    for k in range(n):
        col, row = dists.extract_matrix(col_index=k), dists.extract_matrix(row_index=k)
        if dists.isne(dists.eadd(col.mxm(row, FP64.MIN_PLUS), FP64.MIN)):
            raise ValueError("Negative cycle")

    return [
        [i, [dists.get(i, j, default=math.inf) for j in range(n)]] for i in range(n)
    ]
