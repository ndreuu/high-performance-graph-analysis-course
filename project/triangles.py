from typing import List

from pygraphblas import Matrix, INT64
from pygraphblas.descriptor import T0


def vertex_triangle_count(graph: Matrix) -> List[int]:
    """
    Counts the number of triangles for each vertex of undirected graph

    Parameters
    ----------
    graph: Matrix
        adjacency boolean matrix of undirected graph
    Returns
    -------
    result: List[int]
        The list in i-th position indicates the number of triangles for vertex i
    """
    triangles = graph.mxm(graph, cast=INT64, mask=graph).reduce_vector(desc=T0)
    return list(
        (
            triangles.dense(
                INT64, size=triangles.size, fill=triangles if triangles.nvals else 0
            )
            / 2
        ).vals
    )


def cohen(graph: Matrix) -> int:
    """
    Cohen's algorithm which calculates number of triangles in undirected graph

    Parameters
    ----------
    graph: Matrix
        adjacency boolean matrix of undirected graph
    Returns
    -------
    result: int
        number of triangles
    """
    return sum(graph.tril().mxm(graph.triu(), cast=INT64, mask=graph).vals) // 2


def sandia(graph: Matrix) -> int:
    """
    Sandia algorithm which calculates number of triangles in undirected graph
    Parameters
    ----------
    graph: Matrix
        adjacency boolean matrix of undirected graph
    Returns
    -------
    result: int
        number of triangles
    """
    u = graph.triu()
    return sum(u.mxm(u, cast=INT64, mask=u).vals)
