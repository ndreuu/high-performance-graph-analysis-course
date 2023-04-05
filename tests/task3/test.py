import pytest
from project.bfs import *
from pygraphblas import Matrix

from project.triangles import vertex_triangle_count, cohen, sandia
from tests.utils import get_data


@pytest.mark.parametrize(
    "I, J, source, ans",
    get_data(
        "test_triangles_algorithms",
        lambda d: (d["from"], d["to"], d["source"], d["ans"]),
    ),
)
def test_triangles_algorithms(I, J, source, ans):
    nrows = ncols = max(I + J) + 1
    graph = Matrix.from_lists(
        I, J, list(map(lambda x: True, I)), nrows=nrows, ncols=ncols
    )

    assert (cohen(graph), sandia(graph)) == (ans, ans)


@pytest.mark.parametrize(
    "I, J, source, ans",
    get_data(
        "test_triangles_vertex_count",
        lambda d: (d["from"], d["to"], d["source"], d["ans"]),
    ),
)
def test_triangles_vertex_count(I, J, source, ans):
    nrows = ncols = max(I + J) + 1
    graph = Matrix.from_lists(
        I, J, list(map(lambda x: True, I)), nrows=nrows, ncols=ncols
    )

    assert vertex_triangle_count(graph) == ans
