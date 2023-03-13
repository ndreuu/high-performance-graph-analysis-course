import pytest
from project.bfs import *
from pygraphblas import Matrix


@pytest.mark.parametrize(
    "I, J, start, expeted",
    [
        (
            [0, 0, 2, 3, 4],
            [1, 2, 1, 4, 3],
            3,
            [-1, -1, -1, 0, 1],
        ),
        (
            [0, 0, 2, 3, 4],
            [1, 2, 1, 4, 3],
            0,
            [0, 1, 1, -1, -1],
        ),
        (
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            0,
            [0, 1, 2, 3, 4],
        )
    ],
)
def test_bfs(I, J, start, expeted):
    nrows = ncols = max(I + J) + 1
    graph = Matrix.from_lists(I, J, list(map(lambda x: True, I)), nrows=nrows, ncols=ncols)
    assert bfs(graph, start) == expeted
