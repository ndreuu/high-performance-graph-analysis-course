import pytest
from project.bfs import *
from pygraphblas import Matrix

from tests.utils import get_data


@pytest.mark.parametrize(
    "I, J, start, expected",
    get_data(
        "test_bfs",
        lambda d: (d["from"], d["to"], d["start"], d["expected"]),
    ),
)
def test_read_cfg(I, J, start: int, expected: list[int]):
    nrows = ncols = max(I + J) + 1
    graph = Matrix.from_lists(
        I, J, list(map(lambda x: True, I)), nrows=nrows, ncols=ncols
    )
    assert bfs(graph, start) == expected
