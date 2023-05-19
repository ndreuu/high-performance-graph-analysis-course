import pytest

from project.pathes import *
from tests.utils import get_data


def new_matrix(edge_list: List[Tuple[int, float, int]]) -> Matrix:
    u, w, v = zip(*edge_list)
    n = max(u + v) + 1
    return Matrix.from_lists(u, v, w, nrows=n, ncols=n)


def normilize(
    xs: list[
        Union[list[Union[int, list[float]]], list[Union[int, list[Union[str, float]]]]]
    ]
) -> list[Union[int, list[float]]]:
    acc: list[Union[int, list[float]]] = []
    for x in xs:
        _acc: list[float] = []
        for v in x[1]:
            if v == "inf":
                _acc.append(math.inf)
            else:
                _acc.append(v)
        acc.append([x[0], _acc])
    return acc


def normilize1(xs: list[Union[str, float]]) -> list[float]:
    acc: list[float] = []
    for x in xs:
        if x == "inf":
            acc.append(math.inf)
        else:
            acc.append(x)

    return acc


@pytest.mark.parametrize(
    "edges, expected",
    get_data(
        "test_apsp",
        lambda d: (d["edges"], normilize(d["expected"])),
    ),
)
def test_apsp(edges, expected):
    graph = new_matrix(edges)
    assert apsp(graph) == expected


@pytest.mark.parametrize(
    "edges, starts, expected",
    get_data(
        "test_mssp",
        lambda d: (d["edges"], d["starts"], normilize(d["expected"])),
    ),
)
def test_mssp(edges, starts, expected):
    graph = new_matrix(edges)
    assert mssp(graph, starts) == expected


@pytest.mark.parametrize(
    "edges, start, expected",
    get_data(
        "test_sssp",
        lambda d: (d["edges"], d["start"], normilize1(d["expected"])),
    ),
)
def test_sssp(edges, start, expected):
    graph = new_matrix(edges)
    assert sssp(graph, start) == expected
