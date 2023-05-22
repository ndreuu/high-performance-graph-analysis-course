import networkx as nx
import pytest

from project.task9 import dijkstra, DynamicSSSP
from tests.utils import get_data

import random


@pytest.mark.parametrize(
    "graph, start, expected",
    get_data(
        "test_dijkstra",
        lambda d: (
            nx.node_link_graph(d["graph"], directed=True, multigraph=False),
            d["start"],
            {v: float(l) for v, l in d["expected"].items()},
        ),
    ),
)
def test_dijkstra(graph: nx.DiGraph, start: str, expected: dict[str, float]):
    actual = dijkstra(graph, start)

    assert actual == expected


def test_dynamic():
    for i in range(1, 100):
        g: nx.DiGraph = nx.DiGraph(nx.generators.atlas.graph_atlas(i))

        edges = list(g.edges)
        rm_edges_num = len(edges) // 2
        add_edges_num = len(edges) - rm_edges_num

        add_edges = []
        for _ in range(add_edges_num):
            i = random.randint(0, len(edges) - 1)
            u, v = edges[i]
            add_edges.append(edges[i])
            g.remove_edge(u, v)
            del edges[i]

        del_edges = list(g.edges)
        dynamic_sssp = DynamicSSSP(g, list(g.nodes)[0])

        while add_edges or del_edges:
            if add_edges and random.random() < 0.5:
                u, v = add_edges.pop()
                dynamic_sssp.add_edge(u, v)
            elif del_edges:
                u, v = del_edges.pop()
                dynamic_sssp.remove_edge(u, v)
            else:
                u, v = add_edges.pop()
                dynamic_sssp.add_edge(u, v)

            expected = dijkstra(g, list(g.nodes)[0])
            assert dynamic_sssp.get_distances() == expected


def test_dynamic_inc():
    for i in range(1, 100):
        g1: nx.DiGraph = nx.DiGraph(nx.generators.atlas.graph_atlas(i))

        modifiable_graph = nx.DiGraph()
        modifiable_graph.add_nodes_from(g1.nodes)

        dynamic_sssp = DynamicSSSP(modifiable_graph, 0)
        for u, v in g1.edges:
            dynamic_sssp.add_edge(u, v)

            expected = dijkstra(modifiable_graph, 0)
            assert dynamic_sssp.get_distances() == expected


def test_dynamic_dec():
    for i in range(1, 100):
        modifiable_graph: nx.DiGraph = nx.DiGraph(nx.generators.atlas.graph_atlas(i))

        start_vertex = list(modifiable_graph)[0]
        dynamic_sssp = DynamicSSSP(modifiable_graph, start_vertex)
        edges = list(modifiable_graph.edges)
        while edges:
            i = random.randint(0, len(edges) - 1)
            u, v = edges[i]
            del edges[i]

            dynamic_sssp.remove_edge(u, v)

            expected = dijkstra(modifiable_graph, start_vertex)
            assert dynamic_sssp.get_distances() == expected
