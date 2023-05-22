import heapq
import itertools
import math
from typing import Dict, Hashable

import networkx as nx
from boltons.queueutils import HeapPriorityQueue


def dijkstra(graph: nx.Graph, start: int) -> dict[Hashable, float]:
    """
    Dijkstra algorithm

    Parameters
    ----------
    graph: nx.Graph
        graph
    start: int
        start vertex id
    Returns
    -------
    result: dict[Hashable, float]
        map from ids to dist their distances
    """
    acc = {node: math.inf for node in graph.nodes}
    acc[start] = 0
    q = [(start, 0)]

    while q:
        x, d = heapq.heappop(q)
        if d > acc[x]:
            continue
        for neighbor in graph.neighbors(x):
            d_upd = acc[x] + graph[x][neighbor].get("weight", 1)
            if d_upd < acc[neighbor]:
                acc[neighbor] = d_upd
                heapq.heappush(q, (neighbor, d_upd))

    return acc


class DynamicSSSP:
    """
    DynamicSSSP algorithm
    See more here:
    An Incremental Algorithm for a Generalization of the Shortest-Path Problem
    G. Ramalingam† and Thomas Reps‡
    """

    def __init__(self, graph: nx.DiGraph, start_vertex: int):
        self._graph: nx.DiGraph = graph
        self._start_vertex = start_vertex
        self._dists = dijkstra(graph, start_vertex)
        self._modified_vertices = set()

    def get_distances(self) -> Dict[Hashable, float]:
        self._update()
        return self._dists

    def _update(self):
        rhs = {}
        heap = HeapPriorityQueue(priority_key=lambda x: x)
        for u in self._modified_vertices:
            rhs[u] = self._compute_rhs(u)
            if rhs[u] != self._dists[u]:
                key = min(rhs[u], self._dists[u])
                heap.add(u, key)

        while heap:
            u = heap.pop()

            if rhs[u] < self._dists[u]:
                self._dists[u] = rhs[u]
                to_update_rhs = self._graph.successors(u)
            else:
                self._dists[u] = math.inf
                to_update_rhs = itertools.chain(self._graph.successors(u), [u])

            for v in to_update_rhs:
                rhs[v] = self._compute_rhs(v)
                if rhs[v] != self._dists[v]:
                    heap.add(v, priority=min(rhs[v], self._dists[v]))
                else:
                    if v in heap._entry_map:
                        heap.remove(v)

    def remove_edge(self, u, v):
        self._graph.remove_edge(u, v)
        self._modified_vertices.add(v)

    def add_edge(self, u, v):
        self._graph.add_edge(u, v, weight=1)
        self._modified_vertices.add(v)

    def _compute_rhs(self, v):
        if v == self._start_vertex:
            return 0
        else:
            return min(
                (
                    self._dists[u] + self._graph[u][v].get("weight", 1)
                    for u in self._graph.predecessors(v)
                ),
                default=float("inf"),
            )
