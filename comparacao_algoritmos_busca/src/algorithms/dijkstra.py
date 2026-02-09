"""
Algoritmo de Dijkstra: implementação manual.
Caminho de custo mínimo em grafo com pesos não negativos (busca exaustiva).
Usa apenas estruturas do grafo (NetworkX): nós, arestas, sucessores e custo via graph.get_edge_cost.
Fila de prioridade: heapq (min-heap).
"""

from __future__ import annotations
import heapq
from typing import List, Optional, Tuple

import networkx as nx

from ..graph import get_edge_cost, path_cost, validate_path_nodes


def dijkstra(
    G: nx.DiGraph,
    start: str,
    goal: str,
) -> Tuple[Optional[List[str]], float]:
    """
    Retorna (caminho do start ao goal como lista de ids, custo total)
    ou ([], inf) se não houver caminho.

    Implementação manual: dist[] e prev[]; fila (dist, nó) com heapq;
    a cada passo expande o nó com menor distância atual e relaxa as arestas.
    """
    validate_path_nodes(G, start, goal)
    dist: dict[str, float] = {start: 0.0}
    prev: dict[str, Optional[str]] = {start: None}
    heap: List[Tuple[float, str]] = [(0.0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if u == goal:
            path: List[str] = []
            cur: Optional[str] = goal
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path, path_cost(G, path)

        if d > dist.get(u, float("inf")):
            continue

        for v in G.successors(u):
            w = get_edge_cost(G, u, v)
            if w == float("inf"):
                continue
            alt = dist[u] + w
            if alt < dist.get(v, float("inf")):
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))

    return [], float("inf")
