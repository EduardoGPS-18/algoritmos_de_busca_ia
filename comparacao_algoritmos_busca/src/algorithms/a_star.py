"""
Algoritmo A*: implementação manual.
Busca heurística com f(n) = g(n) + h(n); reduz expansão de nós em relação ao Dijkstra.
Usa apenas estruturas do grafo (NetworkX): nós, sucessores e custo via graph.get_edge_cost;
heurística: graph.get_straight_line_distance. Fila de prioridade: heapq (min-heap por f).
"""

from __future__ import annotations
import heapq
from typing import Callable, List, Optional, Tuple

import networkx as nx

from ..graph import get_edge_cost, get_straight_line_distance, path_cost, validate_path_nodes


def a_star(
    G: nx.DiGraph,
    start: str,
    goal: str,
    heuristic: Optional[Callable[[str, str], float]] = None,
) -> Tuple[Optional[List[str]], float]:
    """
    Retorna (caminho do start ao goal, custo total) ou ([], inf) se não houver caminho.
    Heurística padrão: distância em linha reta entre o nó e o objetivo (admissível).
    """
    validate_path_nodes(G, start, goal)
    if heuristic is None:
        def default_heuristic(n: str, target: str) -> float:
            return get_straight_line_distance(G, n, target)
        heuristic = default_heuristic

    g_score: dict[str, float] = {start: 0.0}
    h_start = heuristic(start, goal)
    f_score: dict[str, float] = {start: g_score[start] + h_start}
    prev: dict[str, Optional[str]] = {start: None}
    open_set: List[Tuple[float, str]] = [(f_score[start], start)]
    closed: set[str] = set()

    while open_set:
        _, u = heapq.heappop(open_set)
        if u in closed:
            continue
        if u == goal:
            path: List[str] = []
            cur: Optional[str] = goal
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path, path_cost(G, path)

        closed.add(u)

        for v in G.successors(u):
            if v in closed:
                continue
            w = get_edge_cost(G, u, v)
            if w == float("inf"):
                continue
            tentative_g = g_score[u] + w
            if tentative_g < g_score.get(v, float("inf")):
                prev[v] = u
                g_score[v] = tentative_g
                f_score[v] = tentative_g + heuristic(v, goal)
                heapq.heappush(open_set, (f_score[v], v))

    return [], float("inf")
