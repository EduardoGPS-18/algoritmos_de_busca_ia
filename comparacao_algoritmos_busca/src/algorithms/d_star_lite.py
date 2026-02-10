"""
D* Lite: implementação manual.
Replanejamento incremental para ambientes dinâmicos; busca reversa (objetivo → origem).
Usa apenas estruturas do grafo (NetworkX): nós, predecessores (G.predecessors) e custo via graph.get_edge_cost.
Fila de prioridade: heapq (min-heap por f). Versão simplificada para comparação com Dijkstra e A*.
"""

from __future__ import annotations
import heapq
from typing import Dict, List, Optional, Tuple

import networkx as nx

from ..graph_operations import GraphOperations


def d_star_lite(
    G: nx.DiGraph,
    start: str,
    goal: str,
) -> Tuple[Optional[List[str]], float]:
    """
    Retorna (caminho do start ao goal, custo total).
    Busca reversa (do goal ao start) sobre G; usa predecessores e custo (v, u).
    """
    GraphOperations.validate_path_nodes(G, start, goal)
    def h(u: str) -> float:
        return GraphOperations.get_straight_line_distance(G, u, start)

    g_score: Dict[str, float] = {goal: 0.0}
    f_score: Dict[str, float] = {goal: h(goal)}
    prev: Dict[str, Optional[str]] = {goal: None}
    open_set: List[Tuple[float, str]] = [(f_score[goal], goal)]

    while open_set:
        _, u = heapq.heappop(open_set)
        if u == start:
            path: List[str] = []
            cur: Optional[str] = start
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path, g_score[start]

        # No grafo reverso: predecessores de u (aresta v -> u no original)
        for v in G.predecessors(u):
            w = GraphOperations.get_edge_cost(G, v, u)
            if w == float("inf"):
                continue
            tentative_g = g_score[u] + w
            if tentative_g < g_score.get(v, float("inf")):
                prev[v] = u
                g_score[v] = tentative_g
                f_score[v] = tentative_g + h(v)
                heapq.heappush(open_set, (f_score[v], v))

    return [], float("inf")


def d_star_lite_replan(
    G: nx.DiGraph,
    start: str,
    goal: str,
    changed_edges: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[Optional[List[str]], float]:
    """
    Interface para replanejamento: alterações já refletidas em G.graph.
    Retorna novo caminho.
    """
    return d_star_lite(G, start, goal)
