"""
Operações sobre o grafo: custo das arestas, função de peso, validação, distância e custo de caminho.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict

import networkx as nx

# Chaves em G.graph para parâmetros do cenário (chuva = toda a área)
KEY_RAIN_MULTIPLIER = "rain_multiplier"
KEY_SLOPE_PENALTY_FACTOR = "slope_penalty_factor"
KEY_CONGESTION_FACTOR = "congestion_factor"
KEY_CONGESTION_FACTOR_BY_REGION = "congestion_factor_by_region"
KEY_CONGESTION_FACTOR_BY_EDGE = "congestion_factor_by_edge"
KEY_EDGE_OVERRIDE = "edge_override"
KEY_EDGE_COST_MULTIPLIER = "edge_cost_multiplier"

# Prefixos de nó para região (grafo regional: op_, mariana_, cachoeira_)
REGION_PREFIXES = ("op_", "mariana_", "cachoeira_")


class GraphOperations:
    """Responsável pelos cálculos de custo das vias e operações sobre o grafo."""

    @staticmethod
    def get_region(node_id: str) -> str:
        """
        Retorna a região do nó a partir do id.
        op_* -> "op", mariana_* -> "mariana", cachoeira_* -> "cachoeira", senão -> "default".
        """
        if not isinstance(node_id, str):
            return "default"
        for prefix in REGION_PREFIXES:
            if node_id.startswith(prefix):
                return prefix.rstrip("_")
        return "default"

    @staticmethod
    def compute_edge_cost(
        d: Dict[str, Any],
        rain_multiplier: float = 1.0,
        slope_penalty_factor: float = 1.0,
    ) -> float:
        """
        Custo do arco: distância + penalização por declividade.
        d deve conter: distance, slope_pct.
        """
        dist = d.get("distance", 0.0)
        slope_pct = d.get("slope_pct", 0.0)

        # Penalidade por declividade: íngreme custa muito mais que longo (esforço sobe não-linearmente).
        # Termo linear (base) + termo quadrático (penaliza forte ladeiras; carro às vezes nem sobe).
        if slope_pct > 0:
            s = slope_pct / 100.0
            slope_penalty = s * slope_penalty_factor * rain_multiplier
            slope_penalty += (s * s) * slope_penalty_factor * 10.0 * rain_multiplier  # quadrático: íngreme >> longo
        else:
            slope_penalty = 0.0

        cost = dist * (1.0 + slope_penalty)
        return max(cost, 1e-6)

    @staticmethod
    def get_region_from_graph(G: nx.DiGraph, node_id: Any) -> str:
        """Região do nó: atributo G.nodes[n]['region'] ou inferida por prefixo."""
        return G.nodes.get(node_id, {}).get("region", GraphOperations.get_region(str(node_id)))

    @staticmethod
    def get_weight_function(G: nx.DiGraph) -> Callable[[Any, Any, Dict], float]:
        """
        Retorna uma função (u, v, d) -> peso para uso em nx.dijkstra_path, nx.astar_path, etc.
        Chuva (KEY_RAIN_MULTIPLIER) aplica-se a toda a área.
        """
        override = G.graph.get(KEY_EDGE_OVERRIDE, {})
        edge_multiplier = G.graph.get(KEY_EDGE_COST_MULTIPLIER, {})
        rain = G.graph.get(KEY_RAIN_MULTIPLIER, 1.0)
        slope_factor = G.graph.get(KEY_SLOPE_PENALTY_FACTOR, 1.0)

        def weight(u: Any, v: Any, d: Dict) -> float:
            key = (u, v)
            if key in override:
                return override[key]
            base = GraphOperations.compute_edge_cost(d, rain, slope_factor)
            return base * edge_multiplier.get(key, 1.0)

        return weight

    @staticmethod
    def get_edge_cost(G: nx.DiGraph, u: Any, v: Any) -> float:
        """Custo atual da aresta (u, v) com base nos atributos e no cenário em G.graph."""
        if not G.has_edge(u, v):
            return float("inf")
        wf = GraphOperations.get_weight_function(G)
        return wf(u, v, G.edges[u, v])

    @staticmethod
    def validate_path_nodes(G: nx.DiGraph, start: Any, goal: Any) -> None:
        """
        Levanta NetworkXError se start ou goal não existirem no grafo.
        """
        missing = [n for n in (start, goal) if n not in G]
        if not missing:
            return
        nodes_list = list(G.nodes())[:15]
        hint = (
            "Grafo OSM (build_op_graph): use nós do grafo, ex. START/GOAL em G.nodes(). "
            "Nós neste grafo (amostra): "
        ) + str(nodes_list) + ("..." if len(G) > 15 else "")
        raise nx.NetworkXError(f"Nó(s) {missing} não existem no grafo. {hint}")

    @staticmethod
    def get_straight_line_distance(G: nx.DiGraph, u: Any, v: Any) -> float:
        """Distância em linha reta entre dois nós (atributo 'pos' = (x, y)). Para heurística A*."""
        if u not in G.nodes or v not in G.nodes:
            return float("inf")
        pos_u = G.nodes[u].get("pos", (0, 0))
        pos_v = G.nodes[v].get("pos", (0, 0))
        return math.sqrt((pos_u[0] - pos_v[0]) ** 2 + (pos_u[1] - pos_v[1]) ** 2)

    @staticmethod
    def path_cost(G: nx.DiGraph, path: list) -> float:
        """Custo total de um caminho (lista de nós)."""
        if len(path) < 2:
            return 0.0
        total = 0.0
        wf = GraphOperations.get_weight_function(G)
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if not G.has_edge(u, v):
                return float("inf")
            total += wf(u, v, G.edges[u, v])
        return total
