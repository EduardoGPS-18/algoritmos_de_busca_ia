"""
Cenários de teste (evento, clima, congestionamento). Responsável por aplicar e limpar cenários no grafo.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import networkx as nx

from .graph_operations import (
    KEY_CONGESTION_FACTOR,
    KEY_CONGESTION_FACTOR_BY_EDGE,
    KEY_CONGESTION_FACTOR_BY_REGION,
    KEY_EDGE_COST_MULTIPLIER,
    KEY_EDGE_OVERRIDE,
    KEY_RAIN_MULTIPLIER,
    KEY_RAIN_MULTIPLIER_BY_REGION,
    KEY_SLOPE_PENALTY_FACTOR,
)


class TestCases:
    """Responsável por ter os cenários de testes (aplicar/limpar evento, clima, congestionamento)."""

    # Arestas interditadas no cenário de evento (origem, destino)
    EVENT_BLOCKED_EDGES: List[Tuple[str, str]] = [
        ("praca_tiradentes", "diogo_vasconcelos"),
        ("diogo_vasconcelos", "praca_tiradentes"),
    ]

    BARRIERS_EXTRA_OURO_PRETO: List[Tuple[str, str]] = [
        ("praca_tiradentes", "diogo_vasconcelos"),
        ("diogo_vasconcelos", "praca_tiradentes"),
        ("centro", "sao_jose"),
        ("sao_jose", "centro"),
        ("praca_tiradentes", "rua_pilar"),
        ("rua_pilar", "praca_tiradentes"),
    ]

    CONGESTED_EDGES_OURO_PRETO: List[Tuple[str, str]] = [
        ("centro", "sao_jose"),
        ("sao_jose", "centro"),
    ]

    CONGESTED_EDGES_REGIONAL_OP: List[Tuple[str, str]] = [
        ("op_tiradentes", "op_centro"),
        ("op_centro", "op_tiradentes"),
        ("op_centro", "op_antonio_dias"),
        ("op_antonio_dias", "op_centro"),
    ]

    OURO_PRETO_BLOCKED_CASE1: List[Tuple[str, str]] = [
        ("rua_rio_negro", "ladeira_joão_de_paiva"),
        ("rua_rio_negro", "way_1428886859"),
        ("rua_rio_piracicaba", "way_385991379"),
        ("rua_rio_piracicaba", "rua_rio_verde"),
        ("rua_rio_piracicaba", "way_385994085"),
        ("rua_rio_piracicaba", "way_385994083"),
    ]

    OURO_PRETO_CONGESTED_CASE2: dict = {
        ("ladeira_joão_de_paiva", "rua_hugo_soderi"): 2.5,
        ("rua_hugo_soderi", "ladeira_joão_de_paiva"): 2.5,
        ("rua_hugo_soderi", "avenida_américo_rené_gianetti"): 2.0,
        ("avenida_américo_rené_gianetti", "rua_hugo_soderi"): 2.0,
    }

    OURO_PRETO_RAIN_REGIONS_CASE3: dict = {
        "ladeira_joão_de_paiva": 2.0,
        "rua_hugo_soderi": 2.0,
        "avenida_américo_rené_gianetti": 1.8,
    }

    OURO_PRETO_BLOCKED_CASE4: List[Tuple[str, str]] = [
        ("rua_rio_piracicaba", "ladeira_joão_de_paiva"),
        ("ladeira_joão_de_paiva", "rua_rio_piracicaba"),
    ]
    OURO_PRETO_CONGESTED_CASE4: dict = {
        ("rua_hugo_soderi", "avenida_américo_rené_gianetti"): 3.0,
        ("avenida_américo_rené_gianetti", "rua_hugo_soderi"): 3.0,
    }

    OURO_PRETO_BLOCKED_CASE5: List[Tuple[str, str]] = [
        ("ladeira_joão_de_paiva", "rua_hugo_soderi"),
        ("rua_hugo_soderi", "ladeira_joão_de_paiva"),
    ]

    OURO_PRETO_CONGESTED_CASE6: dict = {
        ("rua_rio_piracicaba", "ladeira_joão_de_paiva"): 2.0,
        ("ladeira_joão_de_paiva", "rua_rio_piracicaba"): 2.0,
        ("ladeira_joão_de_paiva", "rua_hugo_soderi"): 2.5,
        ("rua_hugo_soderi", "ladeira_joão_de_paiva"): 2.5,
        ("rua_hugo_soderi", "avenida_américo_rené_gianetti"): 2.0,
        ("avenida_américo_rené_gianetti", "rua_hugo_soderi"): 2.0,
    }

    OURO_PRETO_BLOCKED_CASE7: List[Tuple[str, str]] = [
        ("rua_hugo_soderi", "avenida_américo_rené_gianetti"),
        ("avenida_américo_rené_gianetti", "rua_hugo_soderi"),
    ]
    OURO_PRETO_RAIN_REGIONS_CASE7: dict = {
        "ladeira_joão_de_paiva": 2.5,
        "rua_hugo_soderi": 2.5,
        "rua_rio_piracicaba": 2.0,
    }

    BARRIERS_EXTRA_REGIONAL: List[Tuple[str, str]] = [
        ("op_tiradentes", "op_centro"),
        ("op_centro", "op_tiradentes"),
        ("op_centro", "op_antonio_dias"),
        ("op_antonio_dias", "op_centro"),
        ("mariana_centro", "mariana_terminal"),
        ("mariana_terminal", "mariana_centro"),
    ]

    @staticmethod
    def apply_event_scenario(
        G: nx.DiGraph,
        blocked_edges: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """Cenário de evento: interditar ruas (custo infinito nas arestas em blocked_edges)."""
        if blocked_edges is None:
            blocked_edges = TestCases.EVENT_BLOCKED_EDGES
        override = G.graph.setdefault(KEY_EDGE_OVERRIDE, {})
        for (u, v) in blocked_edges:
            override[(u, v)] = float("inf")

    @staticmethod
    def clear_event_scenario(
        G: nx.DiGraph,
        blocked_edges: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """Remove interdições do cenário de evento."""
        if blocked_edges is None:
            blocked_edges = TestCases.EVENT_BLOCKED_EDGES
        override = G.graph.get(KEY_EDGE_OVERRIDE, {})
        for (u, v) in blocked_edges:
            override.pop((u, v), None)

    @staticmethod
    def apply_barriers(G: nx.DiGraph, blocked_edges: List[Tuple[str, str]]) -> None:
        """Aplica barreiras (interdições) em vários trechos."""
        override = G.graph.setdefault(KEY_EDGE_OVERRIDE, {})
        for (u, v) in blocked_edges:
            override[(u, v)] = float("inf")

    @staticmethod
    def apply_traffic_slowdown(G: nx.DiGraph, edge_multipliers: dict) -> None:
        """Lentidão de trânsito em trechos específicos: multiplicador de custo por aresta."""
        G.graph.setdefault(KEY_EDGE_COST_MULTIPLIER, {}).update(edge_multipliers)

    @staticmethod
    def clear_traffic_slowdown(G: nx.DiGraph) -> None:
        """Remove lentidão por trecho."""
        G.graph[KEY_EDGE_COST_MULTIPLIER] = {}

    @staticmethod
    def apply_climate_scenario(G: nx.DiGraph, rain_multiplier: float = 2.0) -> None:
        """Cenário climático global (chuva): ladeiras íngremes com peso ~200% maior."""
        G.graph[KEY_RAIN_MULTIPLIER] = rain_multiplier

    @staticmethod
    def apply_climate_scenario_by_region(G: nx.DiGraph, region_multipliers: dict) -> None:
        """Cenário climático por região."""
        G.graph.setdefault(KEY_RAIN_MULTIPLIER_BY_REGION, {}).update(region_multipliers)

    @staticmethod
    def clear_climate_scenario(G: nx.DiGraph) -> None:
        """Volta multiplicador de chuva ao normal."""
        G.graph[KEY_RAIN_MULTIPLIER] = 1.0
        G.graph[KEY_RAIN_MULTIPLIER_BY_REGION] = {}

    @staticmethod
    def apply_congestion_scenario(G: nx.DiGraph, congestion_factor: float = 2.0) -> None:
        """Cenário de congestionamento global."""
        G.graph[KEY_CONGESTION_FACTOR] = congestion_factor

    @staticmethod
    def apply_congestion_scenario_by_region(G: nx.DiGraph, region_factors: dict) -> None:
        """Cenário de congestionamento por região."""
        G.graph.setdefault(KEY_CONGESTION_FACTOR_BY_REGION, {}).update(region_factors)

    @staticmethod
    def apply_congestion_scenario_by_edge(G: nx.DiGraph, edge_factors: dict) -> None:
        """Cenário de congestionamento por via (aresta)."""
        G.graph.setdefault(KEY_CONGESTION_FACTOR_BY_EDGE, {}).update(edge_factors)

    @staticmethod
    def clear_congestion_scenario(G: nx.DiGraph) -> None:
        """Volta fator de congestionamento ao normal."""
        G.graph[KEY_CONGESTION_FACTOR] = 1.0
        G.graph[KEY_CONGESTION_FACTOR_BY_REGION] = {}
        G.graph[KEY_CONGESTION_FACTOR_BY_EDGE] = {}

    @staticmethod
    def reset_scenarios(G: nx.DiGraph) -> None:
        """Remove todos os overrides e restaura parâmetros padrão."""
        G.graph[KEY_EDGE_OVERRIDE] = {}
        G.graph[KEY_EDGE_COST_MULTIPLIER] = {}
        G.graph[KEY_RAIN_MULTIPLIER] = 1.0
        G.graph[KEY_RAIN_MULTIPLIER_BY_REGION] = {}
        G.graph[KEY_SLOPE_PENALTY_FACTOR] = 1.0
        G.graph[KEY_CONGESTION_FACTOR] = 1.0
        G.graph[KEY_CONGESTION_FACTOR_BY_REGION] = {}
        G.graph[KEY_CONGESTION_FACTOR_BY_EDGE] = {}
