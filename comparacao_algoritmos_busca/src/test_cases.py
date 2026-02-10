"""
Cenários de teste (evento, clima, congestionamento). Responsável por aplicar e limpar cenários no grafo.
Origen e destino fixos: START -> GOAL (rua_joão_pedro_da_silva -> rua_rio_piracicaba).
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
    KEY_SLOPE_PENALTY_FACTOR,
)

# Constantes globais do roteamento (sempre usar estas)
START = "rua_joão_pedro_da_silva"
GOAL = "rua_rio_piracicaba"


class TestCases:
    """
    Cenários de teste para START -> GOAL.
    Bloqueios e trânsito são espalhados ao longo de diversos trechos (início, meio e fim do percurso),
    com fatores distintos por aresta.
    """

    # ---- Chuva (multiplicador global) ----
    RAIN_LIGHT = 1.5
    RAIN_MEDIUM = 2.0
    RAIN_HEAVY = 3.0

    # ---- Simples: um tipo de impeditivo em um trecho ----
    # Caso 1: uma via bloqueada (início do percurso)
    BLOCKED_CASE1: List[Tuple[str, str]] = [
        ("rua_joão_pedro_da_silva", "way_193208512"),
    ]

    # Caso 2: uma via bloqueada (trecho próximo ao GOAL)
    BLOCKED_CASE2: List[Tuple[str, str]] = [
        ("rua_rio_verde", "rua_rio_piracicaba"),
        ("rua_rio_piracicaba", "rua_rio_verde"),
    ]

    # Caso 3: chuva leve (toda a área)

    # Caso 4: trânsito em um trecho só, com fator único
    CONGESTED_CASE4: dict = {
        ("rua_rio_verde", "rua_rio_piracicaba"): 2.0,
        ("rua_rio_piracicaba", "rua_rio_verde"): 2.0,
    }

    # ---- Médio: impeditivos em trechos diferentes (início + meio ou meio + fim) ----
    # Caso 5: duas vias bloqueadas em trechos diferentes (início + fim)
    BLOCKED_CASE5: List[Tuple[str, str]] = [
        ("rua_joão_pedro_da_silva", "way_193208512"),
        ("rua_rio_verde", "rua_rio_piracicaba"),
        ("rua_rio_piracicaba", "rua_rio_verde"),
    ]

    # Caso 6: chuva + uma via bloqueada (meio do percurso)
    BLOCKED_CASE6: List[Tuple[str, str]] = [
        ("ladeira_joão_de_paiva", "rua_rio_negro"),
        ("rua_rio_negro", "ladeira_joão_de_paiva"),
    ]

    # Caso 7: trânsito espalhado em vários trechos, cada um com fator diferente
    CONGESTED_CASE7: dict = {
        # trecho início
        ("way_193208512", "rua_professor_paulo_magalhães_gomes"): 1.5,
        ("rua_professor_paulo_magalhães_gomes", "way_193208512"): 1.5,
        # trecho meio
        ("travessa_roque_de_paiva", "rua_henrique_goerceix"): 1.8,
        ("rua_henrique_goerceix", "travessa_roque_de_paiva"): 1.8,
        # trecho fim
        ("rua_rio_verde", "rua_rio_piracicaba"): 2.0,
        ("rua_rio_piracicaba", "rua_rio_verde"): 2.0,
    }

    # ---- Complexo: bloqueios e trânsito em diversas vias ao longo do percurso, com fatores por trecho ----
    # Caso 8: várias vias bloqueadas espalhadas (início, meio e fim)
    BLOCKED_CASE8: List[Tuple[str, str]] = [
        ("rua_joão_pedro_da_silva", "way_193208512"),
        ("rua_henrique_goerceix", "ladeira_joão_de_paiva_194479756"),
        ("ladeira_joão_de_paiva_194479756", "rua_henrique_goerceix"),
        ("rua_rio_negro", "way_385994082"),
        ("way_385994082", "rua_rio_negro"),
        ("rua_rio_verde", "rua_rio_piracicaba"),
        ("rua_rio_piracicaba", "rua_rio_verde"),
        ("way_385994085", "rua_rio_piracicaba"),
        ("rua_rio_piracicaba", "way_385994085"),
    ]

    # Caso 9: trânsito em diversas vias com fatores diferentes por ponto
    CONGESTED_CASE9: dict = {
        # início
        ("way_193208512", "rua_professor_paulo_magalhães_gomes"): 1.5,
        ("rua_professor_paulo_magalhães_gomes", "way_193208512"): 1.5,
        # meio 1
        ("rua_barão_de_camargos", "rua_professor_ros_p_gomes"): 2.0,
        ("rua_professor_ros_p_gomes", "rua_barão_de_camargos"): 2.0,
        # meio 2
        ("travessa_roque_de_paiva", "rua_henrique_goerceix"): 2.2,
        ("rua_henrique_goerceix", "travessa_roque_de_paiva"): 2.2,
        # meio 3
        ("ladeira_joão_de_paiva", "rua_rio_negro"): 2.5,
        ("rua_rio_negro", "ladeira_joão_de_paiva"): 2.5,
        # fim
        ("way_385994082", "rua_rio_verde"): 1.8,
        ("rua_rio_verde", "way_385994082"): 1.8,
        ("rua_rio_verde", "rua_rio_piracicaba"): 3.0,
        ("rua_rio_piracicaba", "rua_rio_verde"): 3.0,
        ("way_385994085", "rua_rio_piracicaba"): 2.5,
        ("rua_rio_piracicaba", "way_385994085"): 2.5,
    }

    # Caso 10: chuva intensa + bloqueios e trânsito espalhados com fatores por trecho
    BLOCKED_CASE10: List[Tuple[str, str]] = [
        ("rua_joão_pedro_da_silva", "way_193208512"),
        ("ladeira_joão_de_paiva", "rua_rio_negro"),
        ("rua_rio_negro", "ladeira_joão_de_paiva"),
        ("rua_rio_verde", "rua_rio_piracicaba"),
        ("rua_rio_piracicaba", "rua_rio_verde"),
    ]
    CONGESTED_CASE10: dict = {
        ("way_193208512", "rua_professor_paulo_magalhães_gomes"): 1.8,
        ("rua_professor_paulo_magalhães_gomes", "way_193208512"): 1.8,
        ("travessa_roque_de_paiva", "rua_henrique_goerceix"): 2.0,
        ("rua_henrique_goerceix", "travessa_roque_de_paiva"): 2.0,
        ("way_385994082", "rua_rio_verde"): 2.2,
        ("rua_rio_verde", "way_385994082"): 2.2,
        ("way_385994085", "rua_rio_piracicaba"): 2.5,
        ("rua_rio_piracicaba", "way_385994085"): 2.5,
    }

    @staticmethod
    def apply_event_scenario(
        G: nx.DiGraph,
        blocked_edges: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """Cenário de evento: interditar ruas (custo infinito nas arestas em blocked_edges)."""
        if blocked_edges is None:
            blocked_edges = TestCases.BLOCKED_CASE1
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
            blocked_edges = TestCases.BLOCKED_CASE1
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
        """Cenário climático (chuva em toda a área): penalidade de declividade multiplicada."""
        G.graph[KEY_RAIN_MULTIPLIER] = rain_multiplier

    @staticmethod
    def clear_climate_scenario(G: nx.DiGraph) -> None:
        """Volta multiplicador de chuva ao normal."""
        G.graph[KEY_RAIN_MULTIPLIER] = 1.0

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
        G.graph.setdefault(KEY_EDGE_COST_MULTIPLIER, {}).update(edge_factors)

    @staticmethod
    def clear_congestion_scenario(G: nx.DiGraph) -> None:
        """Volta fator de congestionamento ao normal."""
        G.graph[KEY_CONGESTION_FACTOR] = 1.0
        G.graph[KEY_CONGESTION_FACTOR_BY_REGION] = {}
        G.graph[KEY_EDGE_COST_MULTIPLIER] = {}

    @staticmethod
    def reset_scenarios(G: nx.DiGraph) -> None:
        """Remove todos os overrides e restaura parâmetros padrão."""
        G.graph[KEY_EDGE_OVERRIDE] = {}
        G.graph[KEY_EDGE_COST_MULTIPLIER] = {}
        G.graph[KEY_RAIN_MULTIPLIER] = 1.0
        G.graph[KEY_SLOPE_PENALTY_FACTOR] = 1.0
        G.graph[KEY_CONGESTION_FACTOR] = 1.0
        G.graph[KEY_CONGESTION_FACTOR_BY_REGION] = {}
        G.graph[KEY_EDGE_COST_MULTIPLIER] = {}
