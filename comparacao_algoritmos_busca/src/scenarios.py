"""
Cenários de simulação conforme base_trabalho_ia.pdf:

1. Cenário de Evento: peso infinito em ruas interditadas (ex: Diogo de Vasconcelos, Praça Tiradentes).
   O sistema deve sugerir rotas alternativas (Rua do Pilar, Xavier da Veiga).

2. Cenário Climático: aumentar pesos das ladeiras íngremes em 200% (simular chuva).
"""

from __future__ import annotations
from typing import List, Optional, Tuple

import networkx as nx

from .graph import (
    KEY_CONGESTION_FACTOR,
    KEY_CONGESTION_FACTOR_BY_REGION,
    KEY_EDGE_COST_MULTIPLIER,
    KEY_EDGE_OVERRIDE,
    KEY_RAIN_MULTIPLIER,
    KEY_RAIN_MULTIPLIER_BY_REGION,
    KEY_SLOPE_PENALTY_FACTOR,
)

# Arestas interditadas no cenário de evento (origem, destino)
EVENT_BLOCKED_EDGES = [
    ("praca_tiradentes", "diogo_vasconcelos"),
    ("diogo_vasconcelos", "praca_tiradentes"),
]

# Barreiras adicionais (mais trechos interditados) — Ouro Preto
BARRIERS_EXTRA_OURO_PRETO = [
    ("praca_tiradentes", "diogo_vasconcelos"),
    ("diogo_vasconcelos", "praca_tiradentes"),
    ("centro", "sao_jose"),
    ("sao_jose", "centro"),
    ("praca_tiradentes", "rua_pilar"),
    ("rua_pilar", "praca_tiradentes"),
]

# Barreiras adicionais — grafo regional (OP + Mariana + Cachoeira)
BARRIERS_EXTRA_REGIONAL = [
    ("op_tiradentes", "op_centro"),
    ("op_centro", "op_tiradentes"),
    ("op_centro", "op_antonio_dias"),
    ("op_antonio_dias", "op_centro"),
    ("mariana_centro", "mariana_terminal"),
    ("mariana_terminal", "mariana_centro"),
]


def apply_event_scenario(
    G: nx.DiGraph,
    blocked_edges: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """
    Cenário de evento (Semana Santa / Festival): interditar ruas.
    Atribui custo infinito às arestas em blocked_edges.
    """
    if blocked_edges is None:
        blocked_edges = EVENT_BLOCKED_EDGES
    override = G.graph.setdefault(KEY_EDGE_OVERRIDE, {})
    for (u, v) in blocked_edges:
        override[(u, v)] = float("inf")


def clear_event_scenario(
    G: nx.DiGraph,
    blocked_edges: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """Remove interdições do cenário de evento."""
    if blocked_edges is None:
        blocked_edges = EVENT_BLOCKED_EDGES
    override = G.graph.get(KEY_EDGE_OVERRIDE, {})
    for (u, v) in blocked_edges:
        override.pop((u, v), None)


def apply_barriers(
    G: nx.DiGraph,
    blocked_edges: List[Tuple[str, str]],
) -> None:
    """
    Aplica barreiras (interdições) em vários trechos.
    blocked_edges: lista de (origem, destino) com custo infinito.
    """
    override = G.graph.setdefault(KEY_EDGE_OVERRIDE, {})
    for (u, v) in blocked_edges:
        override[(u, v)] = float("inf")


def apply_traffic_slowdown(
    G: nx.DiGraph,
    edge_multipliers: dict,
) -> None:
    """
    Lentidão de trânsito em trechos específicos: multiplicador de custo por aresta.
    edge_multipliers: dict (u, v) -> multiplicador (ex.: 1.5 = 50% mais lento, 2.0 = dobro do tempo).
    """
    G.graph.setdefault(KEY_EDGE_COST_MULTIPLIER, {}).update(edge_multipliers)


def clear_traffic_slowdown(G: nx.DiGraph) -> None:
    """Remove lentidão por trecho."""
    G.graph[KEY_EDGE_COST_MULTIPLIER] = {}


def apply_climate_scenario(G: nx.DiGraph, rain_multiplier: float = 2.0) -> None:
    """
    Cenário climático global (chuva): ladeiras íngremes com peso ~200% maior em todo o grafo.
    """
    G.graph[KEY_RAIN_MULTIPLIER] = rain_multiplier


def apply_climate_scenario_by_region(
    G: nx.DiGraph,
    region_multipliers: dict,
) -> None:
    """
    Cenário climático por região: alguns bairros/áreas com chuva, outros não.
    region_multipliers: dict região -> multiplicador (ex: {"op": 2.0, "mariana": 1.0};
      ou por bairro no grafo Ouro Preto: {"centro": 2.0, "praca_tiradentes": 2.0, "campus": 1.0}).
    No grafo regional: regiões "op", "mariana", "cachoeira". No grafo só Ouro Preto: região = id do nó (centro, campus, etc.).
    """
    G.graph.setdefault(KEY_RAIN_MULTIPLIER_BY_REGION, {}).update(region_multipliers)


def clear_climate_scenario(G: nx.DiGraph) -> None:
    """Volta multiplicador de chuva ao normal (global e por região)."""
    G.graph[KEY_RAIN_MULTIPLIER] = 1.0
    G.graph[KEY_RAIN_MULTIPLIER_BY_REGION] = {}


def apply_congestion_scenario(G: nx.DiGraph, congestion_factor: float = 2.0) -> None:
    """
    Cenário de congestionamento global: alta densidade em todo o grafo.
    congestion_factor > 1 amplifica o efeito do volume_capacity_ratio nas arestas.
    """
    G.graph[KEY_CONGESTION_FACTOR] = congestion_factor


def apply_congestion_scenario_by_region(
    G: nx.DiGraph,
    region_factors: dict,
) -> None:
    """
    Cenário de congestionamento por região: alguns trechos sobrecarregados, outros não.
    region_factors: dict região -> fator (ex: {"op": 2.0, "mariana": 1.0};
      ou por bairro: {"centro": 2.0, "sao_jose": 2.0}).
    No grafo regional: "op", "mariana", "cachoeira". No grafo Ouro Preto: id do nó.
    """
    G.graph.setdefault(KEY_CONGESTION_FACTOR_BY_REGION, {}).update(region_factors)


def clear_congestion_scenario(G: nx.DiGraph) -> None:
    """Volta fator de congestionamento ao normal (global e por região)."""
    G.graph[KEY_CONGESTION_FACTOR] = 1.0
    G.graph[KEY_CONGESTION_FACTOR_BY_REGION] = {}


def reset_scenarios(G: nx.DiGraph) -> None:
    """Remove todos os overrides e restaura parâmetros padrão."""
    G.graph[KEY_EDGE_OVERRIDE] = {}
    G.graph[KEY_EDGE_COST_MULTIPLIER] = {}
    G.graph[KEY_RAIN_MULTIPLIER] = 1.0
    G.graph[KEY_RAIN_MULTIPLIER_BY_REGION] = {}
    G.graph[KEY_SLOPE_PENALTY_FACTOR] = 1.0
    G.graph[KEY_CONGESTION_FACTOR] = 1.0
    G.graph[KEY_CONGESTION_FACTOR_BY_REGION] = {}
