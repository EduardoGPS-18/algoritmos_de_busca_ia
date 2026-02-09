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
    KEY_CONGESTION_FACTOR_BY_EDGE,
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

# Congestionamento por via (grafo Ouro Preto): vias centro ↔ são_jose
CONGESTED_EDGES_OURO_PRETO = [
    ("centro", "sao_jose"),
    ("sao_jose", "centro"),
]

# Congestionamento por via (grafo regional): vias em OP (op_tiradentes ↔ op_centro, op_centro ↔ op_antonio_dias)
CONGESTED_EDGES_REGIONAL_OP = [
    ("op_tiradentes", "op_centro"),
    ("op_centro", "op_tiradentes"),
    ("op_centro", "op_antonio_dias"),
    ("op_antonio_dias", "op_centro"),
]

# --- Casos de teste Ouro Preto (grafo_op_osm: START=rua_rio_piracicaba, GOAL=avenida_américo_rené_gianetti) ---
# Nós conforme cache grafo_op_osm.gpickle (slugs OSM).
# Caminho baseline: rua_rio_piracicaba → ladeira_joão_de_paiva → rua_hugo_soderi → avenida_américo_rené_gianetti.
# Impeditivos em arestas desse caminho para forçar desvio ou custo maior.

# Caso 1: Interditar primeira aresta do caminho (rua_rio_piracicaba → rua_rio_verde no grafo_op_osm)
OURO_PRETO_BLOCKED_CASE1 = [
    ("rua_rio_negro", "ladeira_joão_de_paiva"),
    ("rua_rio_negro", "way_1428886859"),
    ("rua_rio_piracicaba", "way_385991379"),
    ("rua_rio_piracicaba", "rua_rio_verde"),
    ("rua_rio_piracicaba", "way_385994085"),
    ("rua_rio_piracicaba", "way_385994083"),
    
]

# Caso 2: Congestionamento em joao_de_paiva ↔ hugo_soderi e hugo_soderi ↔ americo_rene_gianetti
OURO_PRETO_CONGESTED_CASE2 = {
    ("ladeira_joão_de_paiva", "rua_hugo_soderi"): 2.5,
    ("rua_hugo_soderi", "ladeira_joão_de_paiva"): 2.5,
    ("rua_hugo_soderi", "avenida_américo_rené_gianetti"): 2.0,
    ("avenida_américo_rené_gianetti", "rua_hugo_soderi"): 2.0,
}

# Caso 3: Chuva em “regiões” (nós) do caminho — joao_de_paiva, hugo_soderi
OURO_PRETO_RAIN_REGIONS_CASE3 = {
    "ladeira_joão_de_paiva": 2.0,
    "rua_hugo_soderi": 2.0,
    "avenida_américo_rené_gianetti": 1.8,
}

# Caso 4: Interditar rio_piracicaba ↔ joao_de_paiva + congestionamento em hugo_soderi ↔ americo_rene_gianetti
OURO_PRETO_BLOCKED_CASE4 = [
    ("rua_rio_piracicaba", "ladeira_joão_de_paiva"),
    ("ladeira_joão_de_paiva", "rua_rio_piracicaba"),
]
OURO_PRETO_CONGESTED_CASE4 = {
    ("rua_hugo_soderi", "avenida_américo_rené_gianetti"): 3.0,
    ("avenida_américo_rené_gianetti", "rua_hugo_soderi"): 3.0,
}

# Caso 5: Interditar joao_de_paiva ↔ hugo_soderi (trecho central do caminho)
OURO_PRETO_BLOCKED_CASE5 = [
    ("ladeira_joão_de_paiva", "rua_hugo_soderi"),
    ("rua_hugo_soderi", "ladeira_joão_de_paiva"),
]

# Caso 6: Congestionamento em todas as arestas do caminho baseline
OURO_PRETO_CONGESTED_CASE6 = {
    ("rua_rio_piracicaba", "ladeira_joão_de_paiva"): 2.0,
    ("ladeira_joão_de_paiva", "rua_rio_piracicaba"): 2.0,
    ("ladeira_joão_de_paiva", "rua_hugo_soderi"): 2.5,
    ("rua_hugo_soderi", "ladeira_joão_de_paiva"): 2.5,
    ("rua_hugo_soderi", "avenida_américo_rené_gianetti"): 2.0,
    ("avenida_américo_rené_gianetti", "rua_hugo_soderi"): 2.0,
}

# Caso 7: Interditar hugo_soderi ↔ americo_rene_gianetti + chuva em regiões do caminho
OURO_PRETO_BLOCKED_CASE7 = [
    ("rua_hugo_soderi", "avenida_américo_rené_gianetti"),
    ("avenida_américo_rené_gianetti", "rua_hugo_soderi"),
]
OURO_PRETO_RAIN_REGIONS_CASE7 = {
    "ladeira_joão_de_paiva": 2.5,
    "rua_hugo_soderi": 2.5,
    "rua_rio_piracicaba": 2.0,
}

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
    Cenário de congestionamento por região (legado): afeta todas as arestas que saem de nós da região.
    Prefira apply_congestion_scenario_by_edge para impactar apenas vias específicas.
    region_factors: dict região -> fator (ex: {"op": 2.0, "mariana": 1.0}).
    """
    G.graph.setdefault(KEY_CONGESTION_FACTOR_BY_REGION, {}).update(region_factors)


def apply_congestion_scenario_by_edge(
    G: nx.DiGraph,
    edge_factors: dict,
) -> None:
    """
    Cenário de congestionamento por via (aresta): impacta apenas as vias indicadas.
    edge_factors: dict (origem, destino) -> fator (ex: {("centro", "sao_jose"): 2.0}).
    Congestionamento local afeta uma via somente, não um nó inteiro.
    """
    G.graph.setdefault(KEY_CONGESTION_FACTOR_BY_EDGE, {}).update(edge_factors)


def clear_congestion_scenario(G: nx.DiGraph) -> None:
    """Volta fator de congestionamento ao normal (global, por região e por aresta)."""
    G.graph[KEY_CONGESTION_FACTOR] = 1.0
    G.graph[KEY_CONGESTION_FACTOR_BY_REGION] = {}
    G.graph[KEY_CONGESTION_FACTOR_BY_EDGE] = {}


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
