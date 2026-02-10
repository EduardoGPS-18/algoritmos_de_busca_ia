"""
Modelagem do grafo para roteamento em Ouro Preto usando NetworkX.

Use as classes diretamente:
  from src.graph import BuildGraph, GraphOperations
  from src.clients import GoogleApiClient, OverpassApiClient
"""

from __future__ import annotations

from .build_graph import BuildGraph, OURO_PRETO_BBOX
from .clients.google_api_client import (
    OURO_PRETO_REF_LAT,
    OURO_PRETO_REF_LNG,
    GoogleApiClient,
)
from .graph_operations import (
    KEY_CONGESTION_FACTOR,
    KEY_CONGESTION_FACTOR_BY_EDGE,
    KEY_CONGESTION_FACTOR_BY_REGION,
    KEY_EDGE_COST_MULTIPLIER,
    KEY_EDGE_OVERRIDE,
    KEY_RAIN_MULTIPLIER,
    KEY_SLOPE_PENALTY_FACTOR,
    REGION_PREFIXES,
    GraphOperations,
)

__all__ = [
    "BuildGraph",
    "OURO_PRETO_BBOX",
    "GoogleApiClient",
    "OURO_PRETO_REF_LAT",
    "OURO_PRETO_REF_LNG",
    "GraphOperations",
    "KEY_CONGESTION_FACTOR",
    "KEY_CONGESTION_FACTOR_BY_EDGE",
    "KEY_CONGESTION_FACTOR_BY_REGION",
    "KEY_EDGE_COST_MULTIPLIER",
    "KEY_EDGE_OVERRIDE",
    "KEY_RAIN_MULTIPLIER",
    "KEY_SLOPE_PENALTY_FACTOR",
    "REGION_PREFIXES",
]
