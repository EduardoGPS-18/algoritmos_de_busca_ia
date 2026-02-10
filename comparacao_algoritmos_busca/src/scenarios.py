"""
Cenários de simulação. Use a classe TestCases diretamente:

  from src.scenarios import TestCases
  TestCases.apply_event_scenario(G, blocked_edges=[...])
"""

from __future__ import annotations

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
from .test_cases import TestCases

__all__ = [
    "TestCases",
    "KEY_CONGESTION_FACTOR",
    "KEY_CONGESTION_FACTOR_BY_EDGE",
    "KEY_CONGESTION_FACTOR_BY_REGION",
    "KEY_EDGE_COST_MULTIPLIER",
    "KEY_EDGE_OVERRIDE",
    "KEY_RAIN_MULTIPLIER",
    "KEY_RAIN_MULTIPLIER_BY_REGION",
    "KEY_SLOPE_PENALTY_FACTOR",
]
