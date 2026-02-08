"""
Métricas de avaliação conforme base_trabalho_ia.pdf:

- Throughput (vazão): veículos que completam o trajeto por hora (simplificado: 1/custo ou tempo).
- Estatística GEH: comparação com contagens reais (placeholder para dados Ourotran).
- Latência de re-roteamento: tempo para sugerir nova rota após bloqueio (< 100 ms para tempo real).
"""

import time
from typing import Any, Callable, List, Optional, Tuple


def measure_latency_ms(
    fn: Callable[[], Any],
    repetitions: int = 1,
) -> Tuple[float, Any]:
    """
    Mede o tempo de execução de fn() em milissegundos.
    Retorna (tempo_medio_ms, resultado da última chamada).
    """
    start = time.perf_counter()
    result = None
    for _ in range(repetitions):
        result = fn()
    elapsed = (time.perf_counter() - start) / repetitions * 1000
    return elapsed, result


def path_cost_from_list(
    cost_fn: Callable[[str, str], float],
    path: List[str],
) -> float:
    """Custo total de um caminho (soma dos custos das arestas)."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        total += cost_fn(path[i], path[i + 1])
    return total


def throughput_proxy(path_cost: float, scale: float = 3600.0) -> float:
    """
    Proxy de vazão: inversamente proporcional ao custo do caminho.
    scale pode ser usado para calibrar "veículos/hora" se custo for tempo em segundos.
    """
    if path_cost <= 0:
        return 0.0
    return scale / path_cost


def geh_statistic(observed: float, simulated: float) -> float:
    """
    Estatística GEH: sqrt(2 * (simulated - observed)^2 / (simulated + observed)).
    Usada para validar volume de tráfego simulado vs. contagens reais.
    """
    if simulated + observed == 0:
        return 0.0
    return (2 * (simulated - observed) ** 2 / (simulated + observed)) ** 0.5
