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

