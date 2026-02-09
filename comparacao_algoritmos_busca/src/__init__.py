# Comparação de Algoritmos de Busca em Grafos - IA UFOP
# Roteamento em Ouro Preto (grafo NetworkX)

from pathlib import Path

from dotenv import load_dotenv

# Carrega .env da raiz do projeto (sobe do diretório do pacote até encontrar .env)
_package_dir = Path(__file__).resolve().parent
_root = _package_dir.parent
for _candidate in [_root, _root.parent]:
    _env_file = _candidate / ".env"
    if _env_file.is_file():
        load_dotenv(_env_file)
        break

from .graph import (
    build_ouro_preto_example,
    get_edge_cost,
    get_weight_function,
    path_cost,
)
from .algorithms import dijkstra, a_star, d_star_lite

__all__ = [
    "build_ouro_preto_example",
    "get_edge_cost",
    "get_weight_function",
    "path_cost",
    "dijkstra",
    "a_star",
    "d_star_lite",
]
