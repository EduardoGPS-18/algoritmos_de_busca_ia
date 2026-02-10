# Comparação de Algoritmos de Busca em Grafos - IA UFOP
# Roteamento em Ouro Preto (grafo NetworkX)

from pathlib import Path

from dotenv import load_dotenv

# Carrega .env da raiz do projeto
_package_dir = Path(__file__).resolve().parent
_root = _package_dir.parent
for _candidate in [_root, _root.parent]:
    _env_file = _candidate / ".env"
    if _env_file.is_file():
        load_dotenv(_env_file)
        break
