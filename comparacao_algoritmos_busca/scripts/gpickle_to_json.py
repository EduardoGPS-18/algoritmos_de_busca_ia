#!/usr/bin/env python3
"""
Converte grafo .gpickle (NetworkX + pickle) para JSON (formato node-link).

Uso:
    python scripts/gpickle_to_json.py <input.gpickle> <output.json>

Exemplo:
    python comparacao_algoritmos_busca/scripts/gpickle_to_json.py \\
        comparacao_algoritmos_busca/cache/grafo_op_osm.gpickle \\
        comparacao_algoritmos_busca/cache/grafo_op_osm_1.json
"""
import json
import pickle
import sys
from pathlib import Path
from typing import Any

# Permite importar networkx (e src se necessário)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import networkx as nx


def _json_serializable(obj: Any) -> Any:
    """Converte valores para tipos nativos JSON (ex.: numpy.float64 -> float)."""
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_serializable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return obj


def gpickle_to_json(input_path: Path, output_path: Path) -> Path:
    """Carrega grafo do .gpickle, converte para formato node-link e grava em .json."""
    with open(input_path, "rb") as f:
        G = pickle.load(f)
    data = nx.node_link_data(G)
    data = _json_serializable(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output_path


def main() -> None:
    if len(sys.argv) != 3:
        print("Uso: python gpickle_to_json.py <input.gpickle> <output.json>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.is_file():
        print(f"Erro: arquivo de entrada não encontrado: {input_path}", file=sys.stderr)
        sys.exit(1)

    result = gpickle_to_json(input_path, output_path)
    print(f"JSON salvo em: {result}")


if __name__ == "__main__":
    main()
