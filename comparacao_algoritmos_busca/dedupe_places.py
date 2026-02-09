"""
Script one-off: geocodifica DEFAULT_OURO_PRETO_PLACES via Google API, detecta
itens com mesmo (lat, lng) e remove duplicatas da lista em src/graph.py.
Requer .env com GOOGLE_MAPS_API_KEY. Usa cache em cache/google_maps_api_cache.pickle se existir.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

# Projeto = pasta que contém .env e src/
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# Carregar .env
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    raise SystemExit("Defina GOOGLE_MAPS_API_KEY no .env")

# Importar após env carregado
import sys
sys.path.insert(0, str(PROJECT_ROOT))
from src.graph import _geocode, DEFAULT_OURO_PRETO_PLACES, _CACHE_GOOGLE_MAPS, _CACHE_DIR

def main():
    # Cache de geocode (evita requests já feitos)
    cache = {"geocode": {}, "directions": {}}
    if _CACHE_GOOGLE_MAPS.is_file():
        try:
            with open(_CACHE_GOOGLE_MAPS, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            pass
    if "geocode" not in cache:
        cache["geocode"] = {}

    # Geocodificar cada item (request ou cache)
    place_to_coords = {}
    for p in DEFAULT_OURO_PRETO_PLACES:
        pid, addr = p["id"], p["address"]
        if addr in cache["geocode"]:
            lat_lng = cache["geocode"][addr]
        else:
            lat_lng = _geocode(API_KEY, addr)
            if lat_lng is None:
                print(f"  AVISO: Geocoding falhou para {addr!r}")
                continue
            cache["geocode"][addr] = lat_lng
        place_to_coords[pid] = (p, lat_lng)

    # Agrupar por (lat, lng) arredondado (6 decimais ≈ mesmo ponto)
    from collections import defaultdict
    key_to_places = defaultdict(list)
    for pid, (p, (lat, lng)) in place_to_coords.items():
        key = (round(lat, 6), round(lng, 6))
        key_to_places[key].append(p)

    # Manter o primeiro de cada grupo; marcar os demais para remoção
    to_remove_ids = set()
    for key, group in key_to_places.items():
        if len(group) > 1:
            # Manter o primeiro (ordem original na lista)
            for p in group[1:]:
                to_remove_ids.add(p["id"])
            lat, lng = key
            ids_here = [x["id"] for x in group]
            print(f"  Mesmo (lat, lng) {lat}, {lng}: {ids_here} -> mantém {group[0]['id']!r}, remove {[x['id'] for x in group[1:]]}")

    # Nova lista preservando ordem e comentários aproximados
    new_places = [p for p in DEFAULT_OURO_PRETO_PLACES if p["id"] not in to_remove_ids]
    removed_count = len(DEFAULT_OURO_PRETO_PLACES) - len(new_places)
    print(f"\nRemovidos {removed_count} itens com mesmo lat/lng. Restam {len(new_places)}.")

    # Atualizar src/graph.py: substituir o conteúdo de DEFAULT_OURO_PRETO_PLACES
    graph_py = PROJECT_ROOT / "src" / "graph.py"
    text = graph_py.read_text(encoding="utf-8")

    # Gerar novo bloco da lista (mesmo formato: dicts com "id" e "address")
    lines = [
        "# Lista padrão: somente ruas da sede de Ouro Preto, MG (sem pontos de interesse).",
        "# Mapeamento: id -> endereço para geocoding e grafo. Duplicatas por (lat, lng) removidas.",
        "DEFAULT_OURO_PRETO_PLACES = [",
    ]
    for p in new_places:
        lines.append(f'    {{\"id\": \"{p["id"]}\", \"address\": \"{p["address"]}\"}},')
    lines.append("]")

    new_block = "\n".join(lines)

    # Encontrar e substituir o bloco antigo (de "# Lista padrão" até "]")
    import re
    pattern = r"(# Lista padrão: somente ruas[^\n]*\n# Mapeamento:[^\n]*\n)DEFAULT_OURO_PRETO_PLACES = \[.*?\n\]"
    if not re.search(pattern, text, re.DOTALL):
        pattern = r"# Lista padrão: somente ruas[^\n]*\n# Mapeamento:[^\n]*\nDEFAULT_OURO_PRETO_PLACES = \[.*?\n\]"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise SystemExit("Bloco DEFAULT_OURO_PRETO_PLACES não encontrado em graph.py")
    old_block = match.group(0)
    # Preservar apenas o cabeçalho comentado e a nova lista
    replacement = "# Lista padrão: somente ruas da sede de Ouro Preto, MG (sem pontos de interesse).\n# Mapeamento: id -> endereço para geocoding e grafo. Duplicatas por (lat, lng) removidas.\n" + new_block
    new_text = text.replace(old_block, replacement, 1)
    graph_py.write_text(new_text, encoding="utf-8")
    print(f"Atualizado {graph_py}.")

    # Salvar cache se houve novas requisições (opcional)
    if cache.get("geocode"):
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(_CACHE_GOOGLE_MAPS, "wb") as f:
                pickle.dump(cache, f)
        except Exception:
            pass

if __name__ == "__main__":
    main()
