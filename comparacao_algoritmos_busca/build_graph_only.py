#!/usr/bin/env python3
"""
Apenas constrói o grafo de Ouro Preto e suas conexões (Geocoding + OSM conectividade + Directions).
Salva em cache/grafo_ouro_preto.gpickle. Não executa experimentos nem exibe o grafo.

Uso (na raiz do projeto):
  python build_graph_only.py

Requer .env com GOOGLE_MAPS_API_KEY.
"""
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

def main():
    from src.graph import build_ouro_preto_example, _CACHE_OURO_PRETO

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("Defina GOOGLE_MAPS_API_KEY no .env")
        sys.exit(1)

    print("Construindo grafo (Geocoding + OSM + Directions)...")
    G = build_ouro_preto_example(api_key=api_key, use_cache=True, force_rebuild=True)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"Grafo construído: {n_nodes} nós, {n_edges} arestas.")
    print(f"Salvo em: {_CACHE_OURO_PRETO}")

if __name__ == "__main__":
    main()
