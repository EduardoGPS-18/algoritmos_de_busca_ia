# Comparação de Algoritmos de Busca em Grafos — IA UFOP

Trabalho da disciplina BCC325 – Inteligência Artificial (UFOP).  
Roteamento em Ouro Preto com função de custo (declividade, rugosidade, congestionamento) e comparação entre **Dijkstra**, **A*** e **D* Lite**.

## Estrutura

- **`contexto/`** — Especificação da disciplina, base do trabalho e template SBC.
- **`src/`** — Código Python:
  - `graph.py`: grafo, vértices, arestas e função de custo.
  - `algorithms/`: `dijkstra`, `a_star`, `d_star_lite`.
  - `scenarios.py`: cenário de evento (ruas interditadas) e climático (chuva).
  - `metrics.py`: latência, custo do caminho, proxy de vazão, GEH.
- **`experimentos.ipynb`** — Notebook Jupyter para rodar experimentos e gerar resultados para o relatório.
- **`relatorio/`** — Relatório em LaTeX (template SBC, Overleaf).

## Grafo a partir do OpenStreetMap (Overpass)

O grafo é gerado via **Overpass API** (OSM):

- **`BuildGraph().build_op_graph(levels=4, use_cache=True)`** — constrói o grafo das ruas de Ouro Preto (sede) a partir de ways conectadas, com cache em `cache/grafo_op_osm.gpickle`.

Não é necessária API key do Google para construir o grafo (apenas conversão de coordenadas usa helpers que podem vir do `GoogleApiClient`).

## Como rodar

1. Instalar dependências: `pip install -r requirements.txt`
2. Abrir `experimentos.ipynb` e executar as células (kernel com cwd = pasta do projeto).

Ou no terminal:

```bash
cd comparacao_algoritmos_busca
python3 -c "from src.build_graph import BuildGraph; from src.algorithms import dijkstra; G = BuildGraph().build_op_graph(use_cache=True); nodes = list(G.nodes()); print(dijkstra(G, nodes[0], nodes[-1]) if len(nodes) >= 2 else 'Grafo com poucos nós')"
```

## Relatório

- Até 9 páginas, formato SBC.
- Seções: Introdução, Problema, Técnica de IA, Implementação, Cenários, Resultados e Análise, Conclusões, Referências.
- Em caso de uso de modelos de linguagem: preencher ANEXO I com os prompts utilizados.

## Critérios de avaliação (especificação)

- Qualidade da implementação (10%)
- Adequação da técnica de IA (20%)
- Análise de resultados (35%)
- Clareza na documentação e apresentação (35%)
