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

## Grafo a partir do Google Maps

O grafo é gerado via **Google Maps** (Geocoding + Directions):

- **`build_ouro_preto_example(api_key=None)`** — usa a lista padrão de pontos em Ouro Preto e chama a API.
- **`build_graph_from_google_maps(places, api_key=None, ...)`** — gera o grafo para uma lista customizada de lugares (`places = [{"id": "...", "address": "..."}, ...]`).

**API key:** o projeto carrega automaticamente o `.env` da **raiz do projeto** (pasta que contém `comparacao_algoritmos_busca` ou a própria pasta do trabalho). Coloque `GOOGLE_MAPS_API_KEY=sua-chave` no `.env` ou defina a variável de ambiente / passe `api_key=...`. É necessário ativar **Geocoding API** e **Directions API** no Google Cloud.

## Como rodar

1. Instalar dependências: `pip install -r requirements.txt`
2. Definir `GOOGLE_MAPS_API_KEY` (ou passar `api_key` ao chamar `build_ouro_preto_example(api_key="...")`).
3. Abrir `experimentos.ipynb` e executar as células (kernel com cwd = pasta do projeto).

Ou no terminal:

```bash
cd comparacao_algoritmos_busca
export GOOGLE_MAPS_API_KEY="sua-chave"
python3 -c "from src.graph import build_ouro_preto_example; from src.algorithms import dijkstra, a_star, d_star_lite; G = build_ouro_preto_example(); print(dijkstra(G, 'praca_tiradentes', 'campus'))"
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
