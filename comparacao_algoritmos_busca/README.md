# Comparação de Algoritmos de Busca em Grafos — IA UFOP

Projeto de comparação de algoritmos de busca (Dijkstra, A*, D* Lite) em grafo de ruas de Ouro Preto (OSM).

## Requisitos

- **Python 3.8+**
- (Opcional) Chave da API Google Maps, para enriquecer o grafo com elevação no primeiro build

## Como rodar

### 1. Clonar / entrar no projeto

```bash
cd comparacao_algoritmos_busca
```

### 2. Criar ambiente virtual (recomendado)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Variáveis de ambiente (opcional)

Para (re)construir o grafo com dados de elevação (Google Elevation API), crie um arquivo `.env` na raiz do projeto:

```bash
GOOGLE_MAPS_API_KEY=sua_chave_aqui
```

Se não definir a chave, o build do grafo pode usar cache existente ou construir sem elevação (conforme o código em `src/build_graph.py`).

### 5. Executar os experimentos

A aplicação é executada via **Jupyter Notebook**:

```bash
jupyter notebook experimentos.ipynb
```

Ou com JupyterLab:

```bash
jupyter lab experimentos.ipynb
```

No notebook, execute as células em ordem:

1. **Células iniciais** — ajuste de `sys.path` e imports.
2. **Grafo base** — carrega ou constrói o grafo de Ouro Preto (`builder.build_op_graph(..., use_cache=True)`).
3. **Cenários** — comparação dos três algoritmos e casos de teste (bloqueios, chuva, trânsito, etc.).

### Cache

O grafo e caches OSM ficam em `cache/`. Com `use_cache=True`, o notebook reutiliza o grafo já construído e evita novas chamadas à API.

## Estrutura resumida

| Pasta/arquivo   | Descrição |
|-----------------|-----------|
| `experimentos.ipynb` | Notebook principal: build do grafo e experimentos |
| `src/`          | Código: algoritmos (Dijkstra, A*, D* Lite), build do grafo, clientes OSM/Google, métricas |
| `cache/`        | Grafos e caches (`.gpickle`, `.json`, etc.) |
| `requirements.txt` | Dependências Python |

## Dependências principais

- `networkx` — grafos
- `requests` — chamadas HTTP (Overpass, Google)
- `python-dotenv` — carregamento de `.env`
- `jupyter`, `notebook`, `ipykernel` — execução do notebook
- Opcionais: `pandas`, `matplotlib`, `dash`, `dash-cytoscape` (análise e visualização)
