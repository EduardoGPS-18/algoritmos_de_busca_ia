"""
Grafo editável simples: input + botão Adicionar, sem validação e sem API.
Apenas adiciona um nó na tela com o texto digitado.
"""

from __future__ import annotations

from typing import Any, Dict, List

import dash_cytoscape as cyto  # type: ignore[reportMissingImports]
from dash import Dash, Input, Output, State, html, dcc, no_update


def _state_to_elements(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Converte estado {nodes: [{id, label, x, y}]} em elementos Cytoscape."""
    nodes = state.get("nodes", [])
    return [
        {
            "data": {"id": n["id"], "label": n.get("label", n["id"])},
            "position": {"x": n["x"], "y": n["y"]},
        }
        for n in nodes
    ]


def display_simple_editable_graph(
    display_in_notebook: bool = True,
    iframe_height: int = 500,
) -> Any:
    """
    Exibe um grafo mínimo: campo de texto + botão Adicionar.
    Ao clicar, adiciona um nó com o texto digitado (sem validação, sem API).
    """
    app = Dash(__name__)
    initial_state = {"nodes": [], "edges": []}

    app.layout = html.Div([
        dcc.Store(id="graph-store", data=initial_state),
        html.Div([
            dcc.Input(
                id="search-address",
                type="text",
                placeholder="Digite um nome para o nó",
                style={
                    "width": "200px",
                    "padding": "8px",
                    "marginRight": "8px",
                },
            ),
            html.Button(
                "Adicionar",
                id="add-place-btn",
                style={"padding": "8px 16px", "cursor": "pointer"},
            ),
        ], style={"marginBottom": "12px"}),
        html.Div(id="status", children="Digite e clique em Adicionar.", style={"marginBottom": "8px", "fontSize": "12px"}),
        cyto.Cytoscape(
            id="cytoscape-simple",
            elements=[],
            layout={"name": "preset", "fit": True, "padding": 20},
            style={"width": "100%", "height": "%dpx" % iframe_height, "backgroundColor": "#f0f0f0"},
            stylesheet=[
                {"selector": "node", "style": {"content": "data(label)", "background-color": "#7bc", "font-size": "12px"}},
                {"selector": "edge", "style": {"width": 2, "line-color": "#999"}},
            ],
        ),
    ])

    @app.callback(
        [Output("graph-store", "data"), Output("search-address", "value"), Output("status", "children")],
        Input("add-place-btn", "n_clicks"),
        State("graph-store", "data"),
        State("search-address", "value"),
        prevent_initial_call=True,
    )
    def add_node(n_clicks, current_state, value):
        state = current_state or initial_state
        nodes = list(state.get("nodes", []))
        n = len(nodes)
        node_id = "node_%d" % n
        label = (value or "").strip() or node_id
        x, y = 80 * (n % 5), 80 * (n // 5)
        nodes.append({"id": node_id, "label": label, "x": x, "y": y})
        new_state = {"nodes": nodes, "edges": state.get("edges", [])}
        return new_state, "", "Adicionado: %s" % label

    @app.callback(
        Output("cytoscape-simple", "elements"),
        Input("graph-store", "data"),
    )
    def store_to_elements(data):
        if not data:
            return []
        return _state_to_elements(data)

    if display_in_notebook:
        app.run(jupyter_mode="inline", jupyter_height=iframe_height + 150, use_reloader=False, port=0)
    else:
        app.run(use_reloader=False, port=8051)
    return app
