"""
Helpers para exibição e manipulação de grafos (Dash Cytoscape, posições, etc.).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import dash_cytoscape as cyto
from dash import Dash, Input, Output, html

if TYPE_CHECKING:
    import networkx as nx


def spread_positions(
    pos_dict: Dict[str, Tuple[float, float]],
    min_distance: float = 400,
    iterations: int = 8,
    factor: float = 0.4,
) -> Dict[str, Tuple[float, float]]:
    """Afasta nós muito próximos mantendo o formato geral (evita sobreposição)."""
    pos = {n: (float(x), float(y)) for n, (x, y) in pos_dict.items()}
    for _ in range(iterations):
        disp = {n: (0.0, 0.0) for n in pos}
        nodes = list(pos.keys())
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                dx = pos[u][0] - pos[v][0]
                dy = pos[u][1] - pos[v][1]
                d = (dx * dx + dy * dy) ** 0.5
                if d < min_distance and d > 1e-6:
                    push = factor * (min_distance - d) / 2
                    dx, dy = dx / d, dy / d
                    disp[u] = (disp[u][0] + dx * push, disp[u][1] + dy * push)
                    disp[v] = (disp[v][0] - dx * push, disp[v][1] - dy * push)
        for n in pos:
            pos[n] = (pos[n][0] + disp[n][0], pos[n][1] + disp[n][1])
    return pos


def _short_label(node_id: str, max_len: int = 18) -> str:
    s = node_id.replace("_", " ").title()
    if len(s) > max_len:
        s = s[: max_len - 2] + "…"
    return s


def _build_cytoscape_elements(
    G: "nx.DiGraph",
    pos: Dict[str, Tuple[float, float]],
    edge_costs: Dict[Tuple[str, str], float],
    edge_tooltips: Dict[Tuple[str, str], str],
    cost_to_color: Any,
) -> List[Dict[str, Any]]:
    """Monta a lista de elementos (nós + arestas) no formato do Dash Cytoscape."""
    elements: List[Dict[str, Any]] = []
    for n in G.nodes():
        x, y = pos[n][0], pos[n][1]
        elements.append({
            "data": {"id": n, "label": _short_label(n), "title": G.nodes[n].get("label", n)},
            "position": {"x": x, "y": -y},
        })
    for u, v in G.edges():
        edge_id = f"{u}->{v}"
        cost = edge_costs[(u, v)]
        blocked = not math.isfinite(cost) and (cost == math.inf or cost == float("inf"))
        color = cost_to_color(cost)
        tooltip = edge_tooltips.get((u, v), "")
        data: Dict[str, Any] = {
            "id": edge_id,
            "source": u,
            "target": v,
            "color": color,
            "tooltip": tooltip,
        }
        if blocked:
            data["blocked"] = True
        elements.append({"data": data})
    return elements


def display_graph(
    G: "nx.DiGraph",
    html_path: str = "grafo_ouro_preto.html",
    min_distance: float = 400,
    iterations: int = 8,
    factor: float = 0.4,
    height: str = "550px",
    width: str = "100%",
    iframe_width: int = 900,
    iframe_height: int = 780,
    display_in_notebook: bool = True,
) -> Any:
    """
    Gera um grafo interativo com Dash Cytoscape (posições fixas, cores por custo) e
    exibe no notebook (inline) ou no navegador (external).

    O parâmetro html_path é mantido por compatibilidade; com Cytoscape o grafo
    é exibido via app Dash (não gera arquivo HTML).

    Retorna o app Dash (para reutilização ou run manual).
    """
    from src.graph import get_weight_function

    pos = {n: G.nodes[n]["pos"] for n in G.nodes()}
    pos = spread_positions(pos, min_distance=min_distance, iterations=iterations, factor=factor)

    wf = get_weight_function(G)
    edge_costs = {(u, v): wf(u, v, G.edges[u, v]) for (u, v) in G.edges()}
    valid_costs = [c for c in edge_costs.values() if math.isfinite(c)]
    min_c = min(valid_costs) if valid_costs else 0.0
    max_c = max(valid_costs) if valid_costs else 1.0

    def edge_tooltip(u: str, v: str) -> str:
        d = G.edges[u, v]
        dist = d.get("distance", 0)
        slope = d.get("slope_pct", 0)
        rough = d.get("roughness", 1)
        volcap = d.get("volume_capacity_ratio", 0)
        cost = edge_costs.get((u, v), 0)
        orig, dest = _short_label(u), _short_label(v)
        return "\n".join([
            f"Conecta: {orig} → {dest}",
            f"Distância (distance): {dist:.0f} m",
            f"Declividade (slope_pct): {slope:.1f} %",
            f"Rugosidade (roughness): {rough:.2f}",
            f"Volume/Capacidade (volume_capacity_ratio): {volcap:.2f}",
            f"Custo: {cost:.1f}",
        ])

    def cost_to_color(cost: float) -> str:
        if not math.isfinite(cost):
            return "#888888"
        t = (cost - min_c) / (max_c - min_c) if max_c > min_c else 0.0
        if not math.isfinite(t):
            return "#888888"
        t = max(0.0, min(1.0, t))
        if t <= 0.5:
            r = int(510 * t)
            g = int(100 + 80 * t)
        else:
            r = 255
            g = int(140 - 280 * (t - 0.5))
        r, g = max(0, min(255, r)), max(0, min(255, g))
        return f"#{r:02x}{g:02x}00"

    edge_tooltips_dict = {(u, v): edge_tooltip(u, v) for u, v in G.edges()}
    elements = _build_cytoscape_elements(G, pos, edge_costs, edge_tooltips_dict, cost_to_color)

    stylesheet = [
        {
            "selector": "node",
            "style": {
                "content": "data(label)",
                "background-color": "#87CEEB",
                "color": "#ffffff",
                "font-size": "12px",
                "text-valign": "center",
                "text-halign": "center",
            },
        },
        {
            "selector": "edge",
            "style": {
                "line-color": "data(color)",
                "width": 2,
                "curve-style": "bezier",
                "target-arrow-color": "data(color)",
                "target-arrow-shape": "triangle",
                "arrow-scale": 1,
            },
        },
        {
            "selector": "edge[blocked]",
            "style": {
                "label": "✕",
                "color": "#e74c3c",
                "font-size": "28px",
                "font-weight": "bold",
                "text-margin-y": 0,
                "text-background-opacity": 0,
                "text-background-color": "transparent",
                "text-background-padding": "4px",
                "line-color": "#666",
                "target-arrow-color": "#666",
                "width": 1.5,
                "line-style": "dashed",
            },
        },
    ]

    default_layout = {"name": "preset", "fit": True, "padding": 30}
    reset_layout = {"name": "preset", "fit": True, "animate": True, "animationDuration": 300, "padding": 30}

    app = Dash(__name__)
    app.layout = html.Div([
        cyto.Cytoscape(
            id="cytoscape-graph",
            elements=elements,
            layout=default_layout,
            style={
                "width": width,
                "height": height,
                "backgroundColor": "#404040",
                "overflow": "hidden",
            },
            stylesheet=stylesheet,
        ),
        html.Button(
            "Resetar vista (R ou Espaço)",
            id="reset-view-btn",
            title="Clique no grafo antes de usar R ou Espaço (para o atalho funcionar no notebook)",
            **{"data-reset-view": "true"},
            style={
                "marginTop": "8px",
                "padding": "6px 12px",
                "fontSize": "12px",
                "cursor": "pointer",
                "backgroundColor": "#555",
                "color": "#fff",
                "border": "1px solid #666",
                "borderRadius": "4px",
            },
        ),
        html.Div(
            id="cytoscape-hover-output",
            style={
                "marginTop": "8px",
                "fontSize": "12px",
                "color": "#ccc",
                "minHeight": "60px",
                "padding": "8px",
                "border": "1px solid #555",
                "borderRadius": "4px",
                "backgroundColor": "#333",
            },
        ),
        html.Div(id="cytoscape-tap-edge-output", style={"marginTop": "8px", "fontSize": "12px", "color": "#ccc"}),
        html.Script(
            children=r"""
            (function() {
                function findResetBtn(doc) {
                    doc = doc || document;
                    return doc.querySelector('[data-reset-view="true"]')
                        || doc.getElementById('reset-view-btn')
                        || doc.querySelector('[id*="reset-view-btn"]')
                        || Array.from(doc.querySelectorAll('button')).find(function(b) {
                            return b.textContent && b.textContent.indexOf('Resetar vista') !== -1;
                        });
                }
                function onKey(e) {
                    if (e.key === 'r' || e.key === 'R' || e.key === ' ') {
                        e.preventDefault();
                        var btn = findResetBtn(document);
                        if (!btn && window.parent && window.parent !== window) {
                            try {
                                var frames = window.parent.document.querySelectorAll('iframe');
                                for (var i = 0; i < frames.length; i++) {
                                    try {
                                        btn = findResetBtn(frames[i].contentDocument);
                                        if (btn) break;
                                    } catch (err) {}
                                }
                            } catch (err) {}
                        }
                        if (btn) btn.click();
                    }
                }
                function setup() {
                    document.removeEventListener('keydown', onKey);
                    document.addEventListener('keydown', onKey);
                    try {
                        if (window.parent && window.parent !== window && window.parent.document) {
                            window.parent.document.removeEventListener('keydown', onKey);
                            window.parent.document.addEventListener('keydown', onKey);
                        }
                    } catch (err) {}
                }
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', setup);
                } else {
                    setup();
                }
                setTimeout(setup, 500);
                setTimeout(setup, 2000);
            })();
            """
        ),
    ])

    @app.callback(
        Output("cytoscape-graph", "layout"),
        Input("reset-view-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_view(n_clicks):
        if not n_clicks:
            return default_layout
        return {**reset_layout, "trigger": n_clicks}

    @app.callback(
        Output("cytoscape-hover-output", "children"),
        Input("cytoscape-graph", "mouseoverNodeData"),
        Input("cytoscape-graph", "mouseoverEdgeData"),
    )
    def display_hover(node_data, edge_data):
        if node_data is not None:
            label = node_data.get("label", node_data.get("id", ""))
            title = node_data.get("title", "")
            lines = [f"Nó: {label}"]
            if title:
                lines.append(title)
            return html.Pre("\n".join(lines), style={"whiteSpace": "pre-wrap", "margin": 0})
        if edge_data is not None:
            tooltip = edge_data.get("tooltip", "")
            if tooltip:
                return html.Pre(tooltip, style={"whiteSpace": "pre-wrap", "margin": 0})
            return f"Aresta: {edge_data.get('source', '')} → {edge_data.get('target', '')}"
        return "Passe o mouse sobre um nó ou aresta para ver detalhes."

    @app.callback(
        Output("cytoscape-tap-edge-output", "children"),
        Input("cytoscape-graph", "tapEdgeData"),
    )
    def display_tap_edge_data(data):
        if data is None:
            return "Clique em uma aresta para ver detalhes (distância, custo, etc.)."
        tooltip = data.get("tooltip", "")
        if tooltip:
            return html.Pre(tooltip.replace("\n", "\n"), style={"whiteSpace": "pre-wrap", "margin": 0})
        return f"Aresta: {data.get('source', '')} → {data.get('target', '')}"

    if display_in_notebook:
        app.run(jupyter_mode="inline", jupyter_height=iframe_height, use_reloader=False)
    else:
        app.run(use_reloader=False)

    return app
