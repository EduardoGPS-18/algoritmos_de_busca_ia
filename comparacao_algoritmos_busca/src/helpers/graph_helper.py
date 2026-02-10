"""
Helpers para exibição e manipulação de grafos (Dash Cytoscape, posições, etc.).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

import dash_cytoscape as cyto
from dash import Dash, Input, Output, html, dcc

if TYPE_CHECKING:
    import networkx as nx


def _escape_cytoscape_id(node_id: str) -> str:
    """Escapa id de nó para uso em seletor Cytoscape (ex.: pontos viram \\.)."""
    return node_id.replace("\\", "\\\\").replace(".", "\\.").replace(":", "\\:")


# Raio mínimo em px ao redor de cada nó no spawn (distância centro a centro >= 2 * NODE_SPAWN_RADIUS)
NODE_SPAWN_RADIUS_PX: float = 30.0

# Canvas em pixels para layout (posições do grafo são escaladas para este tamanho antes do spread)
_LAYOUT_CANVAS_WIDTH: float = 800.0
_LAYOUT_CANVAS_HEIGHT: float = 600.0
_LAYOUT_PADDING: float = 40.0


def _scale_positions_to_canvas(
    pos_dict: Dict[str, Tuple[float, float]],
    width: float = _LAYOUT_CANVAS_WIDTH,
    height: float = _LAYOUT_CANVAS_HEIGHT,
    padding: float = _LAYOUT_PADDING,
) -> Dict[str, Tuple[float, float]]:
    """Escala posições (unidades do grafo, ex.: metros) para um canvas em pixels.
    Assim o spread_positions com min_distance em px faz efeito na tela.
    """
    if not pos_dict:
        return {}
    xs = [p[0] for p in pos_dict.values()]
    ys = [p[1] for p in pos_dict.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max_x - min_x
    range_y = max_y - min_y
    if range_x < 1e-6:
        range_x = 1.0
    if range_y < 1e-6:
        range_y = 1.0
    inner_w = width - 2 * padding
    inner_h = height - 2 * padding
    out = {}
    for n, (x, y) in pos_dict.items():
        nx = padding + (x - min_x) / range_x * inner_w
        ny = padding + (y - min_y) / range_y * inner_h
        out[n] = (float(nx), float(ny))
    return out


def spread_positions(
    pos_dict: Dict[str, Tuple[float, float]],
    min_distance: Optional[float] = None,
    iterations: int = 8,
    factor: float = 0.4,
) -> Dict[str, Tuple[float, float]]:
    """Afasta nós muito próximos mantendo o formato geral (evita sobreposição).
    min_distance: distância mínima centro a centro (px). Se None, usa 2 * NODE_SPAWN_RADIUS_PX.
    """
    if min_distance is None:
        min_distance = 2.0 * NODE_SPAWN_RADIUS_PX
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


# Abreviações de cidade no label (nome completo -> iniciais)
_CITY_ABBREV: Dict[str, str] = {
    "Cachoeira do Campo": "CdC",
    "Ouro Preto": "OP",
    "Mariana": "M",
}


def _abbreviate_city_in_label(text: str) -> str:
    """Substitui o nome completo da cidade pelas iniciais no texto do label."""
    out = str(text)
    for full_name, initials in _CITY_ABBREV.items():
        out = out.replace(full_name, initials)
    return out


def _build_cytoscape_elements(
    G: "nx.DiGraph",
    pos: Dict[str, Tuple[float, float]],
    edge_costs: Dict[Tuple[str, str], float],
    edge_tooltips: Dict[Tuple[str, str], str],
    cost_to_color: Any,
    path_edges: Optional[Set[Tuple[str, str]]] = None,
    start_node: Optional[str] = None,
    goal_node: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Monta a lista de elementos (nós + arestas) no formato do Dash Cytoscape.
    Nós em regiões com chuva local (rain_multiplier_by_region > 1) ganham ícone 💧.
    Arestas congestionadas (congestion_factor_by_edge > 1) ganham ícone 🚗 na via (desenhada à frente).
    Se path_edges for passado, arestas do caminho recebem in_path=True (opacidade 100%); demais in_path=False (60%).
    start_node e goal_node recebem classes para destaque (verde escuro e vermelho).
    Arestas com segunda cor (efeito listrado): use G.edges[u,v]['stripe_color'] = '#hex' ou
    G.graph['striped_edges'] = {(u,v): '#hex', ...}; a linha principal fica sólida e uma camada tracejada
    com a segunda cor é desenhada por baixo.
    """
    from src.graph_operations import (
        KEY_CONGESTION_FACTOR_BY_EDGE,
        KEY_RAIN_MULTIPLIER_BY_REGION,
        GraphOperations,
    )
    rain_by_region = G.graph.get(KEY_RAIN_MULTIPLIER_BY_REGION, {})
    congestion_by_edge = G.graph.get(KEY_CONGESTION_FACTOR_BY_EDGE, {})
    elements: List[Dict[str, Any]] = []
    for n in G.nodes():
        x, y = pos[n][0], pos[n][1]
        display_label = G.nodes[n].get("label") or n
        if not (display_label and str(display_label).strip()):
            display_label = n.replace("_", " ").replace("-", " ").title()
        base_label = _abbreviate_city_in_label(str(display_label).strip())
        region = GraphOperations.get_region_from_graph(G, n)
        has_rain = bool(rain_by_region and rain_by_region.get(region, 1.0) > 1.0)
        if has_rain:
            base_label += " 💧"
        node_elem: Dict[str, Any] = {
            "data": {"id": n, "label": base_label, "title": display_label},
            "position": {"x": x, "y": -y},
        }
        classes = []
        if n == start_node:
            classes.append("start")
        if n == goal_node:
            classes.append("goal")
        if classes:
            node_elem["classes"] = " ".join(classes)
        elements.append(node_elem)
    # Opcional: arestas com segunda cor (efeito listrado). Chave no grafo ou em G.edges[u,v]["stripe_color"]
    striped_edges = G.graph.get("striped_edges", {})
    for u, v in G.edges():
        edge_id = f"{u}->{v}"
        cost = edge_costs[(u, v)]
        blocked = not math.isfinite(cost) and (cost == math.inf or cost == float("inf"))
        color = cost_to_color(cost)
        tooltip = edge_tooltips.get((u, v), "")
        congested = congestion_by_edge.get((u, v), 1.0) > 1.0
        stripe_color = G.edges[u, v].get("stripe_color") or striped_edges.get((u, v))
        data: Dict[str, Any] = {
            "id": edge_id,
            "source": u,
            "target": v,
            "color": color,
            "tooltip": tooltip,
        }
        if blocked:
            data["blocked"] = True
        if congested:
            data["congested"] = True
        if path_edges is not None:
            data["in_path"] = (u, v) in path_edges
            data["edge_opacity"] = 1.0 if (u, v) in path_edges else 0.2
        else:
            data["edge_opacity"] = 1.0
        elements.append({"data": data})
        # Segunda cor (listrado): desenhar uma aresta tracejada por baixo com a cor alternativa
        if stripe_color and not blocked:
            stripe_id = f"{u}->{v}_stripe"
            stripe_opacity = data.get("edge_opacity", 1.0)
            elements.append({
                "data": {
                    "id": stripe_id,
                    "source": u,
                    "target": v,
                    "color": stripe_color,
                    "edge_opacity": stripe_opacity,
                    "stripe_layer": True,
                }
            })
    return elements


def display_graph(
    G: "nx.DiGraph",
    html_path: str = "grafo_ouro_preto.html",
    min_distance: Optional[float] = 100,
    iterations: int = 20,
    factor: float = 0.7,
    height: str = "550px",
    width: str = "100%",
    iframe_width: int = 900,
    iframe_height: int = 780,
    display_in_notebook: bool = True,
    path: Optional[List[str]] = None,
    start: Optional[str] = None,
    goal: Optional[str] = None,
) -> Any:
    """
    Gera um grafo interativo com Dash Cytoscape (posições fixas, cores por custo) e
    exibe no notebook (inline) ou no navegador (external).

    O parâmetro html_path é mantido por compatibilidade; com Cytoscape o grafo
    é exibido via app Dash (não gera arquivo HTML).

    path: lista de nós do caminho percorrido (ex.: retorno de dijkstra). Se passado,
    arestas fora do caminho ficam com opacidade 60% e arestas do caminho com 100%,
    sem remover nenhuma informação do grafo.

    start: id do nó de partida/saída (verde escuro #14532d). goal: id do nó de destino (vermelho).

    min_distance: distância mínima centro a centro entre nós (px). Se None, usa 2 * NODE_SPAWN_RADIUS_PX (60px),
    garantindo raio de 30px ao redor de cada nó no spawn.

    Retorna o app Dash (para reutilização ou run manual).
    """
    from src.graph_operations import GraphOperations

    pos = {n: G.nodes[n]["pos"] for n in G.nodes()}
    pos = _scale_positions_to_canvas(pos)
    pos = spread_positions(
        pos,
        min_distance=min_distance if min_distance is not None else 2.0 * NODE_SPAWN_RADIUS_PX,
        iterations=iterations,
        factor=factor,
    )

    wf = GraphOperations.get_weight_function(G)
    edge_costs = {(u, v): wf(u, v, G.edges[u, v]) for (u, v) in G.edges()}
    valid_costs = [c for c in edge_costs.values() if math.isfinite(c)]
    min_c = min(valid_costs) if valid_costs else 0.0
    max_c = max(valid_costs) if valid_costs else 1.0

    def edge_tooltip(u: str, v: str) -> str:
        d = G.edges[u, v]
        dist = d.get("distance", 0)
        slope = d.get("slope_pct", 0)
        cost = edge_costs.get((u, v), 0)
        orig, dest = _short_label(u), _short_label(v)
        return "\n".join([
            f"Conecta: {orig} → {dest}",
            f"Distância (distance): {dist:.0f} m",
            f"Declividade (slope_pct): {slope:.1f} %",
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
    path_edges: Optional[Set[Tuple[str, str]]] = None
    if path and len(path) >= 2:
        path_edges = set((path[i], path[i + 1]) for i in range(len(path) - 1))
    elements = _build_cytoscape_elements(
        G, pos, edge_costs, edge_tooltips_dict, cost_to_color,
        path_edges=path_edges, start_node=start, goal_node=goal,
    )

    edge_base_style: Dict[str, Any] = {
        "line-color": "data(color)",
        "width": 1,
        "curve-style": "bezier",
        "target-arrow-color": "data(color)",
        "target-arrow-shape": "triangle",
        "arrow-scale": 1,
        "opacity": "data(edge_opacity)",
    }

    stylesheet: List[Dict[str, Any]] = [
        {
            "selector": "node",
            "style": {
                "content": "data(label)",
                "background-color": "#87CEEB",
                "color": "#ffffff",
                "font-size": "12px",
                "text-valign": "bottom",
                "text-halign": "center",
                "text-margin-y": 10,
                "text-wrap": "wrap",
                "text-max-width": "120px",
            },
        },
    ]
    # Nó de origem (saída): verde escuro #14532d; nó de destino: vermelho (por classe e por id para garantir)
    if start is not None:
        stylesheet.append({
            "selector": "#" + _escape_cytoscape_id(start),
            "style": {"background-color": "#14532d", "color": "#ffffff"},
        })
        stylesheet.append({
            "selector": "node.start",
            "style": {"background-color": "#14532d", "color": "#ffffff"},
        })
    if goal is not None:
        stylesheet.append({
            "selector": "#" + _escape_cytoscape_id(goal),
            "style": {"background-color": "#ef4444", "color": "#ffffff"},
        })
        stylesheet.append({
            "selector": "node.goal",
            "style": {"background-color": "#ef4444", "color": "#ffffff"},
        })
    stylesheet.extend([
        {
            "selector": "edge",
            "style": edge_base_style,
        },
        # Arestas do caminho real (path): mais espessas que as demais (base = 2)
        {
            "selector": "edge[in_path]",
            "style": {"width": 6},
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
        {
            "selector": "edge[congested]",
            "style": {
                "label": "🚗",
                "color": "#f1c40f",
                "font-size": "22px",
                "text-margin-y": -10,
                "text-background-opacity": 0,
                "text-background-color": "transparent",
                "text-background-padding": "2px",
                "z-index": 10,
            },
        },
        # Aresta “listrada”: segunda camada tracejada (cor em data(color)), atrás da linha principal
        {
            "selector": "edge[stripe_layer]",
            "style": {
                "line-color": "data(color)",
                "target-arrow-color": "data(color)",
                "target-arrow-shape": "none",
                "width": 3,
                "line-style": "dashed",
                "opacity": "data(edge_opacity)",
                "z-index": -1,
            },
        },
    ])

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
