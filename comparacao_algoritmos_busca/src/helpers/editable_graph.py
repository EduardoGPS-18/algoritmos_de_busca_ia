"""
Grafo editável: barra de pesquisa para adicionar lugares dinamicamente,
recalculando vizinhanças com cache SQLite (geocode + directions).
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dash_cytoscape as cyto  # type: ignore[reportMissingImports]
import networkx as nx
from dash import Dash, Input, Output, State, ctx, html, dcc, no_update

from src.graph import (
    OURO_PRETO_REF_LAT,
    OURO_PRETO_REF_LNG,
    get_weight_function,
    _geocode_with_components,
    _get_route_distance_meters,
    _lat_lng_to_xy_meters,
)


# --- Cache SQLite (editable graph) ---
_EDITABLE_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "cache"
_EDITABLE_CACHE_DB = _EDITABLE_CACHE_DIR / "editable_graph_cache.sqlite"


def _dir_key(orig: Tuple[float, float], dest: Tuple[float, float]) -> str:
    return f"{round(orig[0], 6)},{round(orig[1], 6)},{round(dest[0], 6)},{round(dest[1], 6)}"


def get_cached_geocode(address: str) -> Optional[Tuple[float, float, str, str]]:
    """Retorna (lat, lng, bairro, estado) do cache SQLite ou None."""
    _EDITABLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(_EDITABLE_CACHE_DB)
        cur = conn.execute(
            "SELECT lat, lng, bairro, estado FROM geocode WHERE address = ?",
            (address.strip(),),
        )
        row = cur.fetchone()
        conn.close()
        if row is None:
            return None
        return (float(row[0]), float(row[1]), row[2] or "", row[3] or "")
    except Exception:
        return None


def set_cached_geocode(
    address: str,
    lat: float,
    lng: float,
    bairro: str = "",
    estado: str = "",
) -> None:
    _EDITABLE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(_EDITABLE_CACHE_DB)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS geocode (address TEXT PRIMARY KEY, lat REAL, lng REAL, bairro TEXT, estado TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO geocode (address, lat, lng, bairro, estado) VALUES (?, ?, ?, ?, ?)",
            (address.strip(), lat, lng, bairro, estado),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_cached_direction(orig: Tuple[float, float], dest: Tuple[float, float]) -> Tuple[Optional[float], bool]:
    """Retorna (distância em metros ou None se sem rota, in_cache)."""
    key = _dir_key(orig, dest)
    try:
        conn = sqlite3.connect(_EDITABLE_CACHE_DB)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS directions (key TEXT PRIMARY KEY, distance REAL)"
        )
        cur = conn.execute("SELECT distance FROM directions WHERE key = ?", (key,))
        row = cur.fetchone()
        conn.close()
        if row is None:
            return (None, False)
        return (float(row[0]) if row[0] is not None else None, True)
    except Exception:
        return (None, False)


def set_cached_direction(
    orig: Tuple[float, float],
    dest: Tuple[float, float],
    distance: Optional[float],
) -> None:
    key = _dir_key(orig, dest)
    try:
        conn = sqlite3.connect(_EDITABLE_CACHE_DB)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS directions (key TEXT PRIMARY KEY, distance REAL)"
        )
        conn.execute("INSERT OR REPLACE INTO directions (key, distance) VALUES (?, ?)", (key, distance))
        conn.commit()
        conn.close()
    except Exception:
        pass


def clear_editable_graph_cache() -> None:
    """Remove todos os dados do cache SQLite do grafo editável."""
    try:
        if _EDITABLE_CACHE_DB.exists():
            _EDITABLE_CACHE_DB.unlink()
    except Exception:
        pass


# --- Nominatim (OpenStreetMap): busca por endereço ---
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_HEADERS = {"User-Agent": "EditableGraph/1.0 (UFOP IA project; comparacao-algoritmos-busca)"}


def nominatim_search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Busca endereços/lugares via Nominatim (OpenStreetMap).
    Retorna lista de dicts com display_name, lat, lon, place_id, etc.
    """
    q = (query or "").strip()
    if len(q) < 2:
        return []
    try:
        params = urllib.parse.urlencode({"q": q, "format": "json", "limit": min(limit, 40)})
        url = f"{NOMINATIM_URL}?{params}"
        req = urllib.request.Request(url, headers=NOMINATIM_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode("utf-8")
        results = json.loads(data)
        if isinstance(results, list):
            return results
        return []
    except Exception:
        return []


# --- ViaCEP: busca por CEP ou por endereço (UF/Cidade/Logradouro) ---
VIACEP_BASE = "https://viacep.com.br/ws"


def _viacep_format_label(item: Dict[str, Any]) -> str:
    """Formata um resultado ViaCEP como string para exibição."""
    parts = [
        item.get("logradouro") or "",
        item.get("bairro") or "",
        item.get("localidade") or "",
        item.get("uf") or "",
    ]
    parts = [p.strip() for p in parts if p and str(p).strip()]
    return ", ".join(parts) if parts else (item.get("cep") or "")


def viacep_busca_cep(cep: str) -> List[Dict[str, Any]]:
    """
    Busca por CEP (8 dígitos). Retorna lista com um item ou vazia se inválido/erro.
    """
    cep_digits = re.sub(r"\D", "", str(cep or ""))
    if len(cep_digits) != 8:
        return []
    try:
        url = f"{VIACEP_BASE}/{cep_digits}/json/"
        req = urllib.request.Request(url, headers={"User-Agent": "Dash-Editable-Graph/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode("utf-8")
        obj = json.loads(data)
        if isinstance(obj, dict) and obj.get("erro"):
            return []
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return obj
        return []
    except Exception:
        return []


def viacep_busca_endereco(uf: str, cidade: str, logradouro: str) -> List[Dict[str, Any]]:
    """
    Busca por endereço (UF, Cidade, Logradouro). Mínimo 3 caracteres em cidade e logradouro.
    Retorna até 50 resultados.
    """
    uf = (uf or "").strip().upper()[:2]
    cidade = (cidade or "").strip()
    logradouro = (logradouro or "").strip()
    if len(uf) != 2 or len(cidade) < 3 or len(logradouro) < 3:
        return []
    try:
        path = f"{uf}/{urllib.parse.quote(cidade)}/{urllib.parse.quote(logradouro)}"
        url = f"{VIACEP_BASE}/{path}/json/"
        req = urllib.request.Request(url, headers={"User-Agent": "Dash-Editable-Graph/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode("utf-8")
        obj = json.loads(data)
        if isinstance(obj, dict) and obj.get("erro"):
            return []
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return obj[:50]
        return []
    except Exception:
        return []


# --- Estado inicial a partir do grafo NetworkX ---
def _graph_to_initial_state(G: nx.DiGraph) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    for n in G.nodes():
        nd = G.nodes[n]
        pos = nd.get("pos", (0.0, 0.0))
        x, y = float(pos[0]), float(pos[1])
        label = nd.get("label", n)
        nodes.append({
            "id": n,
            "label": label,
            "title": label,
            "lat": nd.get("lat", 0.0),
            "lng": nd.get("lng", 0.0),
            "x": x,
            "y": y,
        })
    edges = [{"source": u, "target": v} for u, v in G.edges()]
    return {"nodes": nodes, "edges": edges}


def _state_to_elements(
    state: Dict[str, Any],
    max_extent_px: float = 800.0,
) -> List[Dict[str, Any]]:
    nodes = state.get("nodes", [])
    edges = state.get("edges", [])
    if not nodes:
        return []
    xs = [n["x"] for n in nodes]
    ys = [n["y"] for n in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = max_x - min_x if max_x > min_x else 1.0
    dy = max_y - min_y if max_y > min_y else 1.0
    scale = max_extent_px / max(dx, dy, 1.0)
    elements: List[Dict[str, Any]] = []
    for n in nodes:
        x = (n["x"] - min_x) * scale
        y = -(n["y"] - min_y) * scale
        elements.append({
            "data": {
                "id": n["id"],
                "label": n.get("label", n["id"]),
                "title": n.get("title", n.get("label", n["id"])),
            },
            "position": {"x": x, "y": y},
        })
    for e in edges:
        sid, tid = e.get("source"), e.get("target")
        if sid and tid:
            elements.append({
                "data": {"id": f"{sid}->{tid}", "source": sid, "target": tid},
            })
    return elements


def display_editable_graph(
    G: nx.DiGraph,
    display_in_notebook: bool = True,
    iframe_height: int = 780,
    max_extent_px: float = 800.0,
) -> Any:
    """
    Exibe um grafo editável: adicionar lugares por endereço, remover por double-click
    (com confirmação). Usa cache SQLite para geocode e directions.
    """
    initial_state = _graph_to_initial_state(G)
    default_layout = {"name": "preset", "fit": True, "padding": 30}
    reset_layout = {"name": "preset", "fit": True, "animate": True, "animationDuration": 300, "padding": 30}

    app = Dash(__name__)
    app.layout = html.Div([
        dcc.Store(id="graph-store", data=initial_state),
        dcc.Store(id="selected-node", data=None),
        dcc.Store(id="last-tap-node", data=None),
        html.Div(
            [
                dcc.Input(
                    id="search-query",
                    type="text",
                    placeholder="Buscar endereço (ex: Ouro Preto, MG)",
                    style={
                        "width": "60%",
                        "padding": "10px 12px",
                        "fontSize": "14px",
                        "marginRight": "8px",
                        "border": "1px solid #666",
                        "borderRadius": "4px",
                        "backgroundColor": "#333",
                        "color": "#eee",
                    },
                ),
                html.Button(
                    "Buscar",
                    id="buscar-address-btn",
                    title="Consultar endereços no Nominatim (OpenStreetMap)",
                    style={
                        "padding": "10px 18px",
                        "fontSize": "14px",
                        "cursor": "pointer",
                        "backgroundColor": "#5a8",
                        "color": "#fff",
                        "border": "none",
                        "borderRadius": "4px",
                    },
                ),
            ],
            style={"marginBottom": "8px"},
        ),
        html.Div(
            id="address-dropdown-container",
            children=[
                dcc.Dropdown(
                    id="address-dropdown",
                    options=[],
                    placeholder="Resultados da consulta — selecione um endereço para poder adicionar",
                    clearable=True,
                    style={"width": "100%", "marginBottom": "8px"},
                ),
            ],
            style={"marginBottom": "8px"},
        ),
        html.Div(
            [
                html.Button(
                    "Adicionar",
                    id="add-place-btn",
                    disabled=True,
                    title="Habilitado apenas após selecionar um resultado da busca",
                    style={
                        "padding": "10px 20px",
                        "fontSize": "14px",
                        "cursor": "pointer",
                        "backgroundColor": "#4a7",
                        "color": "#fff",
                        "border": "none",
                        "borderRadius": "4px",
                    },
                ),
            ],
            style={"marginBottom": "12px"},
        ),
        html.Div(
            id="add-place-status",
            children="Busque um endereço acima e selecione um resultado na lista; depois clique em Adicionar.",
            style={"fontSize": "12px", "color": "#aaa", "marginBottom": "8px"},
        ),
        html.Div(
            [
                html.Strong("Debug:", style={"color": "#888"}),
                html.Span(" Input: ", style={"color": "#666"}),
                html.Span(id="debug-input", children="—", style={"fontFamily": "monospace", "fontSize": "11px", "color": "#9cf"}),
                html.Span(" | Add: ", style={"color": "#666"}),
                html.Span(id="debug-add", children="—", style={"fontFamily": "monospace", "fontSize": "11px"}),
            ],
            style={"fontSize": "11px", "color": "#666", "marginBottom": "8px", "padding": "4px", "border": "1px solid #444", "backgroundColor": "#2a2a2a"},
        ),
        html.Div(
            "Se os botões não responderem: reinicie o kernel do Jupyter e rode a célula de novo; ou use display_editable_graph(G, display_in_notebook=False) para abrir no navegador.",
            style={"fontSize": "10px", "color": "#555", "marginBottom": "8px", "fontStyle": "italic"},
        ),
        html.Div(id="delete-status", style={"fontSize": "12px", "color": "#aaa", "marginBottom": "8px"}),
        html.Button(
            "Resetar vista (R ou Espaço)",
            id="reset-view-btn",
            title="Recentralizar o grafo na tela",
            **{"data-reset-view": "true"},
            style={
                "marginBottom": "8px",
                "marginRight": "8px",
                "padding": "6px 12px",
                "fontSize": "12px",
                "cursor": "pointer",
                "backgroundColor": "#555",
                "color": "#fff",
                "border": "1px solid #666",
                "borderRadius": "4px",
            },
        ),
        dcc.Input(id="hold-delete-node-id", value="", type="text", style={"display": "none"}, debounce=True),
        cyto.Cytoscape(
            id="cytoscape-editable",
            elements=[],
            layout=default_layout,
            style={"width": "100%", "height": "550px", "backgroundColor": "#404040", "overflow": "hidden"},
            stylesheet=[
                {"selector": "node", "style": {"content": "data(label)", "background-color": "#87CEEB", "color": "#333", "font-size": "11px"}},
                {"selector": "edge", "style": {"width": 2, "line-color": "#888", "curve-style": "bezier", "target-arrow-color": "#888", "target-arrow-shape": "triangle"}},
            ],
        ),
        html.Div(
            id="editable-hover-output",
            style={"marginTop": "8px", "fontSize": "12px", "color": "#ccc", "minHeight": "50px", "padding": "8px", "border": "1px solid #555", "borderRadius": "4px", "backgroundColor": "#333"},
        ),
        html.Script(
            children=r"""
            (function() {
                function findResetBtn(doc) {
                    doc = doc || document;
                    return doc.querySelector('[data-reset-view="true"]') || doc.getElementById('reset-view-btn')
                        || Array.from(doc.querySelectorAll('button')).find(function(b) { return b.textContent && b.textContent.indexOf('Resetar vista') !== -1; });
                }
                function onKey(e) {
                    if (e.key === 'r' || e.key === 'R' || e.key === ' ') { e.preventDefault(); var btn = findResetBtn(document);
                        if (!btn && window.parent && window.parent !== window) {
                            try { var frames = window.parent.document.querySelectorAll('iframe');
                                for (var i = 0; i < frames.length; i++) { try { btn = findResetBtn(frames[i].contentDocument); if (btn) break; } catch (err) {} }
                            } catch (err) {}
                        }
                        if (btn) btn.click();
                    }
                }
                if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', function() { document.addEventListener('keydown', onKey); });
                else document.addEventListener('keydown', onKey);
                setTimeout(function() { document.addEventListener('keydown', onKey); }, 500);
            })();
            """
        ),
    ])

    @app.callback(
        Output("address-dropdown", "options"),
        [Input("buscar-address-btn", "n_clicks"), Input("search-query", "n_submit")],
        State("search-query", "value"),
        prevent_initial_call=True,
    )
    def buscar_nominatim(_n_clicks, _n_submit, query):
        q = (query or "").strip()
        if len(q) < 2:
            return []
        results = nominatim_search(q, limit=10)
        options = []
        for r in results:
            dn = r.get("display_name") or ""
            lat = r.get("lat")
            lon = r.get("lon")
            if dn and lat is not None and lon is not None:
                options.append({
                    "label": dn,
                    "value": json.dumps({"display_name": dn, "lat": lat, "lon": lon}),
                })
        return options

    @app.callback(
        Output("add-place-btn", "disabled"),
        Input("address-dropdown", "value"),
    )
    def toggle_add_button(selected_address):
        return not bool(selected_address)

    @app.callback(
        [
            Output("graph-store", "data"),
            Output("add-place-status", "children"),
            Output("address-dropdown", "value"),
            Output("debug-add", "children"),
            Output("last-tap-node", "data"),
        ],
        [
            Input("add-place-btn", "n_clicks"),
            Input("cytoscape-editable", "tapNodeData"),
        ],
        State("graph-store", "data"),
        State("address-dropdown", "value"),
        State("last-tap-node", "data"),
        prevent_initial_call=True,
    )
    def add_or_remove_place(n_clicks, tap_node_data, current_state, selected_address, last_tap):
        triggered = ctx.triggered_id if ctx.triggered_id else None
        state = current_state if current_state else initial_state
        state = {"nodes": state.get("nodes", initial_state["nodes"]), "edges": state.get("edges", initial_state["edges"])}
        nodes = list(state["nodes"])
        edges = list(state["edges"])

        # Double-click no nó (dois cliques no mesmo nó em ~500 ms) → remove o nó
        if triggered == "cytoscape-editable" and tap_node_data:
            node_id = tap_node_data.get("id")
            if not node_id:
                return (no_update, no_update, no_update, no_update, no_update)
            now = time.time()
            if last_tap and isinstance(last_tap, dict) and last_tap.get("id") == node_id and (now - last_tap.get("time", 0)) < 0.5:
                nodes = [n for n in nodes if n.get("id") != node_id]
                edges = [e for e in edges if e.get("source") != node_id and e.get("target") != node_id]
                new_state = {"nodes": nodes, "edges": edges}
                return (new_state, "Nó removido.", no_update, "removido", None)
            return (no_update, no_update, no_update, no_update, {"id": node_id, "time": now})

        # Botão Adicionar → só adiciona se houver endereço selecionado na busca
        if triggered != "add-place-btn" or not selected_address:
            return (no_update, no_update, no_update, no_update, no_update)
        try:
            data = json.loads(selected_address)
        except (TypeError, ValueError):
            return (no_update, no_update, no_update, no_update, no_update)
        display_name = (data.get("display_name") or "").strip()
        lat_val = data.get("lat")
        lon_val = data.get("lon")
        if not display_name or lat_val is None or lon_val is None:
            return (no_update, no_update, no_update, no_update, no_update)
        try:
            lat_f = float(lat_val)
            lon_f = float(lon_val)
        except (TypeError, ValueError):
            return (no_update, no_update, no_update, no_update, no_update)
        node_id = f"{display_name}_{lat_f}_{lon_f}"
        x, y = _lat_lng_to_xy_meters(lat_f, lon_f, OURO_PRETO_REF_LAT, OURO_PRETO_REF_LNG)
        nodes.append({
            "id": node_id,
            "label": display_name,
            "title": display_name,
            "lat": lat_f,
            "lng": lon_f,
            "x": x,
            "y": y,
        })
        new_state = {"nodes": nodes, "edges": edges}
        msg = "Adicionado: %s (%d nós)" % (display_name, len(nodes))
        return (new_state, msg, None, msg, no_update)

    @app.callback(
        Output("cytoscape-editable", "elements"),
        Input("graph-store", "data"),
    )
    def store_to_elements(data):
        return _state_to_elements(data, max_extent_px=max_extent_px)

    @app.callback(
        Output("cytoscape-editable", "layout"),
        Input("reset-view-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset_view(n_clicks):
        if not n_clicks:
            return default_layout
        return {**reset_layout, "trigger": n_clicks}

    @app.callback(
        Output("editable-hover-output", "children"),
        Input("cytoscape-editable", "mouseoverNodeData"),
        Input("cytoscape-editable", "mouseoverEdgeData"),
    )
    def editable_hover(node_data, edge_data):
        if node_data:
            return html.Pre(f"Nó: {node_data.get('label', '')}\n{node_data.get('title', '')}", style={"whiteSpace": "pre-wrap", "margin": 0})
        if edge_data:
            t = edge_data.get("tooltip", "")
            return html.Pre(t, style={"whiteSpace": "pre-wrap", "margin": 0}) if t else f"Aresta: {edge_data.get('source', '')} → {edge_data.get('target', '')}"
        return "Passe o mouse sobre um nó ou aresta."

    @app.callback(
        Output("debug-input", "children"),
        Input("search-query", "value"),
    )
    def debug_input(value):
        return value if value is not None and value != "" else "—"

    @app.callback(
        Output("selected-node", "data"),
        Input("cytoscape-editable", "tapNodeData"),
    )
    def store_selected_node(node_data):
        return node_data.get("id") if node_data else None

    @app.callback(
        Output("delete-status", "children"),
        Input("cytoscape-editable", "tapNodeData"),
        Input("cytoscape-editable", "tapEdgeData"),
    )
    def editable_tap(node_data, edge_data):
        if node_data:
            label = node_data.get("label", node_data.get("id", ""))
            return f"Nó '{label}'. Clique duas vezes no nó para removê-lo."
        if edge_data:
            return f"Aresta: {edge_data.get('source', '')} → {edge_data.get('target', '')}"
        return "Clique em um nó ou aresta."

    if display_in_notebook:
        app.run(jupyter_mode="inline", jupyter_height=iframe_height + 150, use_reloader=False, port=0)
    else:
        app.run(use_reloader=False, port=8051)
    return app
