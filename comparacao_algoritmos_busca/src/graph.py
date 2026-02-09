"""
Modelagem do grafo para roteamento em Ouro Preto usando NetworkX.

Vértices: interseções e pontos de interesse (nós com 'pos' e 'label').
Arestas: segmentos de vias direcionadas (atributos: distance, slope_pct, roughness, volume_capacity_ratio).
Função de custo: distância + penalização por declividade + rugosidade + congestionamento.

Cenários (evento, clima) são armazenados em G.graph e usados pela função de peso.
"""

from __future__ import annotations
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import requests

# Diretório de cache (raiz do projeto = pasta comparacao_algoritmos_busca)
_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
_CACHE_OURO_PRETO = _CACHE_DIR / "grafo_ouro_preto.gpickle"
_CACHE_OP_GRAPH = _CACHE_DIR / "grafo_op_osm.gpickle"
_CACHE_OSM_WAY_NEIGHBORS = _CACHE_DIR / "osm_way_neighbors_cache.pickle"
_CACHE_REGIONAL = _CACHE_DIR / "grafo_regional.gpickle"
_CACHE_GOOGLE_MAPS = _CACHE_DIR / "google_maps_api_cache.pickle"


# Chaves em G.graph para parâmetros do cenário
KEY_RAIN_MULTIPLIER = "rain_multiplier"
KEY_RAIN_MULTIPLIER_BY_REGION = "rain_multiplier_by_region"
KEY_SLOPE_PENALTY_FACTOR = "slope_penalty_factor"
KEY_CONGESTION_FACTOR = "congestion_factor"
KEY_CONGESTION_FACTOR_BY_REGION = "congestion_factor_by_region"
KEY_CONGESTION_FACTOR_BY_EDGE = "congestion_factor_by_edge"
KEY_EDGE_OVERRIDE = "edge_override"
KEY_EDGE_COST_MULTIPLIER = "edge_cost_multiplier"

# Prefixos de nó para região (grafo regional: op_, mariana_, cachoeira_)
REGION_PREFIXES = ("op_", "mariana_", "cachoeira_")

# Mapeamento OSM surface -> rugosidade numérica (fácil de alterar).
# Ref: https://wiki.openstreetmap.org/wiki/Key:surface
# Valor 1.0 = asfalto liso; maior = mais rugoso (pé de moleque ~1.57, terra ~2.0).
OSM_SURFACE_TO_ROUGHNESS: Dict[str, float] = {
    "asphalt": 1.0,
    "paved": 1.0,
    "concrete": 1.05,
    "concrete:plates": 1.1,
    "concrete:lanes": 1.05,
    "paving_stones": 1.2,
    "sett": 1.3,
    "cobblestone": 1.5,
    "cobblestone:flattened": 1.35,
    "unpaved": 1.6,
    "gravel": 1.4,
    "fine_gravel": 1.35,
    "pebblestone": 1.45,
    "compacted": 1.25,
    "dirt": 1.7,
    "earth": 1.7,
    "grass": 1.8,
    "grass_paver": 1.5,
    "sand": 1.9,
    "mud": 2.0,
    "wood": 1.3,
    "woodchips": 1.6,
    "metal": 1.1,
    # Calçamento pé de moleque (comum em Ouro Preto)
    "stone": 1.5,
    "compacted;gravel": 1.35,
}


def surface_to_roughness(surface: Optional[str]) -> float:
    """
    Mapeia o valor da tag OSM 'surface' para um número de rugosidade.
    Alterar o mapeamento: edite o dicionário OSM_SURFACE_TO_ROUGHNESS no topo do módulo.
    """
    if surface is None or not isinstance(surface, str):
        return 1.0
    return OSM_SURFACE_TO_ROUGHNESS.get(
        surface.strip().lower(),
        1.0,
    )


def _get_region(node_id: str) -> str:
    """
    Retorna a região do nó a partir do id (para custo por região).
    op_* -> "op", mariana_* -> "mariana", cachoeira_* -> "cachoeira", senão -> "default".
    """
    if not isinstance(node_id, str):
        return "default"
    for prefix in REGION_PREFIXES:
        if node_id.startswith(prefix):
            return prefix.rstrip("_")
    return "default"


def _compute_edge_cost(
    d: Dict[str, Any],
    rain_multiplier: float = 1.0,
    slope_penalty_factor: float = 1.0,
    congestion_factor: float = 1.0,
) -> float:
    """
    Custo do arco conforme base_trabalho_ia.pdf.
    d deve conter: distance, slope_pct, roughness, volume_capacity_ratio.
    """
    dist = d.get("distance", 0.0)
    slope_pct = d.get("slope_pct", 0.0)
    roughness = d.get("roughness", 1.0)
    vol_cap = d.get("volume_capacity_ratio", 0.0)

    if slope_pct > 15:
        slope_penalty = (slope_pct - 15) / 100 * slope_penalty_factor * rain_multiplier
    elif slope_pct > 0:
        slope_penalty = (slope_pct / 100) * slope_penalty_factor * rain_multiplier
    else:
        slope_penalty = 0.0

    congestion_penalty = vol_cap * congestion_factor
    cost = dist * (1 + slope_penalty) * roughness * (1 + congestion_penalty)
    return max(cost, 1e-6)


def _get_region_from_graph(G: nx.DiGraph, node_id: Any) -> str:
    """Região do nó: atributo G.nodes[n]['region'] ou inferida por prefixo (op_, mariana_, cachoeira_)."""
    return G.nodes.get(node_id, {}).get("region", _get_region(node_id))


def get_weight_function(G: nx.DiGraph) -> Callable[[Any, Any, Dict], float]:
    """
    Retorna uma função (u, v, d) -> peso para uso em nx.dijkstra_path, nx.astar_path, etc.
    Usa G.graph para rain_multiplier, edge_override, etc.
    Chuva: por região do nó de origem u (rain_multiplier_by_region).
    Congestionamento: por aresta (congestion_factor_by_edge); se não definido para (u,v), usa global.
    """
    override = G.graph.get(KEY_EDGE_OVERRIDE, {})
    edge_multiplier = G.graph.get(KEY_EDGE_COST_MULTIPLIER, {})
    rain_global = G.graph.get(KEY_RAIN_MULTIPLIER, 1.0)
    rain_by_region = G.graph.get(KEY_RAIN_MULTIPLIER_BY_REGION, {})
    slope_factor = G.graph.get(KEY_SLOPE_PENALTY_FACTOR, 1.0)
    congestion_global = G.graph.get(KEY_CONGESTION_FACTOR, 1.0)
    congestion_by_edge = G.graph.get(KEY_CONGESTION_FACTOR_BY_EDGE, {})
    congestion_by_region = G.graph.get(KEY_CONGESTION_FACTOR_BY_REGION, {})

    def weight(u: Any, v: Any, d: Dict) -> float:
        key = (u, v)
        if key in override:
            return override[key]
        region = _get_region_from_graph(G, u)
        rain = rain_by_region.get(region, rain_global)
        # Congestionamento por via (aresta) tem prioridade; senão, fallback por região e global
        congestion = congestion_by_edge.get(key, congestion_by_region.get(region, congestion_global))
        base = _compute_edge_cost(d, rain, slope_factor, congestion)
        return base * edge_multiplier.get(key, 1.0)

    return weight


def get_edge_cost(G: nx.DiGraph, u: Any, v: Any) -> float:
    """Custo atual da aresta (u, v) com base nos atributos e no cenário em G.graph."""
    if not G.has_edge(u, v):
        return float("inf")
    wf = get_weight_function(G)
    return wf(u, v, G.edges[u, v])


def validate_path_nodes(G: nx.DiGraph, start: Any, goal: Any) -> None:
    """
    Levanta NetworkXError se start ou goal não existirem no grafo.
    Mensagem sugere verificar list(G.nodes()) e diferencia grafo exemplo vs regional.
    """
    missing = [n for n in (start, goal) if n not in G]
    if not missing:
        return
    nodes_list = list(G.nodes())[:15]
    hint = (
        "Grafo exemplo (build_ouro_preto_example): use START='praca_tiradentes', GOAL='campus'. "
        "Grafo regional (build_ouro_preto_mariana_cachoeira): use START='op_tiradentes', GOAL='op_campus' ou "
        "START='op_tiradentes', GOAL='mariana_centro'. Nós neste grafo: "
    ) + str(nodes_list) + ("..." if len(G) > 15 else "")
    raise nx.NetworkXError(f"Nó(s) {missing} não existem no grafo. {hint}")


def get_straight_line_distance(G: nx.DiGraph, u: Any, v: Any) -> float:
    """Distância em linha reta entre dois nós (atributo 'pos' = (x, y)). Para heurística A*."""
    if u not in G.nodes or v not in G.nodes:
        return float("inf")
    pos_u = G.nodes[u].get("pos", (0, 0))
    pos_v = G.nodes[v].get("pos", (0, 0))
    return math.sqrt((pos_u[0] - pos_v[0]) ** 2 + (pos_u[1] - pos_v[1]) ** 2)


def path_cost(G: nx.DiGraph, path: list) -> float:
    """Custo total de um caminho (lista de nós)."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    wf = get_weight_function(G)
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if not G.has_edge(u, v):
            return float("inf")
        total += wf(u, v, G.edges[u, v])
    return total


# --- Google Maps: Geocoding e Directions ---

# Centro aproximado de Ouro Preto (ref para converter lat/lng em metros)
OURO_PRETO_REF_LAT = -20.3855
OURO_PRETO_REF_LNG = -43.5034


def _lat_lng_to_xy_meters(lat: float, lng: float, ref_lat: float, ref_lng: float) -> Tuple[float, float]:
    """Converte (lat, lng) em (x, y) em metros relativos a ref (para heurística A*)."""
    # Aproximação: 1° lat ≈ 110540 m, 1° lng ≈ 111320*cos(lat) m
    y = (lat - ref_lat) * 110540.0
    x = (lng - ref_lng) * 111320.0 * math.cos(math.radians(ref_lat))
    return (x, y)


def _straight_line_distance_meters(
    lat1: float, lng1: float, lat2: float, lng2: float,
    ref_lat: float, ref_lng: float,
) -> float:
    """Distância em linha reta entre dois pontos (lat,lng) em metros, usando a mesma projeção que _lat_lng_to_xy_meters."""
    x1, y1 = _lat_lng_to_xy_meters(lat1, lng1, ref_lat, ref_lng)
    x2, y2 = _lat_lng_to_xy_meters(lat2, lng2, ref_lat, ref_lng)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _geocode(api_key: str, address: str) -> Optional[Tuple[float, float]]:
    """Obtém (lat, lng) de um endereço via Google Geocoding API."""
    result = _geocode_with_components(api_key, address)
    return (result[0], result[1]) if result else None


def _geocode_with_components(
    api_key: str, address: str
) -> Optional[Tuple[float, float, str, str]]:
    """
    Obtém (lat, lng, bairro, estado) via Google Geocoding API.
    bairro = long_name de sublocality_level_1/sublocality/neighborhood (como vem do Google).
    estado = short_name de administrative_area_level_1 (2 caracteres, ex: MG).
    Retorna None se falhar; bairro/estado vazios se não encontrados.
    """
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": api_key}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK" or not data.get("results"):
            return None
        res = data["results"][0]
        loc = res["geometry"]["location"]
        lat, lng = loc["lat"], loc["lng"]
        bairro = ""
        estado = ""
        for comp in res.get("address_components", []):
            types = comp.get("types", [])
            if "administrative_area_level_1" in types:
                estado = (comp.get("short_name") or comp.get("long_name", ""))[:2]
            if not bairro and any(t in types for t in ("sublocality_level_1", "sublocality", "neighborhood")):
                bairro = comp.get("long_name", "")
        if not bairro:
            for comp in res.get("address_components", []):
                if "locality" in comp.get("types", []):
                    bairro = comp.get("long_name", "")
                    break
        if not bairro:
            bairro = res.get("formatted_address", address).split(",")[0].strip()
        return (lat, lng, bairro, estado)
    except Exception:
        return None


def _get_elevations_batch(
    api_key: str,
    locations: List[Tuple[float, float]],
) -> List[Optional[float]]:
    """
    Obtém elevação (m) para uma lista de pontos via Google Elevation API.
    Retorna lista na mesma ordem de locations; None onde falhou.
    Máximo 512 pontos por request (limite da API).
    Ref: https://developers.google.com/maps/documentation/elevation/overview
    """
    if not locations:
        return []
    # locations = "lat1,lng1|lat2,lng2|..."
    loc_str = "|".join(f"{lat},{lng}" for lat, lng in locations)
    url = "https://maps.googleapis.com/maps/api/elevation/json"
    params = {"locations": loc_str, "key": api_key}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK" or "results" not in data:
            return [None] * len(locations)
        results = data["results"]
        out = []
        for i, res in enumerate(results):
            if i < len(locations) and "elevation" in res:
                out.append(float(res["elevation"]))
            else:
                out.append(None)
        # Garantir mesmo tamanho
        while len(out) < len(locations):
            out.append(None)
        return out[: len(locations)]
    except Exception:
        return [None] * len(locations)


def _slope_pct_from_elevation(
    elev_origin_m: float,
    elev_dest_m: float,
    distance_m: float,
) -> float:
    """
    Declividade em % entre dois pontos: (elev_dest - elev_origin) / distance * 100.
    Positivo = subida (origem → destino), negativo = descida.
    """
    if distance_m is None or distance_m <= 0:
        return 0.0
    return (elev_dest_m - elev_origin_m) / distance_m * 100.0


# Overpass API (OpenStreetMap) para rugosidade e conectividade de ruas.
# Ref: https://wiki.openstreetmap.org/wiki/Overpass_API
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"

# Bbox Ouro Preto — sede (south, west, north, east) em graus, para consultas Overpass.
# Centro ref: (-20.3855, -43.5034). Raio ~0.04° ≈ 4,4 km (só a área urbana da sede).
# Valores explícitos: south=-20.4255, west=-43.5434, north=-20.3455, east=-43.4634
_OURO_PRETO_RADIUS_DEG = 0.04
OURO_PRETO_BBOX = (
    OURO_PRETO_REF_LAT - _OURO_PRETO_RADIUS_DEG,
    OURO_PRETO_REF_LNG - _OURO_PRETO_RADIUS_DEG,
    OURO_PRETO_REF_LAT + _OURO_PRETO_RADIUS_DEG,
    OURO_PRETO_REF_LNG + _OURO_PRETO_RADIUS_DEG,
)


def _get_street_name_from_address(address: str) -> str:
    """Extrai o nome da rua do endereço (parte antes da primeira vírgula). Ex: 'Rua São José, Ouro Preto, MG' -> 'Rua São José'."""
    return address.split(",")[0].strip() if address else ""


def _query_overpass_connected_street_names(
    street_name: str,
    south: float,
    west: float,
    north: float,
    east: float,
    timeout: int = 25,
    url: str = OVERPASS_API_URL,
) -> Set[str]:
    """
    Consulta Overpass: ruas (ways highway) que compartilham nós com a rua de nome street_name na bbox.
    Retorna conjunto de nomes de ruas conectadas (inclui a própria rua; o chamador pode remover).
    Baseado em: way["name"="X"]; node(w); way(bn)[highway]; -> nomes dos ways.
    """
    if not street_name or not street_name.strip():
        return set()
    # Aspas no nome quebram a query; substituir por espaço para segurança
    name_safe = street_name.replace('"', " ").strip()
    if not name_safe:
        return set()
    query = (
        f'[out:json][timeout:{timeout}];'
        f'way["name"="{name_safe}"]({south},{west},{north},{east})->.rua_origem;'
        "node(w.rua_origem)->.nos_da_rua;"
        'way(bn.nos_da_rua)["highway"];'
        "out body;"
    )
    try:
        r = requests.post(url, data={"data": query}, timeout=timeout + 5)
        r.raise_for_status()
        data = r.json()
        names: Set[str] = set()
        for el in data.get("elements", []):
            if el.get("type") == "way":
                n = (el.get("tags") or {}).get("name")
                if n and isinstance(n, str):
                    names.add(n.strip())
        return names
    except Exception:
        return set()


def get_osm_connected_pairs(
    places: List[Dict[str, Any]],
    bbox: Tuple[float, float, float, float],
    use_cache: bool = True,
) -> Set[Tuple[str, str]]:
    """
    Para cada lugar, obtém o nome da rua (do address), consulta Overpass para ruas que tocam essa rua,
    e monta o conjunto de pares (place_id_origem, place_id_destino) onde a rua de destino está conectada
    à rua de origem segundo o OSM. Retorna set de (id_a, id_b); não inclui (i,i).
    """
    south, west, north, east = bbox
    place_id_to_street: Dict[str, str] = {}
    street_to_place_ids: Dict[str, List[str]] = {}
    for p in places:
        pid = p.get("id", "")
        addr = p.get("address", "")
        street = _get_street_name_from_address(addr)
        place_id_to_street[pid] = street
        street_to_place_ids.setdefault(street, []).append(pid)

    cache_osm: Dict[str, Set[str]] = {}
    cache_file = _CACHE_DIR / "osm_connectivity_cache.pickle"
    if use_cache and cache_file.is_file():
        try:
            with open(cache_file, "rb") as f:
                cache_osm = pickle.load(f)
        except Exception:
            cache_osm = {}

    connected_pairs: Set[Tuple[str, str]] = set()
    for p in places:
        pid = p["id"]
        street = place_id_to_street.get(pid, "")
        if not street:
            continue
        if street not in cache_osm:
            cache_osm[street] = _query_overpass_connected_street_names(street, south, west, north, east)
        connected_names = cache_osm[street]
        for other_street in connected_names:
            for other_id in street_to_place_ids.get(other_street, []):
                if other_id != pid:
                    connected_pairs.add((pid, other_id))

    if use_cache and cache_osm:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(cache_osm, f)
        except Exception:
            pass
    return connected_pairs


def _query_overpass_ways_surface(
    south: float,
    west: float,
    north: float,
    east: float,
    timeout: int = 15,
    url: str = OVERPASS_API_URL,
) -> List[str]:
    """
    Consulta a Overpass API por ways com tag 'surface' na bbox (south, west, north, east).
    Retorna lista de valores da tag surface (ex.: ['asphalt', 'cobblestone']).
    """
    query = (
        f'[out:json][timeout:{timeout}];'
        f'way["surface"]({south},{west},{north},{east});'
        "out body;"
    )
    try:
        r = requests.post(url, data={"data": query}, timeout=timeout + 5)
        r.raise_for_status()
        data = r.json()
        surfaces = []
        for el in data.get("elements", []):
            tags = el.get("tags") or {}
            s = tags.get("surface")
            if s:
                surfaces.append(s)
        return surfaces
    except Exception:
        return []


def get_roughness_for_location(
    lat: float,
    lng: float,
    radius_deg: float = 0.001,
) -> float:
    """
    Rugosidade numérica para um ponto (lat, lng), consultando Overpass na bbox ao redor.
    radius_deg: metade do lado da bbox em graus (~0.001 ≈ 100 m).
    Usa surface_to_roughness() para mapear; altere OSM_SURFACE_TO_ROUGHNESS para mudar o mapeamento.
    """
    south, north = lat - radius_deg, lat + radius_deg
    west, east = lng - radius_deg, lng + radius_deg
    surfaces = _query_overpass_ways_surface(south, west, north, east)
    if not surfaces:
        return 1.0
    roughnesses = [surface_to_roughness(s) for s in surfaces]
    return max(roughnesses)


def enrich_graph_with_roughness_from_overpass(
    G: nx.DiGraph,
    radius_deg: float = 0.001,
    default_roughness: float = 1.0,
) -> None:
    """
    Enriquece o grafo com rugosidade por nó e por aresta usando Overpass API (tag surface).
    Para cada nó: consulta ways com surface na bbox ao redor de (lat, lng), mapeia para número,
    armazena em G.nodes[n]['roughness']. Para cada aresta (u,v): usa o máximo das rugosidades de u e v.
    Mapeamento alterável em OSM_SURFACE_TO_ROUGHNESS e surface_to_roughness().
    """
    for nid in G.nodes():
        nd = G.nodes.get(nid, {})
        lat, lng = nd.get("lat"), nd.get("lng")
        if lat is not None and lng is not None:
            G.nodes[nid]["roughness"] = get_roughness_for_location(
                float(lat), float(lng), radius_deg=radius_deg
            )
        else:
            G.nodes[nid]["roughness"] = default_roughness
    for u, v in G.edges():
        ru = G.nodes.get(u, {}).get("roughness", default_roughness)
        rv = G.nodes.get(v, {}).get("roughness", default_roughness)
        G.edges[u, v]["roughness"] = max(ru, rv)


def _get_route_distance_meters(
    api_key: str,
    origin: Tuple[float, float],
    destination: Tuple[float, float],
) -> Optional[float]:
    """Obtém a distância em metros da rota entre origem e destino (Google Directions API)."""
    info = _get_route_info(api_key, origin, destination)
    return info[0] if info else None


def _get_route_info(
    api_key: str,
    origin: Tuple[float, float],
    destination: Tuple[float, float],
) -> Optional[Tuple[float, int]]:
    """
    Obtém (distância em metros, número de steps) da rota entre origem e destino (Google Directions API).
    Cada 'step' = um trecho da rota (ex.: "Siga na Rua X", "Vire à esquerda"). Rotas com 1 step
    (ou poucos) indicam ligação direta entre os dois pontos; muitas steps indicam que a rota
    passa por outras ruas. Retorna None se não houver rota.
    """
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin[0]},{origin[1]}",
        "destination": f"{destination[0]},{destination[1]}",
        "key": api_key,
        "mode": "driving",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "OK" or not data.get("routes"):
            return None
        leg = data["routes"][0]["legs"][0]
        dist = leg["distance"]["value"]
        num_steps = len(leg.get("steps", []))
        return (float(dist), num_steps)
    except Exception:
        return None


def enrich_graph_with_elevation(
    G: nx.DiGraph,
    api_key: Optional[str] = None,
    default_slope_pct: float = 0.0,
) -> None:
    """
    Enriquece o grafo com declividade por aresta usando Google Elevation API.
    Atualiza G.edges[u, v]['slope_pct'] para cada aresta (u, v).
    Nós devem ter atributos 'lat' e 'lng'. Uma única chamada em lote (até 512 nós).
    """
    key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise ValueError(
            "API key do Google Maps não informada. Passe api_key ou defina GOOGLE_MAPS_API_KEY no ambiente."
        )
    node_ids = sorted(G.nodes())
    locations = []
    for nid in node_ids:
        nd = G.nodes.get(nid, {})
        lat, lng = nd.get("lat"), nd.get("lng")
        if lat is None or lng is None:
            locations.append((0.0, 0.0))  # placeholder
        else:
            locations.append((float(lat), float(lng)))
    elevations = _get_elevations_batch(key, locations)
    elev_by_node = {nid: elevations[i] for i, nid in enumerate(node_ids)}
    for u, v in G.edges():
        dist = G.edges[u, v].get("distance")
        eu, ev = elev_by_node.get(u), elev_by_node.get(v)
        if eu is not None and ev is not None and dist is not None and dist > 0:
            G.edges[u, v]["slope_pct"] = _slope_pct_from_elevation(eu, ev, dist)
        else:
            G.edges[u, v]["slope_pct"] = default_slope_pct


def build_graph_from_google_maps(
    places: List[Dict[str, str]],
    api_key: Optional[str] = None,
    ref_lat: float = OURO_PRETO_REF_LAT,
    ref_lng: float = OURO_PRETO_REF_LNG,
    default_slope_pct: float = 0.0,
    default_roughness: float = 1.0,
    default_volume_capacity_ratio: float = 0.0,
    use_elevation: bool = True,
    use_roughness: bool = True,
    use_cache: bool = True,
    use_osm_connectivity: bool = True,
    fetch_distances: bool = True,
    direct_connection_max_steps: Optional[int] = 2,
) -> nx.DiGraph:
    """
    Gera um grafo (NetworkX DiGraph) a partir do Google Maps.

    - places: lista de dicts com "id" e "address" (ex: {"id": "praca_tiradentes", "address": "Praça Tiradentes, Ouro Preto, MG"}).
    - api_key: chave da API Google Maps. Se None, usa a variável de ambiente GOOGLE_MAPS_API_KEY.
    - ref_lat, ref_lng: ponto de referência para converter coordenadas em (x,y) em metros.
    - default_*: valores padrão para atributos das arestas (declividade, rugosidade, congestionamento).
    - use_elevation: se True, chama Google Elevation API para obter elevação de cada nó e calcula slope_pct por aresta.
    - use_roughness: se True, chama Overpass API para obter rugosidade (tag surface) por nó e atualiza roughness nas arestas.
    - use_cache: se True, usa cache de Geocoding e Directions em google_maps_api_cache.pickle para evitar chamadas repetidas.
    - use_osm_connectivity: se True (recomendado), usa Overpass API (OSM) para obter quais ruas estão diretamente ligadas
      (compartilham nós); Directions é chamado apenas para esses pares. Grafo fica esparso e realista.
    - fetch_distances: se True (padrão), obtém distância entre nós via Google Directions. Se False, só constrói o grafo
      com Geocoding (lat/long) e conexões (Overpass); a distância das arestas é a distância em linha reta (metros).
    - direct_connection_max_steps: usado só quando use_osm_connectivity=False. Se definido, só cria aresta quando
      a rota tem no máximo esse número de steps (Directions). Use None para desativar.

    Nós = um por lugar (id, pos em metros, label = address).
    Arestas = entre pares conectados (OSM); distância via Directions se fetch_distances=True, senão linha reta.
    """
    key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise ValueError(
            "API key do Google Maps não informada. Passe api_key ou defina GOOGLE_MAPS_API_KEY no ambiente."
        )

    # Primeira camada de cache: Geocoding e Directions (evita chamadas desnecessárias à API Google Maps)
    cache: Dict[str, Any] = {"geocode": {}, "directions": {}}
    if use_cache and _CACHE_GOOGLE_MAPS.is_file():
        try:
            with open(_CACHE_GOOGLE_MAPS, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {"geocode": {}, "directions": {}}

    def _directions_key(orig: Tuple[float, float], dest: Tuple[float, float]) -> Tuple[float, float, float, float]:
        return (round(orig[0], 6), round(orig[1], 6), round(dest[0], 6), round(dest[1], 6))

    G = nx.DiGraph()
    G.graph[KEY_RAIN_MULTIPLIER] = 1.0
    G.graph[KEY_RAIN_MULTIPLIER_BY_REGION] = {}
    G.graph[KEY_SLOPE_PENALTY_FACTOR] = 1.0
    G.graph[KEY_CONGESTION_FACTOR] = 1.0
    G.graph[KEY_CONGESTION_FACTOR_BY_REGION] = {}
    G.graph[KEY_EDGE_OVERRIDE] = {}
    G.graph[KEY_EDGE_COST_MULTIPLIER] = {}

    print(f"Building graph from Google Maps (use_elevation={use_elevation}, use_roughness={use_roughness}, use_osm_connectivity={use_osm_connectivity}, fetch_distances={fetch_distances})")
    # Geocodificar todos os lugares (com cache)
    coords: Dict[str, Tuple[float, float]] = {}
    for p in places:
        pid = p["id"]
        addr = p["address"]
        if use_cache and addr in cache["geocode"]:
            lat_lng = cache["geocode"][addr]
        else:
            lat_lng = _geocode(key, addr)
            if lat_lng is None:
                raise ValueError(f"Geocoding falhou para: {addr}")
            if use_cache:
                cache["geocode"][addr] = lat_lng
        coords[pid] = lat_lng
        x, y = _lat_lng_to_xy_meters(lat_lng[0], lat_lng[1], ref_lat, ref_lng)
        region = p.get("region", _get_region(pid) if _get_region(pid) != "default" else pid)
        G.add_node(pid, pos=(x, y), label=addr, lat=lat_lng[0], lng=lat_lng[1], region=region)

    # Pares (origem, destino) para os quais pedir Directions: OSM conectados ou todos
    if use_osm_connectivity:
        connected_pairs = get_osm_connected_pairs(places, OURO_PRETO_BBOX, use_cache=use_cache)
        print(f"OSM conectividade: {len(connected_pairs)} pares de ruas diretamente ligadas.")
    else:
        connected_pairs = None  # todos os pares (exceto i==j)

    # Para cada par permitido, obter rota (Directions) e adicionar aresta (com cache)
    if connected_pairs is not None:
        pairs_to_fetch = list(connected_pairs)
    else:
        pairs_to_fetch = [(p_orig["id"], p_dest["id"]) for i, p_orig in enumerate(places) for j, p_dest in enumerate(places) if i != j]

    for orig_id, dest_id in pairs_to_fetch:
        if fetch_distances:
            dkey = _directions_key(coords[orig_id], coords[dest_id])
            if use_cache and dkey in cache["directions"]:
                cached = cache["directions"][dkey]
                if isinstance(cached, tuple):
                    dist, num_steps = cached
                else:
                    dist, num_steps = cached, 1
            else:
                info = _get_route_info(key, coords[orig_id], coords[dest_id])
                if info is not None:
                    dist, num_steps = info
                    if use_cache:
                        cache["directions"][dkey] = (dist, num_steps)
                else:
                    dist, num_steps = None, 0
            if dist is None or dist <= 0:
                continue
            add_edge = use_osm_connectivity or (direct_connection_max_steps is None or num_steps <= direct_connection_max_steps)
        else:
            dist = _straight_line_distance_meters(
                coords[orig_id][0], coords[orig_id][1],
                coords[dest_id][0], coords[dest_id][1],
                ref_lat, ref_lng,
            )
            add_edge = True
        if add_edge and dist > 0:
            G.add_edge(
                orig_id,
                dest_id,
                distance=float(dist),
                slope_pct=default_slope_pct,
                roughness=default_roughness,
                volume_capacity_ratio=default_volume_capacity_ratio,
            )

    if use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(_CACHE_GOOGLE_MAPS, "wb") as f:
                pickle.dump(cache, f)
        except Exception:
            pass

    if use_elevation:
        print("Enriching graph with elevation...")
        enrich_graph_with_elevation(G, api_key=key, default_slope_pct=default_slope_pct)
    if use_roughness:
        print("Enriching graph with roughness...")
        enrich_graph_with_roughness_from_overpass(G, default_roughness=default_roughness)
    print("Graph built successfully")   

    return G


# Lista padrão: somente ruas da sede de Ouro Preto, MG (sem pontos de interesse).
# Mapeamento: id -> endereço para geocoding e grafo. Duplicatas por (lat, lng) removidas.
# Lista padrão: somente ruas da sede de Ouro Preto, MG (sem pontos de interesse).
# Mapeamento: id -> endereço para geocoding e grafo. Duplicatas por (lat, lng) removidas.
DEFAULT_OURO_PRETO_PLACES = [
    # {"id": "sao_jose", "address": "Rua São José, Ouro Preto, MG"},
    # {"id": "conde_bobadela", "address": "Rua Conde de Bobadela, Ouro Preto, MG"},
    # {"id": "rua_pilar", "address": "Rua do Pilar, Ouro Preto, MG"},
    # {"id": "xavier_veiga", "address": "Rua Xavier da Veiga, Ouro Preto, MG"},
    # {"id": "padre_rolim", "address": "Rua Padre Rolim, Ouro Preto, MG"},
    # {"id": "claudio_manoel", "address": "Rua Cláudio Manoel, Ouro Preto, MG"},
    # {"id": "conselheiro_quintiliano", "address": "Rua Conselheiro Quintiliano, Ouro Preto, MG"},
    # {"id": "paulistas", "address": "Rua dos Paulistas, Ouro Preto, MG"},
    # {"id": "getulio_vargas", "address": "Rua Getúlio Vargas, Ouro Preto, MG"},
    # {"id": "henri_gorceix", "address": "Rua Henri Gorceix, Ouro Preto, MG"},
    # {"id": "joao_de_paiva", "address": "Rua João de Paiva, Ouro Preto, MG"},
    # {"id": "teixeira_amaral", "address": "Rua Teixeira Amaral, Ouro Preto, MG"},
    # {"id": "coronel_alves", "address": "Rua Coronel Alves, Ouro Preto, MG"},
    # {"id": "conego_trindade", "address": "Rua Cônego Trindade, Ouro Preto, MG"},
    # {"id": "direita", "address": "Rua Direita, Ouro Preto, MG"},
    # {"id": "bernardo_vasconcelos", "address": "Rua Bernardo Vasconcelos, Ouro Preto, MG"},
    # {"id": "sao_francisco_assis", "address": "Rua São Francisco de Assis, Ouro Preto, MG"},
    # {"id": "dom_silverio", "address": "Rua Dom Silvério, Ouro Preto, MG"},
    # {"id": "camilo_brito", "address": "Rua Camilo de Brito, Ouro Preto, MG"},
    # {"id": "parana", "address": "Rua Paraná, Ouro Preto, MG"},
    # {"id": "alvarenga", "address": "Rua Alvarenga, Ouro Preto, MG"},
    # {"id": "flores", "address": "Rua das Flores, Ouro Preto, MG"},
    # {"id": "conceicao", "address": "Rua da Conceição, Ouro Preto, MG"},
    {"id": "ouro", "address": "Rua do Ouro, Ouro Preto, MG"},
    {"id": "cruz", "address": "Rua da Cruz, Ouro Preto, MG"},
    {"id": "rosario", "address": "Rua do Rosário, Ouro Preto, MG"},
    {"id": "merces", "address": "Rua das Mercês, Ouro Preto, MG"},
    {"id": "chafariz", "address": "Rua do Chafariz, Ouro Preto, MG"},
    {"id": "prata", "address": "Rua da Prata, Ouro Preto, MG"},
    {"id": "sao_miguel", "address": "Rua São Miguel, Ouro Preto, MG"},
    {"id": "santa_rita", "address": "Rua Santa Rita, Ouro Preto, MG"},
    {"id": "lajes", "address": "Rua das Lajes, Ouro Preto, MG"},
    {"id": "ladeira_barra", "address": "Ladeira da Barra, Ouro Preto, MG"},
    {"id": "ladeira_santa_efigenia", "address": "Ladeira de Santa Efigênia, Ouro Preto, MG"},
    {"id": "ladeira_custodio", "address": "Ladeira do Custódio, Ouro Preto, MG"},
    {"id": "ladeira_piedade", "address": "Ladeira da Piedade, Ouro Preto, MG"},
    {"id": "morro_da_queimada", "address": "Rua Morro da Queimada, Ouro Preto, MG"},
    {"id": "rio_piracicaba", "address": "Rua Rio Piracicaba, Ouro Preto, MG"},
    {"id": "rio_itabira", "address": "Rua Rio Itabira, Ouro Preto, MG"},
    {"id": "treze_maio", "address": "Rua Treze de Maio, Ouro Preto, MG"},
    {"id": "vinte_e_quatro_de_junho", "address": "Rua Vinte e Quatro de Junho, Ouro Preto, MG"},
    {"id": "quinze_de_agosto", "address": "Rua Quinze de Agosto, Ouro Preto, MG"},
    {"id": "santos_dumont", "address": "Rua Santos Dumont, Ouro Preto, MG"},
    {"id": "americo_rene_gianetti", "address": "Avenida Américo Renê Gianetti, Ouro Preto, MG"},
    {"id": "simao_lacerda", "address": "Rua Simão Lacerda, Ouro Preto, MG"},
    {"id": "hugo_soderi", "address": "Rua Hugo Soderi, Ouro Preto, MG"},
    {"id": "domingos_mendes", "address": "Rua Domingos Mendes, Ouro Preto, MG"},
    # {"id": "cristo_rei", "address": "Rua Cristo Rei, Ouro Preto, MG"},
    # {"id": "alberto_ansaloni", "address": "Rua Alberto Ansaloni, Ouro Preto, MG"},
    # {"id": "rua_quatorze", "address": "Rua Quatorze, Ouro Preto, MG"},
    # {"id": "rua_onze", "address": "Rua Onze, Ouro Preto, MG"},
    # {"id": "rua_dois", "address": "Rua Dois, Ouro Preto, MG"},
    # {"id": "rua_tres", "address": "Rua Três, Ouro Preto, MG"},
    # {"id": "rua_quatro", "address": "Rua Quatro, Ouro Preto, MG"},
    # {"id": "rua_nove", "address": "Rua Nove, Ouro Preto, MG"},
    # {"id": "joao_fernandes_vieira", "address": "Rua João Fernandes Vieira, Ouro Preto, MG"},
    # {"id": "jose_aureliano_leocadio", "address": "Rua José Aureliano Leocádio, Ouro Preto, MG"},
    # {"id": "professor_paulo_magalhaes_gomes", "address": "Rua Professor Paulo Magalhães Gomes, Ouro Preto, MG"},
    # {"id": "rua_alfa", "address": "Rua Alfa, Ouro Preto, MG"},
    # {"id": "perimetral", "address": "Rua Perimetral, Ouro Preto, MG"},
    # {"id": "itacolomi", "address": "Rua Itacolomi, Ouro Preto, MG"},
    # {"id": "amaro_lanari", "address": "Rua Amaro Lanari, Ouro Preto, MG"},
    # {"id": "lucio", "address": "Rua Lúcio, Ouro Preto, MG"},
    # {"id": "professor_geraldo_nunes", "address": "Rua Professor Geraldo Nunes, Ouro Preto, MG"},
    # {"id": "alexandre_kassis", "address": "Rua Alexandre Kassis, Ouro Preto, MG"},
    # {"id": "geraldo_quirino_ribeiro", "address": "Rua Geraldo Quirino Ribeiro, Ouro Preto, MG"},
    # {"id": "vereador_paulo_elias", "address": "Rua Vereador Paulo Elías, Ouro Preto, MG"},
    # {"id": "alvaro_guimaraes_bressan", "address": "Rua Álvaro Guimarães Bressan, Ouro Preto, MG"},
    # {"id": "juscelino_kubitscheck", "address": "Avenida Juscelino Kubitscheck, Ouro Preto, MG"},
    # {"id": "rua_um", "address": "Rua Um, Ouro Preto, MG"},
    # {"id": "rodovia_dos_inconfidentes", "address": "Rodovia dos Inconfidentes, Ouro Preto, MG"},
    # {"id": "jussara_gabriele", "address": "Rua Jussara Gabriele, Ouro Preto, MG"},
    # {"id": "oriente", "address": "Rua Oriente, Ouro Preto, MG"},
    # {"id": "heli_coelho_neto", "address": "Rua Heli Coelho Neto, Ouro Preto, MG"},
]


def build_ouro_preto_example(
    api_key: Optional[str] = None,
    use_cache: bool = True,
    force_rebuild: bool = False,
    fetch_distances: bool = True,
    use_elevation: bool = True,
    use_roughness: bool = True,
    use_osm_connectivity: bool = True,
) -> nx.DiGraph:
    """
    Gera o grafo de Ouro Preto a partir de Geocoding (endereço → lat/long) e conexões (Overpass).
    Na primeira execução salva em cache/grafo_ouro_preto.gpickle; nas seguintes carrega do cache.

    - api_key: opcional; se não informada, usa GOOGLE_MAPS_API_KEY (necessária para Geocoding).
    - use_cache: se True (padrão), carrega/grava em cache/.
    - force_rebuild: se True, ignora cache e reconstrói via API.
    - fetch_distances: se True (padrão), obtém distância entre nós via Google Directions. Se False,
      só constrói o grafo com Geocoding e Overpass (conexões); distância das arestas = linha reta em metros.
    - Inclui somente ruas da sede (centro histórico e arredores), sem pontos de interesse.
    """
    if use_cache and not force_rebuild and _CACHE_OURO_PRETO.is_file():
        with open(_CACHE_OURO_PRETO, "rb") as f:
            return pickle.load(f)
    print("Building Ouro Preto graph from Google Maps...")
    G = build_graph_from_google_maps(
        DEFAULT_OURO_PRETO_PLACES,
        api_key=api_key,
        fetch_distances=fetch_distances,
        use_elevation=use_elevation,
        use_roughness=use_roughness,
        use_osm_connectivity=use_osm_connectivity,
        use_cache=use_cache,
    )
    if use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_OURO_PRETO, "wb") as f:
            pickle.dump(G, f)
        print(f"Grafo Ouro Preto salvo em {_CACHE_OURO_PRETO}")
    return G

# Tempo de espera (segundos) antes de retentar uma consulta Overpass que falhou
_OVERPASS_RETRY_DELAY_SEC = 10
# Para 429 (rate limit) ou 504 (gateway timeout): espera 30s e não conta como retentativa
_OVERPASS_RATE_LIMIT_DELAY_SEC = 30
# Número máximo de retentativas; após isso ignora e retorna None (chamador segue para o próximo)
_OVERPASS_MAX_RETRIES = 5


def _overpass_post(query: str, timeout: int = 90, url: str = OVERPASS_API_URL) -> Optional[Dict[str, Any]]:
    """Envia query à Overpass API e retorna o JSON ou None em caso de erro. 429/504: espera 30s sem contar retry; outros erros: 10s até 5 retentativas."""
    last_error: Optional[Exception] = None
    retry_count = 0
    while retry_count <= _OVERPASS_MAX_RETRIES:
        try:
            r = requests.post(url, data={"data": query}, timeout=timeout + 10)
            r.raise_for_status()
            data = r.json()
            if data is not None:
                return data
        except requests.exceptions.HTTPError as e:
            last_error = e
            status = e.response.status_code if e.response is not None else None
            if status in (429, 504):
                print(f"  Overpass: {status} (rate limit/timeout), aguardando {_OVERPASS_RATE_LIMIT_DELAY_SEC}s (não contabiliza retentativa)...")
                time.sleep(_OVERPASS_RATE_LIMIT_DELAY_SEC)
                continue
            print(f"  Overpass: erro na consulta — {type(e).__name__}: {e}")
        except Exception as e:
            last_error = e
            print(f"  Overpass: erro na consulta — {type(e).__name__}: {e}")
        retry_count += 1
        if retry_count <= _OVERPASS_MAX_RETRIES:
            print(f"  Overpass: tentativa {retry_count}/{_OVERPASS_MAX_RETRIES + 1}, aguardando {_OVERPASS_RETRY_DELAY_SEC}s para retentar...")
            time.sleep(_OVERPASS_RETRY_DELAY_SEC)
        else:
            if last_error is not None:
                print(f"  Overpass: ignorando após {_OVERPASS_MAX_RETRIES} retentativas (último erro: {last_error}), seguindo para o próximo.")
            else:
                print(f"  Overpass: ignorando após {_OVERPASS_MAX_RETRIES} retentativas, seguindo para o próximo.")
    return None


def _way_center_from_geometry(geometry: List[Dict[str, float]]) -> Tuple[float, float]:
    """Calcula (lat, lng) do centro de uma way a partir da lista geometry (Overpass)."""
    if not geometry:
        return (OURO_PRETO_REF_LAT, OURO_PRETO_REF_LNG)
    lat_sum = sum(p.get("lat", 0) for p in geometry)
    lon_sum = sum(p.get("lon", 0) for p in geometry)
    n = len(geometry)
    return (lat_sum / n, lon_sum / n)


def _osm_ways_by_name(
    street_name: str,
    south: float, west: float, north: float, east: float,
) -> List[Dict[str, Any]]:
    """Retorna lista de ways OSM com o nome dado na bbox (cada item: id, name, nodes, geometry)."""
    name_safe = street_name.replace('"', " ").strip()
    if not name_safe:
        return []
    query = (
        f'[out:json][timeout:60];'
        f'way["name"="{name_safe}"]["highway"]({south},{west},{north},{east});'
        "out body geom;"
    )
    data = _overpass_post(query)
    if not data or "elements" not in data:
        return []
    out = []
    for el in data["elements"]:
        if el.get("type") != "way":
            continue
        tags = el.get("tags") or {}
        name = tags.get("name") or ""
        nodes = el.get("nodes") or []
        geometry = el.get("geometry") or []
        out.append({"id": el["id"], "name": name, "nodes": nodes, "geometry": geometry})
    return out


# Tamanho do lote de way IDs por request Overpass (menor = menos timeout/rate limit)
_OSM_CONNECTED_WAYS_BATCH_SIZE = 40
# Pausa entre lotes (segundos) para evitar rate limit do servidor Overpass
_OSM_BATCH_DELAY_SEC = 1.5


def _way_neighbors_cache_key(south: float, west: float, north: float, east: float, way_id: int) -> Tuple[float, float, float, float, int]:
    """Chave hashável para cache de vizinhos de uma way (bbox arredondada + way_id)."""
    return (round(south, 6), round(west, 6), round(north, 6), round(east, 6), way_id)


def _get_way_neighbors_cached(
    way_id: int,
    south: float, west: float, north: float, east: float,
    way_neighbors_cache: Dict[Tuple[float, float, float, float, int], List[Dict[str, Any]]],
    cache_file: Path,
    use_cache: bool,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Retorna (ways conectadas à way_id na bbox, from_cache). Em cache hit não chama Overpass."""
    key = _way_neighbors_cache_key(south, west, north, east, way_id)
    if use_cache and key in way_neighbors_cache:
        return (way_neighbors_cache[key], True)
    result = _osm_ways_connected_to_way_ids([way_id], south, west, north, east)
    if use_cache:
        way_neighbors_cache[key] = result
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(way_neighbors_cache, f)
        except Exception:
            pass
    return (result, False)


def _osm_ways_connected_to_way_ids(
    way_ids: List[int],
    south: float, west: float, north: float, east: float,
) -> List[Dict[str, Any]]:
    """Retorna ways (highway) que compartilham pelo menos um nó com alguma das way_ids, na bbox.
    Faz requisições em lotes com pausa entre elas para evitar rate limit e timeout."""
    if not way_ids:
        return []
    seen_ids: Set[int] = set()
    out: List[Dict[str, Any]] = []
    n_batches = (len(way_ids) + _OSM_CONNECTED_WAYS_BATCH_SIZE - 1) // _OSM_CONNECTED_WAYS_BATCH_SIZE
    for i in range(0, len(way_ids), _OSM_CONNECTED_WAYS_BATCH_SIZE):
        if i > 0:
            time.sleep(_OSM_BATCH_DELAY_SEC)
        batch = way_ids[i : i + _OSM_CONNECTED_WAYS_BATCH_SIZE]
        ids_str = ",".join(str(w) for w in batch)
        query = (
            f'[out:json][timeout:120];'
            f'way({ids_str})({south},{west},{north},{east});'
            "node(w)->.n;"
            f'way(bn.n)["highway"]({south},{west},{north},{east});'
            "out body geom;"
        )
        data = _overpass_post(query)
        if not data or "elements" not in data:
            if n_batches > 1:
                print(f"  Overpass: lote {i // _OSM_CONNECTED_WAYS_BATCH_SIZE + 1}/{n_batches} sem dados (timeout/erro)")
            continue
        for el in data["elements"]:
            if el.get("type") != "way":
                continue
            wid = el["id"]
            if wid in seen_ids:
                continue
            seen_ids.add(wid)
            tags = el.get("tags") or {}
            name = tags.get("name") or f"Way {el['id']}"
            nodes = el.get("nodes") or []
            geometry = el.get("geometry") or []
            out.append({"id": wid, "name": name, "nodes": nodes, "geometry": geometry})
    return out


def _slug_from_name(name: str) -> str:
    """Gera id de nó a partir de tags.name: minúsculas, espaços e caracteres especiais -> underscore."""
    if not name or not str(name).strip():
        return ""
    s = str(name).strip().lower()
    s = s.replace(" ", "_").replace("-", "_").replace(".", "_")
    s = "".join(c if c.isalnum() or c == "_" else "_" for c in s)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def _build_way_graph_from_ways(
    ways_by_id: Dict[int, Dict[str, Any]],
    node_to_way_ids: Dict[int, Set[int]],
    ref_lat: float,
    ref_lng: float,
    default_slope_pct: float = 0.0,
    default_roughness: float = 1.0,
    default_volume_capacity_ratio: float = 0.0,
) -> nx.DiGraph:
    """Monta DiGraph a partir de ways e mapa nó->ways. Vértices usam tags.name como id (slug)."""
    G = nx.DiGraph()
    G.graph[KEY_RAIN_MULTIPLIER] = 1.0
    G.graph[KEY_RAIN_MULTIPLIER_BY_REGION] = {}
    G.graph[KEY_SLOPE_PENALTY_FACTOR] = 1.0
    G.graph[KEY_CONGESTION_FACTOR] = 1.0
    G.graph[KEY_CONGESTION_FACTOR_BY_REGION] = {}
    G.graph[KEY_EDGE_OVERRIDE] = {}
    G.graph[KEY_CONGESTION_FACTOR_BY_EDGE] = {}
    G.graph[KEY_EDGE_COST_MULTIPLIER] = {}

    way_id_to_node_id: Dict[int, str] = {}
    used_slugs: Set[str] = set()

    for wid, w in ways_by_id.items():
        name = (w.get("name") or "").strip()
        if name:
            slug = _slug_from_name(name)
            if not slug:
                slug = f"op_w{wid}"
            elif slug in used_slugs:
                slug = f"{slug}_{wid}"
            used_slugs.add(slug)
            nid = slug
            label = nid  # nome exibido = id da rua em snake_case
        else:
            nid = f"passagem_{wid}"
            used_slugs.add(nid)
            label = nid  # nome exibido = id em snake_case
        way_id_to_node_id[wid] = nid
        lat, lng = _way_center_from_geometry(w.get("geometry") or [])
        x, y = _lat_lng_to_xy_meters(lat, lng, ref_lat, ref_lng)
        G.add_node(nid, pos=(x, y), label=label, lat=lat, lng=lng, region="op")

    for node_osm_id, wids in node_to_way_ids.items():
        wids_list = [w for w in wids if w in ways_by_id]
        for i, wa in enumerate(wids_list):
            for wb in wids_list[i + 1 :]:
                nid_a, nid_b = way_id_to_node_id[wa], way_id_to_node_id[wb]
                if nid_a == nid_b:
                    continue
                if not G.has_node(nid_a) or not G.has_node(nid_b):
                    continue
                lat_a, lng_a = _way_center_from_geometry(ways_by_id[wa].get("geometry") or [])
                lat_b, lng_b = _way_center_from_geometry(ways_by_id[wb].get("geometry") or [])
                dist = _straight_line_distance_meters(lat_a, lng_a, lat_b, lng_b, ref_lat, ref_lng)
                dist = max(dist, 1e-6)
                if not G.has_edge(nid_a, nid_b):
                    G.add_edge(
                        nid_a, nid_b,
                        distance=dist,
                        slope_pct=default_slope_pct,
                        roughness=default_roughness,
                        volume_capacity_ratio=default_volume_capacity_ratio,
                    )
                if not G.has_edge(nid_b, nid_a):
                    G.add_edge(
                        nid_b, nid_a,
                        distance=dist,
                        slope_pct=default_slope_pct,
                        roughness=default_roughness,
                        volume_capacity_ratio=default_volume_capacity_ratio,
                    )
    return G


def _expand_op_graph_tail(
    bbox: Tuple[float, float, float, float],
    frontier: List[int],
    seen_way_ids: Set[int],
    ways_by_id: Dict[int, Dict[str, Any]],
    node_to_way_ids: Dict[int, Set[int]],
    current_level: int,
    max_level: int,
    way_neighbors_cache: Dict[Tuple[float, float, float, float, int], List[Dict[str, Any]]],
    way_neighbors_cache_file: Path,
    use_cache: bool,
) -> None:
    """
    Expansão iterativa (BFS por níveis): a partir da fronteira atual (way ids),
    obtém todos os vizinhos (ways que compartilham nó), adiciona ao grafo e
    repete com a nova fronteira até atingir max_level ou fronteira vazia.
    Usa cache incremental de vizinhos por way (arquivo em cache/). Mesmo contrato que a versão recursiva.
    """
    south, west, north, east = bbox
    while current_level < max_level and frontier:
        print(f"build_op_graph: iniciando nivel {current_level}/{max_level} — fronteira={len(frontier)} ways")
        # Chamar Overpass (ou cache) para cada filho da fronteira (cada way), agregando resultados.
        # Sleep apenas entre chamadas à API; itens em cache não disparam sleep.
        connected: List[Dict[str, Any]] = []
        seen_connected_ids: Set[int] = set()
        previous_was_from_cache = True
        for i, wid in enumerate(frontier):
            if i > 0 and not previous_was_from_cache:
                time.sleep(_OSM_BATCH_DELAY_SEC)
            batch_result, from_cache = _get_way_neighbors_cached(
                wid, south, west, north, east,
                way_neighbors_cache, way_neighbors_cache_file, use_cache,
            )
            previous_was_from_cache = from_cache
            for w in batch_result:
                if w["id"] not in seen_connected_ids:
                    seen_connected_ids.add(w["id"])
                    connected.append(w)
            # A cada 10 ways da fronteira processadas, exibir progresso
            processed = i + 1
            if processed % 10 == 0 or processed == len(frontier):
                print(f"build_op_graph: nivel {current_level} — {processed}/{len(frontier)} ways da fronteira processadas")
        next_frontier: List[int] = []
        for w in connected:
            wid = w["id"]
            if wid not in seen_way_ids:
                seen_way_ids.add(wid)
                ways_by_id[wid] = w
                next_frontier.append(wid)
                for nid in w.get("nodes") or []:
                    node_to_way_ids.setdefault(nid, set()).add(wid)
        print(f"build_op_graph: nivel {current_level}/{max_level} — fronteira={len(frontier)} ways, vizinhos obtidos={len(connected)}, novos={len(next_frontier)}")
        frontier = next_frontier
        current_level += 1


def build_op_graph(
    levels: int = 4,
    ref_lat: float = OURO_PRETO_REF_LAT,
    ref_lng: float = OURO_PRETO_REF_LNG,
    default_slope_pct: float = 0.0,
    default_roughness: float = 1.0,
    default_volume_capacity_ratio: float = 0.0,
    use_cache: bool = True,
    force_rebuild: bool = False,
) -> nx.DiGraph:
    """
    Constrói grafo das ruas de Ouro Preto (sede) via OpenStreetMap (Overpass API).
    Começa pela Rua Rio Piracicaba e expande por BFS: a cada nível, inclui ruas que compartilham
    um nó com as ruas já mapeadas. Restringe à bbox de Ouro Preto (sem distritos vizinhos).

    - levels: número de níveis de expansão (default 4). Nível 0 = só Rio Piracicaba; 1 = ruas que
      tocam nela; 2 = ruas que tocam as do nível 1; etc.
    - ref_lat, ref_lng: referência para pos (metros) e distâncias.
    - default_*: atributos das arestas (slope_pct, roughness, volume_capacity_ratio).
    - use_cache: se True (padrão), carrega/grava em cache/grafo_op_osm.gpickle.
    - force_rebuild: se True, ignora cache e reconstrói via Overpass.

    Retorna o mesmo tipo que build_ouro_preto_example: nx.DiGraph com nós (pos, label, lat, lng, region)
    e arestas (distance, slope_pct, roughness, volume_capacity_ratio). IDs dos nós = slug de tags.name
    (ex.: "rua_rio_piracicaba"); se name vazio ou duplicado, usa op_w<osm_way_id> ou nome_wid.
    """
    if use_cache and not force_rebuild and _CACHE_OP_GRAPH.is_file():
        try:
            with open(_CACHE_OP_GRAPH, "rb") as f:
                G = pickle.load(f)
            print(f"build_op_graph: carregado do cache ({G.number_of_nodes()} ruas, {G.number_of_edges()} arestas)")
            return G
        except Exception:
            pass

    south, west, north, east = OURO_PRETO_BBOX

    # Ponto de partida: Rua Rio Piracicaba (tentar nomes comuns no OSM)
    start_names = ["Rua Rio Piracicaba", "Rio Piracicaba"]
    seed_ways: List[Dict[str, Any]] = []
    for name in start_names:
        seed_ways = _osm_ways_by_name(name, south, west, north, east)
        if seed_ways:
            break
    if not seed_ways:
        raise ValueError("Nenhuma way 'Rua Rio Piracicaba' (ou 'Rio Piracicaba') encontrada na bbox de Ouro Preto. Verifique a bbox ou o nome no OSM.")

    seen_way_ids: Set[int] = set()
    ways_by_id: Dict[int, Dict[str, Any]] = {}
    node_to_way_ids: Dict[int, Set[int]] = {}

    for w in seed_ways:
        wid = w["id"]
        seen_way_ids.add(wid)
        ways_by_id[wid] = w
        for nid in w.get("nodes") or []:
            node_to_way_ids.setdefault(nid, set()).add(wid)

    # Cache incremental: vizinhos por way (carregar existente se use_cache)
    way_neighbors_cache: Dict[Tuple[float, float, float, float, int], List[Dict[str, Any]]] = {}
    if use_cache and _CACHE_OSM_WAY_NEIGHBORS.is_file():
        try:
            with open(_CACHE_OSM_WAY_NEIGHBORS, "rb") as f:
                way_neighbors_cache = pickle.load(f)
            print(f"build_op_graph: cache de vizinhos carregado ({len(way_neighbors_cache)} ways em cache)")
        except Exception:
            way_neighbors_cache = {}

    # Expansão recursiva em cauda: a partir do ponto de partida, setar vizinhos e para cada um setar os seus
    _expand_op_graph_tail(
        (south, west, north, east),
        list(seen_way_ids),
        seen_way_ids,
        ways_by_id,
        node_to_way_ids,
        current_level=1,
        max_level=levels,
        way_neighbors_cache=way_neighbors_cache,
        way_neighbors_cache_file=_CACHE_OSM_WAY_NEIGHBORS,
        use_cache=use_cache,
    )

    # Só manter nós que aparecem em pelo menos 2 ways (para arestas entre ways)
    node_to_way_ids = {n: s for n, s in node_to_way_ids.items() if len(s) >= 2}

    G = _build_way_graph_from_ways(
        ways_by_id,
        node_to_way_ids,
        ref_lat, ref_lng,
        default_slope_pct=default_slope_pct,
        default_roughness=default_roughness,
        default_volume_capacity_ratio=default_volume_capacity_ratio,
    )
    print(f"build_op_graph: {G.number_of_nodes()} ruas, {G.number_of_edges()} arestas (levels={levels})")
    if use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(_CACHE_OP_GRAPH, "wb") as f:
                pickle.dump(G, f)
            print(f"Grafo OP (OSM) salvo em {_CACHE_OP_GRAPH}")
        except Exception:
            pass
    return G