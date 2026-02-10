"""
Montagem do grafo e armazenamento em cache. Consome OverpassApiClient (OSM) e GoogleApiClient (conversão de coordenadas no build OSM).
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from .clients import GoogleApiClient, OverpassApiClient
from .clients.google_api_client import OURO_PRETO_REF_LAT, OURO_PRETO_REF_LNG
from .graph_operations import (
    KEY_CONGESTION_FACTOR,
    KEY_CONGESTION_FACTOR_BY_EDGE,
    KEY_CONGESTION_FACTOR_BY_REGION,
    KEY_EDGE_OVERRIDE,
    KEY_EDGE_COST_MULTIPLIER,
    KEY_RAIN_MULTIPLIER,
    KEY_RAIN_MULTIPLIER_BY_REGION,
    KEY_SLOPE_PENALTY_FACTOR,
)

# Diretório de cache (raiz do projeto)
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

# Bbox Ouro Preto — sede (south, west, north, east) em graus
_OURO_PRETO_RADIUS_DEG = 0.04
OURO_PRETO_BBOX = (
    OURO_PRETO_REF_LAT - _OURO_PRETO_RADIUS_DEG,
    OURO_PRETO_REF_LNG - _OURO_PRETO_RADIUS_DEG,
    OURO_PRETO_REF_LAT + _OURO_PRETO_RADIUS_DEG,
    OURO_PRETO_REF_LNG + _OURO_PRETO_RADIUS_DEG,
)

_OSM_BATCH_DELAY_SEC = 1.5


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


def _way_neighbors_cache_key(
    south: float, west: float, north: float, east: float, way_id: int
) -> Tuple[float, float, float, float, int]:
    return (round(south, 6), round(west, 6), round(north, 6), round(east, 6), way_id)


def _default_graph_metadata() -> Dict[str, Any]:
    """Metadados padrão do grafo (chaves globais do NetworkX DiGraph)."""
    return {
        KEY_RAIN_MULTIPLIER: 1.0,
        KEY_RAIN_MULTIPLIER_BY_REGION: {},
        KEY_SLOPE_PENALTY_FACTOR: 1.0,
        KEY_CONGESTION_FACTOR: 1.0,
        KEY_CONGESTION_FACTOR_BY_REGION: {},
        KEY_EDGE_OVERRIDE: {},
        KEY_CONGESTION_FACTOR_BY_EDGE: {},
        KEY_EDGE_COST_MULTIPLIER: {},
    }


def _way_to_node_id_and_label(
    way: Dict[str, Any], way_id: int, used_slugs: Set[str]
) -> Tuple[str, str]:
    """Define node_id e label para uma way; atualiza used_slugs."""
    name = (way.get("name") or "").strip()
    if name:
        slug = _slug_from_name(name) or f"op_w{way_id}"
        if slug in used_slugs:
            slug = f"{slug}_{way_id}"
        used_slugs.add(slug)
        return (slug, slug)
    node_id = f"passagem_{way_id}"
    used_slugs.add(node_id)
    return (node_id, node_id)


def _persist_way_neighbors_cache(
    cache_dir: Path, cache_file: Path, cache_data: Dict[Tuple[float, float, float, float, int], List[Dict[str, Any]]]
) -> None:
    """Grava cache de vizinhos de ways em disco (sem falhar o fluxo)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)
    except Exception:
        pass


def _init_ways_state_from_seeds(
    seed_ways: List[Dict[str, Any]],
) -> Tuple[Set[int], Dict[int, Dict[str, Any]], Dict[int, Set[int]]]:
    """A partir das seed ways, preenche seen_way_ids, ways_by_id e node_to_way_ids. Retorna (seen_way_ids, ways_by_id, node_to_way_ids)."""
    seen_way_ids: Set[int] = set()
    ways_by_id: Dict[int, Dict[str, Any]] = {}
    node_to_way_ids: Dict[int, Set[int]] = {}
    for way in seed_ways:
        way_id = way["id"]
        seen_way_ids.add(way_id)
        ways_by_id[way_id] = way
        for node_osm_id in way.get("nodes") or []:
            node_to_way_ids.setdefault(node_osm_id, set()).add(way_id)
    return (seen_way_ids, ways_by_id, node_to_way_ids)


class BuildGraph:
    """Responsável somente por montar o grafo e armazená-lo no cache. Consome os clients de API."""

    def __init__(
        self,
        google_client: Optional[GoogleApiClient] = None,
        overpass_client: Optional[OverpassApiClient] = None,
        cache_dir: Optional[Path] = None,
    ):
        self._google = google_client or GoogleApiClient()
        self._overpass = overpass_client or OverpassApiClient(cache_dir=cache_dir or _DEFAULT_CACHE_DIR)
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def cache_op_graph(self) -> Path:
        return self._cache_dir / "grafo_op_osm.gpickle"

    @property
    def cache_osm_way_neighbors(self) -> Path:
        return self._cache_dir / "osm_way_neighbors_cache.pickle"

    def _add_edges_at_intersection(
        self,
        G: nx.DiGraph,
        way_ids_at_node: Set[int],
        ways_by_id: Dict[int, Dict[str, Any]],
        way_id_to_node_id: Dict[int, str],
        ref_lat: float,
        ref_lng: float,
        default_slope_pct: float,
    ) -> None:
        """Adiciona arestas bidirecionais entre pares de ways que se cruzam no mesmo nó OSM."""
        wids_list = [wid for wid in way_ids_at_node if wid in ways_by_id]
        for i, way_id_a in enumerate(wids_list):
            for way_id_b in wids_list[i + 1 :]:
                nid_a = way_id_to_node_id[way_id_a]
                nid_b = way_id_to_node_id[way_id_b]
                if nid_a == nid_b or not G.has_node(nid_a) or not G.has_node(nid_b):
                    continue
                lat_a, lng_a = self._overpass.way_center_from_geometry(
                    ways_by_id[way_id_a].get("geometry") or [], ref_lat, ref_lng
                )
                lat_b, lng_b = self._overpass.way_center_from_geometry(
                    ways_by_id[way_id_b].get("geometry") or [], ref_lat, ref_lng
                )
                dist = max(
                    self._google.straight_line_distance_meters(lat_a, lng_a, lat_b, lng_b, ref_lat, ref_lng),
                    1e-6,
                )
                if not G.has_edge(nid_a, nid_b):
                    G.add_edge(nid_a, nid_b, distance=dist, slope_pct=default_slope_pct)
                if not G.has_edge(nid_b, nid_a):
                    G.add_edge(nid_b, nid_a, distance=dist, slope_pct=default_slope_pct)

    def _build_way_graph_from_ways(
        self,
        ways_by_id: Dict[int, Dict[str, Any]],
        node_to_way_ids: Dict[int, Set[int]],
        ref_lat: float,
        ref_lng: float,
        default_slope_pct: float = 0.0,
    ) -> nx.DiGraph:
        """Monta DiGraph a partir de ways e mapa nó OSM -> ways."""
        G = nx.DiGraph()
        for key, value in _default_graph_metadata().items():
            G.graph[key] = value

        way_id_to_node_id: Dict[int, str] = {}
        used_slugs: Set[str] = set()

        for way_id, way in ways_by_id.items():
            node_id, label = _way_to_node_id_and_label(way, way_id, used_slugs)
            way_id_to_node_id[way_id] = node_id
            lat, lng = self._overpass.way_center_from_geometry(
                way.get("geometry") or [], ref_lat, ref_lng
            )
            x, y = self._google.lat_lng_to_xy_meters(lat, lng, ref_lat, ref_lng)
            G.add_node(node_id, pos=(x, y), label=label, lat=lat, lng=lng, region="op")

        for _node_osm_id, way_ids_at_node in node_to_way_ids.items():
            self._add_edges_at_intersection(
                G, way_ids_at_node, ways_by_id, way_id_to_node_id,
                ref_lat, ref_lng, default_slope_pct,
            )
        return G

    def _get_way_neighbors_cached(
        self,
        way_id: int,
        south: float, west: float, north: float, east: float,
        way_neighbors_cache: Dict[Tuple[float, float, float, float, int], List[Dict[str, Any]]],
        cache_file: Path,
        use_cache: bool,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Retorna vizinhos da way (da API ou cache). Segundo valor: True se veio do cache."""
        key = _way_neighbors_cache_key(south, west, north, east, way_id)
        if use_cache and key in way_neighbors_cache:
            return (way_neighbors_cache[key], True)
        result = self._overpass.ways_connected_to_way_ids([way_id], south, west, north, east)
        if use_cache:
            way_neighbors_cache[key] = result
            _persist_way_neighbors_cache(self._cache_dir, cache_file, way_neighbors_cache)
        return (result, False)

    def _collect_connected_ways_from_frontier(
        self,
        frontier: List[int],
        current_level: int,
        south: float, west: float, north: float, east: float,
        way_neighbors_cache: Dict[Tuple[float, float, float, float, int], List[Dict[str, Any]]],
        way_neighbors_cache_file: Path,
        use_cache: bool,
    ) -> List[Dict[str, Any]]:
        """Obtém todas as ways conectadas à fronteira atual (uma chamada por way na fronteira)."""
        connected: List[Dict[str, Any]] = []
        seen_connected_ids: Set[int] = set()
        previous_was_from_cache = True
        for i, way_id in enumerate(frontier):
            if i > 0 and not previous_was_from_cache:
                time.sleep(_OSM_BATCH_DELAY_SEC)
            batch_result, from_cache = self._get_way_neighbors_cached(
                way_id, south, west, north, east,
                way_neighbors_cache, way_neighbors_cache_file, use_cache,
            )
            previous_was_from_cache = from_cache
            for w in batch_result:
                if w["id"] not in seen_connected_ids:
                    seen_connected_ids.add(w["id"])
                    connected.append(w)
            processed = i + 1
            if processed % 10 == 0 or processed == len(frontier):
                print(f"build_op_graph: nivel {current_level} — {processed}/{len(frontier)} ways da fronteira processadas")
        return connected

    def _merge_connected_ways_into_state(
        self,
        connected: List[Dict[str, Any]],
        seen_way_ids: Set[int],
        ways_by_id: Dict[int, Dict[str, Any]],
        node_to_way_ids: Dict[int, Set[int]],
    ) -> List[int]:
        """Incorporar ways conectadas ao estado e retornar a próxima fronteira (ids novos)."""
        next_frontier: List[int] = []
        for way_data in connected:
            way_id = way_data["id"]
            if way_id not in seen_way_ids:
                seen_way_ids.add(way_id)
                ways_by_id[way_id] = way_data
                next_frontier.append(way_id)
                for node_osm_id in way_data.get("nodes") or []:
                    node_to_way_ids.setdefault(node_osm_id, set()).add(way_id)
        return next_frontier

    def _expand_op_graph_by_levels(
        self,
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
        """Expansão iterativa (BFS) por níveis: obtém vizinhos de cada way da fronteira e repete até max_level."""
        south, west, north, east = bbox
        while current_level < max_level and frontier:
            print(f"build_op_graph: iniciando nivel {current_level}/{max_level} — fronteira={len(frontier)} ways")
            connected = self._collect_connected_ways_from_frontier(
                frontier, current_level, south, west, north, east,
                way_neighbors_cache, way_neighbors_cache_file, use_cache,
            )
            next_frontier = self._merge_connected_ways_into_state(
                connected, seen_way_ids, ways_by_id, node_to_way_ids,
            )
            print(f"build_op_graph: nivel {current_level}/{max_level} — fronteira={len(frontier)} ways, vizinhos={len(connected)}, novos={len(next_frontier)}")
            frontier = next_frontier
            current_level += 1

    def _try_load_cached_op_graph(
        self, use_cache: bool, force_rebuild: bool
    ) -> Optional[nx.DiGraph]:
        """Carrega grafo do cache se use_cache, não force_rebuild e arquivo existir. Caso contrário retorna None."""
        if not (use_cache and not force_rebuild and self.cache_op_graph.is_file()):
            return None
        try:
            with open(self.cache_op_graph, "rb") as f:
                G = pickle.load(f)
            print(f"build_op_graph: carregado do cache ({G.number_of_nodes()} ruas, {G.number_of_edges()} arestas)")
            return G
        except Exception:
            return None

    def _find_seed_ways_ouro_preto(self) -> List[Dict[str, Any]]:
        """Busca ways iniciais por nome na bbox de Ouro Preto. Levanta ValueError se nenhuma for encontrada."""
        south, west, north, east = OURO_PRETO_BBOX
        candidate_names = ["Rua Rio Piracicaba", "Rio Piracicaba"]
        for name in candidate_names:
            seed_ways = self._overpass.ways_by_name(name, south, west, north, east)
            if seed_ways:
                return seed_ways
        raise ValueError("Nenhuma way 'Rua Rio Piracicaba' (ou 'Rio Piracicaba') encontrada na bbox de Ouro Preto.")

    def _load_way_neighbors_cache(self, use_cache: bool) -> Dict[Tuple[float, float, float, float, int], List[Dict[str, Any]]]:
        """Carrega cache de vizinhos de ways do disco, ou retorna dict vazio."""
        if not use_cache or not self.cache_osm_way_neighbors.is_file():
            return {}
        try:
            with open(self.cache_osm_way_neighbors, "rb") as f:
                cache = pickle.load(f)
            print(f"build_op_graph: cache de vizinhos carregado ({len(cache)} ways em cache)")
            return cache
        except Exception:
            return {}

    def _save_op_graph_to_cache(self, G: nx.DiGraph) -> None:
        """Persiste grafo no cache (mkdir + pickle). Não propaga exceção de I/O."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_op_graph, "wb") as f:
                pickle.dump(G, f)
            print(f"Grafo OP (OSM) salvo em {self.cache_op_graph}")
        except Exception:
            pass

    def build_op_graph(
        self,
        levels: int = 4,
        ref_lat: float = OURO_PRETO_REF_LAT,
        ref_lng: float = OURO_PRETO_REF_LNG,
        default_slope_pct: float = 0.0,
        use_cache: bool = True,
        force_rebuild: bool = False,
    ) -> nx.DiGraph:
        """Constrói grafo das ruas de Ouro Preto (sede) via OpenStreetMap (Overpass API)."""
        cached = self._try_load_cached_op_graph(use_cache, force_rebuild)
        if cached is not None:
            return cached

        seed_ways = self._find_seed_ways_ouro_preto()
        seen_way_ids, ways_by_id, node_to_way_ids = _init_ways_state_from_seeds(seed_ways)
        way_neighbors_cache = self._load_way_neighbors_cache(use_cache)

        south, west, north, east = OURO_PRETO_BBOX
        self._expand_op_graph_by_levels(
            (south, west, north, east),
            list(seen_way_ids),
            seen_way_ids,
            ways_by_id,
            node_to_way_ids,
            current_level=1,
            max_level=levels,
            way_neighbors_cache=way_neighbors_cache,
            way_neighbors_cache_file=self.cache_osm_way_neighbors,
            use_cache=use_cache,
        )

        node_to_way_ids = {n: s for n, s in node_to_way_ids.items() if len(s) >= 2}
        G = self._build_way_graph_from_ways(
            ways_by_id, node_to_way_ids, ref_lat, ref_lng, default_slope_pct=default_slope_pct
        )
        print(f"build_op_graph: {G.number_of_nodes()} ruas, {G.number_of_edges()} arestas (levels={levels})")
        if use_cache:
            self._save_op_graph_to_cache(G)
        return G
