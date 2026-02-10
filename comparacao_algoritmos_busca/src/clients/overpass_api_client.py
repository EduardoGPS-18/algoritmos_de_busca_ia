"""
Cliente para integração com OpenStreetMap Overpass API (conectividade de ruas, ways).
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from .google_api_client import OURO_PRETO_REF_LAT, OURO_PRETO_REF_LNG

OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"

_OVERPASS_RETRY_DELAY_SEC = 10
_OVERPASS_RATE_LIMIT_DELAY_SEC = 30
_OVERPASS_MAX_RETRIES = 5
_OSM_CONNECTED_WAYS_BATCH_SIZE = 40
_OSM_BATCH_DELAY_SEC = 1.5


class OverpassApiClient:
    """Cliente responsável pelas integrações com OpenStreetMap (Overpass API)."""

    def __init__(self, url: str = OVERPASS_API_URL, cache_dir: Optional[Path] = None):
        self._url = url
        self._cache_dir = cache_dir

    def post(
        self,
        query: str,
        timeout: int = 90,
    ) -> Optional[Dict[str, Any]]:
        """
        Envia query à Overpass API e retorna o JSON ou None em caso de erro.
        429/504: espera 30s sem contar retry; outros erros: 10s até 5 retentativas.
        """
        last_error: Optional[Exception] = None
        retry_count = 0
        while retry_count <= _OVERPASS_MAX_RETRIES:
            try:
                r = requests.post(self._url, data={"data": query}, timeout=timeout + 10)
                r.raise_for_status()
                data = r.json()
                if data is not None:
                    return data
            except requests.exceptions.HTTPError as e:
                last_error = e
                status = e.response.status_code if e.response is not None else None
                if status in (429, 504):
                    print(
                        f"  Overpass: {status} (rate limit/timeout), aguardando {_OVERPASS_RATE_LIMIT_DELAY_SEC}s (não contabiliza retentativa)..."
                    )
                    time.sleep(_OVERPASS_RATE_LIMIT_DELAY_SEC)
                    continue
                print(f"  Overpass: erro na consulta — {type(e).__name__}: {e}")
            except Exception as e:
                last_error = e
                print(f"  Overpass: erro na consulta — {type(e).__name__}: {e}")
            retry_count += 1
            if retry_count <= _OVERPASS_MAX_RETRIES:
                print(
                    f"  Overpass: tentativa {retry_count}/{_OVERPASS_MAX_RETRIES + 1}, aguardando {_OVERPASS_RETRY_DELAY_SEC}s para retentar..."
                )
                time.sleep(_OVERPASS_RETRY_DELAY_SEC)
            else:
                if last_error is not None:
                    print(
                        f"  Overpass: ignorando após {_OVERPASS_MAX_RETRIES} retentativas (último erro: {last_error}), seguindo para o próximo."
                    )
                else:
                    print(
                        f"  Overpass: ignorando após {_OVERPASS_MAX_RETRIES} retentativas, seguindo para o próximo."
                    )
        return None

    def query_connected_street_names(
        self,
        street_name: str,
        south: float,
        west: float,
        north: float,
        east: float,
        timeout: int = 25,
    ) -> Set[str]:
        """
        Consulta Overpass: ruas (ways highway) que compartilham nós com a rua de nome street_name na bbox.
        Retorna conjunto de nomes de ruas conectadas (inclui a própria rua).
        """
        if not street_name or not street_name.strip():
            return set()
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
            data = self.post(query, timeout=timeout + 5)
            if not data:
                return set()
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
        self,
        places: List[Dict[str, Any]],
        bbox: Tuple[float, float, float, float],
        use_cache: bool = True,
        cache_dir: Optional[Path] = None,
    ) -> Set[Tuple[str, str]]:
        """
        Para cada lugar, obtém o nome da rua (do address), consulta Overpass para ruas que tocam essa rua,
        e monta o conjunto de pares (place_id_origem, place_id_destino). Retorna set de (id_a, id_b).
        """
        south, west, north, east = bbox
        cache_d = cache_dir or self._cache_dir
        cache_file = (cache_d / "osm_connectivity_cache.pickle") if cache_d else None

        place_id_to_street: Dict[str, str] = {}
        street_to_place_ids: Dict[str, List[str]] = {}
        for p in places:
            pid = p.get("id", "")
            addr = p.get("address", "")
            street = self.get_street_name_from_address(addr)
            place_id_to_street[pid] = street
            street_to_place_ids.setdefault(street, []).append(pid)

        cache_osm: Dict[str, Set[str]] = {}
        if use_cache and cache_file and cache_file.is_file():
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
                cache_osm[street] = self.query_connected_street_names(
                    street, south, west, north, east
                )
            connected_names = cache_osm[street]
            for other_street in connected_names:
                for other_id in street_to_place_ids.get(other_street, []):
                    if other_id != pid:
                        connected_pairs.add((pid, other_id))

        if use_cache and cache_osm and cache_d and cache_file:
            cache_d.mkdir(parents=True, exist_ok=True)
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(cache_osm, f)
            except Exception:
                pass
        return connected_pairs

    def ways_by_name(
        self,
        street_name: str,
        south: float,
        west: float,
        north: float,
        east: float,
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
        data = self.post(query)
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

    def ways_connected_to_way_ids(
        self,
        way_ids: List[int],
        south: float,
        west: float,
        north: float,
        east: float,
    ) -> List[Dict[str, Any]]:
        """
        Retorna ways (highway) que compartilham pelo menos um nó com alguma das way_ids, na bbox.
        Faz requisições em lotes com pausa entre elas.
        """
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
            data = self.post(query)
            if not data or "elements" not in data:
                if n_batches > 1:
                    print(
                        f"  Overpass: lote {i // _OSM_CONNECTED_WAYS_BATCH_SIZE + 1}/{n_batches} sem dados (timeout/erro)"
                    )
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

    @staticmethod
    def get_street_name_from_address(address: str) -> str:
        """Extrai o nome da rua do endereço (parte antes da primeira vírgula)."""
        return address.split(",")[0].strip() if address else ""

    @staticmethod
    def way_center_from_geometry(
        geometry: List[Dict[str, float]],
        ref_lat: float = OURO_PRETO_REF_LAT,
        ref_lng: float = OURO_PRETO_REF_LNG,
    ) -> Tuple[float, float]:
        """Calcula (lat, lng) do centro de uma way a partir da lista geometry (Overpass)."""
        if not geometry:
            return (ref_lat, ref_lng)
        lat_sum = sum(p.get("lat", 0) for p in geometry)
        lon_sum = sum(p.get("lon", 0) for p in geometry)
        n = len(geometry)
        return (lat_sum / n, lon_sum / n)
