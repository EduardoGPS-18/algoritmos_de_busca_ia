"""
Cliente para integrações com Google Cloud (Geocoding, Directions, Elevation).
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import requests


# Centro aproximado de Ouro Preto (ref para converter lat/lng em metros)
OURO_PRETO_REF_LAT = -20.3855
OURO_PRETO_REF_LNG = -43.5034


class GoogleApiClient:
    """Cliente responsável pelas integrações com Google Cloud (Geocoding, Directions, Elevation)."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")

    def _ensure_key(self) -> str:
        if not self._api_key:
            raise ValueError(
                "API key do Google Maps não informada. Passe api_key ou defina GOOGLE_MAPS_API_KEY no ambiente."
            )
        return self._api_key

    def geocode(self, address: str) -> Optional[Tuple[float, float]]:
        """Obtém (lat, lng) de um endereço via Google Geocoding API."""
        result = self.geocode_with_components(address)
        return (result[0], result[1]) if result else None

    def geocode_with_components(
        self, address: str
    ) -> Optional[Tuple[float, float, str, str]]:
        """
        Obtém (lat, lng, bairro, estado) via Google Geocoding API.
        bairro = long_name de sublocality_level_1/sublocality/neighborhood.
        estado = short_name de administrative_area_level_1 (2 caracteres, ex: MG).
        Retorna None se falhar; bairro/estado vazios se não encontrados.
        """
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": address, "key": self._ensure_key()}
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
                if not bairro and any(
                    t in types for t in ("sublocality_level_1", "sublocality", "neighborhood")
                ):
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

    def get_elevations_batch(
        self,
        locations: List[Tuple[float, float]],
    ) -> List[Optional[float]]:
        """
        Obtém elevação (m) para uma lista de pontos via Google Elevation API.
        Retorna lista na mesma ordem de locations; None onde falhou.
        Máximo 512 pontos por request (limite da API).
        """
        if not locations:
            return []
        loc_str = "|".join(f"{lat},{lng}" for lat, lng in locations)
        url = "https://maps.googleapis.com/maps/api/elevation/json"
        params = {"locations": loc_str, "key": self._ensure_key()}
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            if data.get("status") != "OK" or "results" not in data:
                print(f"Erro ao obter elevação: {data.get('status')}")
                return [None] * len(locations)
            results = data["results"]
            out = []

            print(f"sucesso ao obter elevação: {len(results)} results")
            for i, res in enumerate(results):
                if i < len(locations) and "elevation" in res:
                    out.append(float(res["elevation"]))
                else:
                    out.append(None)
            while len(out) < len(locations):
                out.append(None)
            return out[: len(locations)]
        except Exception as e:
            print(f"Erro ao obter elevação: {e}")
            return [None] * len(locations)

    def get_route_info(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
    ) -> Optional[Tuple[float, int]]:
        """
        Obtém (distância em metros, número de steps) da rota entre origem e destino (Google Directions API).
        Retorna None se não houver rota.
        """
        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            "origin": f"{origin[0]},{origin[1]}",
            "destination": f"{destination[0]},{destination[1]}",
            "key": self._ensure_key(),
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

    def get_route_distance_meters(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
    ) -> Optional[float]:
        """Obtém a distância em metros da rota entre origem e destino (Google Directions API)."""
        info = self.get_route_info(origin, destination)
        return info[0] if info else None

    @staticmethod
    def lat_lng_to_xy_meters(
        lat: float, lng: float, ref_lat: float, ref_lng: float
    ) -> Tuple[float, float]:
        """Converte (lat, lng) em (x, y) em metros relativos a ref."""
        y = (lat - ref_lat) * 110540.0
        x = (lng - ref_lng) * 111320.0 * math.cos(math.radians(ref_lat))
        return (x, y)

    @staticmethod
    def straight_line_distance_meters(
        lat1: float, lng1: float, lat2: float, lng2: float,
        ref_lat: float, ref_lng: float,
    ) -> float:
        """Distância em linha reta entre dois pontos (lat,lng) em metros."""
        x1, y1 = GoogleApiClient.lat_lng_to_xy_meters(lat1, lng1, ref_lat, ref_lng)
        x2, y2 = GoogleApiClient.lat_lng_to_xy_meters(lat2, lng2, ref_lat, ref_lng)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @staticmethod
    def slope_pct_from_elevation(
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
