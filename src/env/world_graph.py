from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Literal

from src.env.pricing import compute_edge_fare_bounds, compute_edge_physical_deltas
from src.env.schema import (
    ActivitySpecConfig,
    EdgeRouteConfig,
    LocationNodeConfig,
    MobilityAndClimateConfig,
    WorldConfig,
)

RoutePreference = Literal["fastest", "cheapest"]


@dataclass(frozen=True)
class PlannedPath:
    """Hasil routing graf antara dua lokasi beserta total jarak, tarif, dan beban fisik."""

    source_node_id: str
    target_node_id: str
    edges: tuple[EdgeRouteConfig, ...]
    total_distance_km: float
    min_fare_idr: float
    max_fare_idr: float
    total_energy_cost: float
    total_mood_delta: float


class WorldGraph:
    """
    Graf spasial lokasi & edge transport. Menggantikan seluruh logika rute dan
    lokasi hardcoded dengan traversal Dijkstra berbasis biaya & kelelahan fisik.
    """

    def __init__(self, world_config: WorldConfig):
        self.world_config = world_config
        self.graph_cfg = world_config.world_graph
        self.mobility_cfg: MobilityAndClimateConfig = world_config.mobility_and_climate
        self.home_node_id: str = self.graph_cfg.home_node_id

        self.nodes: dict[str, LocationNodeConfig] = {
            node.id: node for node in self.graph_cfg.locations
        }
        self.node_ids: tuple[str, ...] = tuple(node.id for node in self.graph_cfg.locations)
        self.adjacency: dict[str, list[tuple[str, EdgeRouteConfig]]] = {nid: [] for nid in self.node_ids}

        for edge in self.graph_cfg.edges:
            self.adjacency[edge.source].append((edge.target, edge))
            if edge.bidirectional and edge.source != edge.target:
                self.adjacency[edge.target].append((edge.source, edge))

        self.validate_connectivity()

    def get_node(self, node_id: str) -> LocationNodeConfig:
        return self.nodes[node_id]

    def get_sleep_nodes(self) -> list[LocationNodeConfig]:
        return [n for n in self.nodes.values() if n.can_sleep]

    def get_class_nodes(self) -> list[LocationNodeConfig]:
        return [n for n in self.nodes.values() if n.is_class_venue]

    def resolve_activity_target_nodes(self, activity: ActivitySpecConfig) -> list[str]:
        """Menentukan node mana saja yang valid untuk menjalankan sebuah ActivitySpec."""
        if activity.target_nodes:
            return list(activity.target_nodes)

        handler = activity.handler
        if handler == "rest":
            return [n.id for n in self.nodes.values() if n.can_sleep]
        if handler == "attend_schedule":
            return [n.id for n in self.nodes.values() if n.is_class_venue]
        if handler == "grocery_shop":
            return [n.id for n in self.nodes.values() if n.has_grocery]
        if handler == "night_cram":
            return [n.id for n in self.nodes.values() if n.can_sleep and n.can_study]
        if handler == "study":
            return [n.id for n in self.nodes.values() if n.can_study]
        if handler == "hub_stay":
            return [
                n.id
                for n in self.nodes.values()
                if n.can_study and n.study_profile.craving_relief > 0.0 and not n.study_profile.craving_reset
            ]
        if handler == "leisure_reset":
            return [
                n.id
                for n in self.nodes.values()
                if n.zone == "third_place" or n.study_profile.craving_reset
            ]
        if handler == "wait":
            return [n.id for n in self.nodes.values() if not n.can_sleep and n.closes_at_night]
        return list(self.node_ids)

    def _edge_weight(
        self,
        edge: EdgeRouteConfig,
        preference: RoutePreference,
        climate: str,
        ojol_tariff: str,
        is_raining: bool,
    ) -> tuple[float, float, float, float, float]:
        min_fare, max_fare = compute_edge_fare_bounds(edge, ojol_tariff, is_raining)
        energy_delta, mood_delta = compute_edge_physical_deltas(
            edge=edge,
            walk_physics=self.mobility_cfg.walk_physics,
            motorized_physics=self.mobility_cfg.motorized_physics,
            heat_sensitivity=float(self.mobility_cfg.heat_sensitivity),
            climate=climate,
            is_raining=is_raining,
        )
        avg_fare = 0.5 * (min_fare + max_fare)
        fatigue_penalty = (-energy_delta) + max(0.0, -mood_delta) * 1.25

        if preference == "cheapest":
            # Prioritaskan ongkos terendah, tie-break kelelahan terendah
            cost_score = avg_fare * 1.0 + fatigue_penalty * 10.0
        else:
            # "fastest": prioritaskan kelelahan & waktu tempuh terendah, tie-break ongkos
            cost_score = fatigue_penalty * 2000.0 + avg_fare * 0.10
        return cost_score, min_fare, max_fare, -energy_delta, mood_delta

    def find_best_path(
        self,
        source_node_id: str,
        target_node_id: str,
        preference: RoutePreference = "fastest",
        climate: str = "cool",
        ojol_tariff: str = "normal",
        is_raining: bool = False,
        max_budget_idr: float | None = None,
    ) -> PlannedPath | None:
        """Mencari rute terbaik pada graf spasial menggunakan algoritma Dijkstra."""
        if source_node_id == target_node_id:
            return PlannedPath(
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                edges=(),
                total_distance_km=0.0,
                min_fare_idr=0.0,
                max_fare_idr=0.0,
                total_energy_cost=0.0,
                total_mood_delta=0.0,
            )

        heap: list[tuple[float, int, str, tuple[EdgeRouteConfig, ...], float, float, float, float, float]] = [
            (0.0, 0, source_node_id, (), 0.0, 0.0, 0.0, 0.0, 0.0)
        ]
        best_score: dict[str, float] = {}
        counter = 0

        while heap:
            score, _, curr, path_edges, dist, min_f, max_f, e_cost, m_delta = heapq.heappop(heap)
            if curr == target_node_id:
                return PlannedPath(
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    edges=path_edges,
                    total_distance_km=dist,
                    min_fare_idr=min_f,
                    max_fare_idr=max_f,
                    total_energy_cost=e_cost,
                    total_mood_delta=m_delta,
                )

            if curr in best_score and score >= best_score[curr]:
                continue
            best_score[curr] = score

            for neighbor, edge in self.adjacency[curr]:
                edge_score, e_min_f, e_max_f, e_ecost, e_mdelta = self._edge_weight(
                    edge, preference, climate, ojol_tariff, is_raining
                )
                new_min_f = min_f + e_min_f
                if max_budget_idr is not None and new_min_f > max_budget_idr:
                    continue
                counter += 1
                heapq.heappush(
                    heap,
                    (
                        score + edge_score,
                        counter,
                        neighbor,
                        path_edges + (edge,),
                        dist + float(edge.distance_km),
                        new_min_f,
                        max_f + e_max_f,
                        e_cost + e_ecost,
                        m_delta + e_mdelta,
                    ),
                )
        return None

    def choose_nearest_target_node(
        self,
        source_node_id: str,
        candidate_node_ids: list[str],
        preference: RoutePreference = "fastest",
        climate: str = "cool",
        ojol_tariff: str = "normal",
        is_raining: bool = False,
        max_budget_idr: float | None = None,
    ) -> tuple[str | None, PlannedPath | None]:
        """Memilih node tujuan paling optimal di antara kandidat untuk suatu aktivitas."""
        if not candidate_node_ids:
            return None, None
        if source_node_id in candidate_node_ids:
            return source_node_id, self.find_best_path(source_node_id, source_node_id)

        best_node: str | None = None
        best_path: PlannedPath | None = None
        best_key: tuple[float, float] | None = None

        for cid in candidate_node_ids:
            path = self.find_best_path(
                source_node_id=source_node_id,
                target_node_id=cid,
                preference=preference,
                climate=climate,
                ojol_tariff=ojol_tariff,
                is_raining=is_raining,
                max_budget_idr=max_budget_idr,
            )
            if path is None:
                continue
            avg_fare = 0.5 * (path.min_fare_idr + path.max_fare_idr)
            fatigue = path.total_energy_cost + max(0.0, -path.total_mood_delta)
            key = (avg_fare, fatigue) if preference == "cheapest" else (fatigue, avg_fare)
            if best_key is None or key < best_key:
                best_key = key
                best_node = cid
                best_path = path

        return best_node, best_path

    def validate_connectivity(self) -> None:
        """Memastikan setiap node yang bisa dicapai dari home juga punya jalur pulang ke home."""
        sleep_ids = {n.id for n in self.get_sleep_nodes()}
        for nid in self.node_ids:
            if nid in sleep_ids:
                continue
            can_go = self.find_best_path(self.home_node_id, nid) is not None
            can_return = any(self.find_best_path(nid, sid) is not None for sid in sleep_ids)
            if can_go and not can_return:
                raise ValueError(
                    f"Node '{nid}' dapat dikunjungi dari '{self.home_node_id}' tetapi tidak memiliki jalur pulang ke lokasi tidur."
                )
