"""Directed road network graph and candidate-to-candidate routing."""

import heapq
from typing import Dict, List, Optional, Tuple, Union

from pathsnap.decoder.models import RoadArc, RoadNode


class RoadNetworkGraph:
    """Directed graph representation of a road network for map matching.

    Preserves original node and arc IDs. Supports directed routing, intermediate
    arc recovery, candidate-to-candidate distance calculations along directed arcs,
    and LRU caching of shortest path queries.
    """

    def __init__(
        self,
        nodes: List[Union[RoadNode, Dict]],
        arcs: List[Union[RoadArc, Dict]],
        max_search_distance_m: float = 2000.0,
        same_arc_reverse_tolerance_m: float = 15.0,
    ) -> None:
        self.max_search_distance_m = max_search_distance_m
        self.same_arc_reverse_tolerance_m = same_arc_reverse_tolerance_m

        # Store node and arc lookups
        self.nodes: Dict[int, RoadNode] = {}
        for n in nodes:
            node_obj = n if isinstance(n, RoadNode) else RoadNode.from_dict(n)
            self.nodes[node_obj.node_id] = node_obj

        self.arcs: Dict[int, RoadArc] = {}
        # Adjacency list: from_node_id -> list of (to_node_id, arc_id, length_m)
        self.adj: Dict[int, List[Tuple[int, int, float]]] = {}
        for a in arcs:
            arc_obj = a if isinstance(a, RoadArc) else RoadArc.from_dict(a)
            self.arcs[arc_obj.arc_id] = arc_obj
            if arc_obj.from_node_id not in self.adj:
                self.adj[arc_obj.from_node_id] = []
            self.adj[arc_obj.from_node_id].append(
                (arc_obj.to_node_id, arc_obj.arc_id, arc_obj.length_m)
            )

        # Cache for shortest path between nodes: (from_node, to_node) -> (distance, [arc_ids])
        self._path_cache: Dict[Tuple[int, int], Tuple[Optional[float], List[int]]] = {}

    def get_candidate_distance_and_path(
        self,
        from_arc_id: int,
        from_frac: float,
        to_arc_id: int,
        to_frac: float,
    ) -> Tuple[Optional[float], List[int]]:
        """Compute the shortest directed network distance and intermediate arc sequence.

        Args:
            from_arc_id: Arc ID of the starting candidate.
            from_frac: Projection fraction along the starting arc in [0, 1].
            to_arc_id: Arc ID of the target candidate.
            to_frac: Projection fraction along the target arc in [0, 1].

        Returns:
            Tuple of (network_distance_m, intermediate_arc_ids).
            If no feasible directed connection exists within max_search_distance_m,
            returns (None, []).
        """
        if from_arc_id not in self.arcs or to_arc_id not in self.arcs:
            return None, []

        from_arc = self.arcs[from_arc_id]
        to_arc = self.arcs[to_arc_id]

        from_frac = max(0.0, min(1.0, from_frac))
        to_frac = max(0.0, min(1.0, to_frac))

        # Case 1: Same directed arc
        if from_arc_id == to_arc_id:
            frac_diff = to_frac - from_frac
            backward_distance = -frac_diff * from_arc.length_m
            if frac_diff < 0 and backward_distance <= self.same_arc_reverse_tolerance_m:
                return 0.0, []
            # Forward movement or within small tolerance
            if frac_diff >= -1e-6:
                dist = max(0.0, frac_diff) * from_arc.length_m
                return dist, []
            else:
                # Backward movement along the same directed segment is not allowed directly.
                # Check if there is a cycle in the road network returning to this arc.
                cycle_dist, intermediate_arcs = self._find_shortest_node_path(
                    from_arc.to_node_id, to_arc.from_node_id
                )
                if cycle_dist is None:
                    return None, []
                # Total distance: remaining on from_arc + cycle + traversed on to_arc
                total_dist = (
                    (1.0 - from_frac) * from_arc.length_m
                    + cycle_dist
                    + to_frac * to_arc.length_m
                )
                if total_dist > self.max_search_distance_m:
                    return None, []
                return total_dist, intermediate_arcs

        # Case 2: Different arcs
        # Immediate direct connection: from_arc.to_node_id == to_arc.from_node_id
        if from_arc.to_node_id == to_arc.from_node_id:
            total_dist = (1.0 - from_frac) * from_arc.length_m + to_frac * to_arc.length_m
            return total_dist, []

        # General case: shortest path from from_arc.to_node_id to to_arc.from_node_id
        node_dist, intermediate_arcs = self._find_shortest_node_path(
            from_arc.to_node_id, to_arc.from_node_id
        )
        if node_dist is None:
            return None, []

        total_dist = (
            (1.0 - from_frac) * from_arc.length_m
            + node_dist
            + to_frac * to_arc.length_m
        )
        if total_dist > self.max_search_distance_m:
            return None, []

        return total_dist, intermediate_arcs

    def _find_shortest_node_path(
        self, start_node: int, goal_node: int
    ) -> Tuple[Optional[float], List[int]]:
        """Dijkstra shortest path between two nodes using directed arcs.

        Returns:
            Tuple of (total_path_length_m, list_of_intermediate_arc_ids).
            If unreachable or exceeding max_search_distance_m, returns (None, []).
        """
        if start_node == goal_node:
            return 0.0, []

        cache_key = (start_node, goal_node)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # Priority queue entries: (cumulative_distance, current_node, arc_path)
        pq: List[Tuple[float, int, List[int]]] = [(0.0, start_node, [])]
        best_dist: Dict[int, float] = {start_node: 0.0}

        while pq:
            d, u, path = heapq.heappop(pq)

            if d > best_dist.get(u, float("inf")):
                continue

            if u == goal_node:
                result = (d, path)
                self._path_cache[cache_key] = result
                return result

            if d >= self.max_search_distance_m:
                continue

            for v, arc_id, length in self.adj.get(u, []):
                new_d = d + length
                if new_d < best_dist.get(v, float("inf")) and new_d <= self.max_search_distance_m:
                    best_dist[v] = new_d
                    heapq.heappush(pq, (new_d, v, path + [arc_id]))

        result = (None, [])
        self._path_cache[cache_key] = result
        return result
