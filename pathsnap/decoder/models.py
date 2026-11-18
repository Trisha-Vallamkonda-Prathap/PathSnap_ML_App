"""Data models and interfaces for the PathSnap decoder and evaluation module."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union


@dataclass
class Observation:
    """A raw GPS track observation point."""

    obs_idx: int
    lon: float
    lat: float
    time_s: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        return cls(
            obs_idx=int(data["obs_idx"]),
            lon=float(data["lon"]),
            lat=float(data["lat"]),
            time_s=float(data["time_s"]),
        )


@dataclass
class RoadNode:
    """A node in the directed road network graph."""

    node_id: int
    lon: float
    lat: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoadNode":
        return cls(
            node_id=int(data["node_id"]),
            lon=float(data["lon"]),
            lat=float(data["lat"]),
        )


@dataclass
class RoadArc:
    """A directed road segment arc."""

    arc_id: int
    from_node_id: int
    to_node_id: int
    length_m: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoadArc":
        return cls(
            arc_id=int(data["arc_id"]),
            from_node_id=int(data["from_node_id"]),
            to_node_id=int(data["to_node_id"]),
            length_m=float(data["length_m"]),
        )


@dataclass
class Trip:
    """Complete trip structure containing track observations and road network."""

    trip_id: str
    observations: List[Observation]
    nodes: List[RoadNode]
    arcs: List[RoadArc]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trip":
        return cls(
            trip_id=str(data["trip_id"]),
            observations=[
                o if isinstance(o, Observation) else Observation.from_dict(o)
                for o in data.get("observations", [])
            ],
            nodes=[
                n if isinstance(n, RoadNode) else RoadNode.from_dict(n)
                for n in data.get("nodes", [])
            ],
            arcs=[
                a if isinstance(a, RoadArc) else RoadArc.from_dict(a)
                for a in data.get("arcs", [])
            ],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoredCandidate:
    """Candidate road segment associated with an observation, scored by the ML classifier."""

    trip_id: str
    obs_idx: int
    arc_id: int
    along_arc_fraction: float
    distance_m: float
    emission_prob: float
    heading_diff_deg: Optional[float] = None
    speed_mps: Optional[float] = None
    delta_t_s: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoredCandidate":
        return cls(
            trip_id=str(data["trip_id"]),
            obs_idx=int(data["obs_idx"]),
            arc_id=int(data["arc_id"]),
            along_arc_fraction=float(data["along_arc_fraction"]),
            distance_m=float(data["distance_m"]),
            emission_prob=float(data["emission_prob"]),
            heading_diff_deg=(
                float(data["heading_diff_deg"])
                if data.get("heading_diff_deg") is not None
                else None
            ),
            speed_mps=(
                float(data["speed_mps"])
                if data.get("speed_mps") is not None
                else None
            ),
            delta_t_s=(
                float(data["delta_t_s"])
                if data.get("delta_t_s") is not None
                else None
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClassifierMatch:
    """Greedy top-1 baseline prediction from the classifier for one observation."""

    trip_id: str
    obs_idx: int
    arc_id: Optional[int]
    along_arc_fraction: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassifierMatch":
        return cls(
            trip_id=str(data["trip_id"]),
            obs_idx=int(data["obs_idx"]),
            arc_id=int(data["arc_id"]) if data.get("arc_id") is not None else None,
            along_arc_fraction=(float(data["along_arc_fraction"]) if data.get("along_arc_fraction") is not None else None),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


AlignmentStatus = Literal["aligned", "ambiguous", "unaligned"]


@dataclass
class Reference:
    """Ground-truth reference label for an observation (used exclusively in evaluation)."""

    trip_id: str
    obs_idx: int
    ref_arc_id: Optional[int]
    alignment_status: AlignmentStatus

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Reference":
        return cls(
            trip_id=str(data["trip_id"]),
            obs_idx=int(data["obs_idx"]),
            ref_arc_id=int(data["ref_arc_id"]) if data.get("ref_arc_id") is not None else None,
            alignment_status=data["alignment_status"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


UnmatchedReason = Literal["no_candidates", "invalid_transition", None]


@dataclass
class Match:
    """Observation match produced by decoding."""

    obs_idx: int
    arc_id: Optional[int]
    unmatched_reason: Optional[Literal["no_candidates", "invalid_transition"]] = None
    along_arc_fraction: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Match":
        return cls(
            obs_idx=int(data["obs_idx"]),
            arc_id=int(data["arc_id"]) if data.get("arc_id") is not None else None,
            unmatched_reason=data.get("unmatched_reason"),
            along_arc_fraction=(float(data["along_arc_fraction"]) if data.get("along_arc_fraction") is not None else None),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


BreakReason = Literal["gps_gap", "no_candidates", "invalid_transition", None]


@dataclass
class RoutePart:
    """Contiguous directed route segment connecting observation matches and intermediate arcs."""

    start_obs_idx: int
    end_obs_idx: int
    arc_ids: List[int] = field(default_factory=list)
    break_before: Optional[Literal["gps_gap", "no_candidates", "invalid_transition"]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutePart":
        return cls(
            start_obs_idx=int(data["start_obs_idx"]),
            end_obs_idx=int(data["end_obs_idx"]),
            arc_ids=[int(a) for a in data.get("arc_ids", [])],
            break_before=data.get("break_before"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecodedTrip:
    """Complete decoded route reconstruction for a trip."""

    trip_id: str
    matches: List[Match]
    route_parts: List[RoutePart]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecodedTrip":
        return cls(
            trip_id=str(data["trip_id"]),
            matches=[
                m if isinstance(m, Match) else Match.from_dict(m)
                for m in data.get("matches", [])
            ],
            route_parts=[
                rp if isinstance(rp, RoutePart) else RoutePart.from_dict(rp)
                for rp in data.get("route_parts", [])
            ],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trip_id": self.trip_id,
            "matches": [m.to_dict() for m in self.matches],
            "route_parts": [rp.to_dict() for rp in self.route_parts],
        }
