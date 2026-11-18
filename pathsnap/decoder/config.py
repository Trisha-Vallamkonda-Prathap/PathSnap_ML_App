"""Configuration for the HMM/Viterbi route decoder and evaluation."""

from dataclasses import dataclass


@dataclass
class DecoderConfig:
    """Configurable parameters for the HMM Viterbi decoder.

    Attributes:
        gps_gap_threshold_s: Threshold in seconds beyond which consecutive GPS
            observations are treated as a disconnected gap (break_before = 'gps_gap').
        emission_floor: Minimum positive probability floor applied to candidate
            emissions before taking natural logarithms to prevent log(0).
        transition_beta: Distance difference scale parameter in metres for the
            exponential transition probability model: exp(-|d_net - d_gps| / beta).
        max_speed_mps: Maximum plausible velocity in metres/second (~162 km/h)
            above which a network transition is considered physically impossible.
        max_network_search_m: Maximum network distance in metres searched between
            consecutive candidate road segments before giving up and declaring unreachable.
        use_equal_transition_weights: If True, feasible graph transitions are assigned
            equal log-probability (0.0), matching the test fixture requirements.
        same_arc_reverse_tolerance_m: Backward projection tolerance in metres treated as
            stationary GPS jitter on the same directed arc.
    """

    gps_gap_threshold_s: float = 30.0
    emission_floor: float = 1e-6
    transition_beta: float = 10.0
    max_speed_mps: float = 45.0
    max_network_search_m: float = 2000.0
    use_equal_transition_weights: bool = False
    same_arc_reverse_tolerance_m: float = 15.0
