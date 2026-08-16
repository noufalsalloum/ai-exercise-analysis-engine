from __future__ import annotations

try:
    from .base_expert import BaseExpert
except ImportError as exc:
    if __package__:
        raise
    from base_expert import BaseExpert


class PlankExpert(BaseExpert):
    """Isometric-core expert with a compact stability adapter."""

    def __init__(
        self,
        input_dim: int = 512,
        temporal_dim: int = 512,
        global_dim: int = 1024,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            temporal_dim=temporal_dim,
            global_dim=global_dim,
            dropout=dropout,
            adapter_bottleneck_dim=96,
            adapter_gate_init=-0.5,
            exercise_id="plank",
        )
