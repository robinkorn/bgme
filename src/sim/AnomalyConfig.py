from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class AnomalyConfig:
    modes: list[str] = field(default_factory=
                             lambda: ["mean_shift", "var_shift", "new_mode"])
    # anomaly rate
    stream_anomaly_prob: float = 0.01

    # using existing template rate
    stream_use_seen_template_prob: float = 0.5

    # max templates
    max_stream_anomaly_templates: int = 32

    # probalities for each anomaly type when a new anomaly is created
    mean_shift_prob: float = 0.4
    var_shift_prob: float = 0.4
    new_mode_prob: float = 0.2

    # scales for anomalies
    mean_shift_scale_min: float = 0.5
    mean_shift_scale_max: float = 2.0

    var_scale_min: float = 1.1
    var_scale_max: float = 3.0


    def __post_init__(self):
        if not (0.0 <= self.stream_anomaly_prob <= 1.0):
            raise ValueError("stream_anomaly_prob must be between 0 and 1")

        if not (0.0 <= self.stream_use_seen_template_prob <= 1.0):
            raise ValueError(
                "stream_use_seen_template_prob must be between 0 and 1"
            )

        if self.max_stream_anomaly_templates <= 0:
            raise ValueError("max_stream_anomaly_templates must be positive")

        for name in ["mean_shift_prob", "var_shift_prob", "new_mode_prob"]:
            val = getattr(self, name)
            if val < 0.0:
                raise ValueError(f"{name} must be non-negative")

        if (
            self.mean_shift_scale_min <= 0.0
            or self.mean_shift_scale_max < self.mean_shift_scale_min
        ):
            raise ValueError(
                "mean_shift_scale_min/max must be positive and min <= max"
            )

        if (
            self.var_scale_min <= 0.0
            or self.var_scale_max < self.var_scale_min
        ):
            raise ValueError(
                "var_scale_min/max must be positive and min <= max"
            )