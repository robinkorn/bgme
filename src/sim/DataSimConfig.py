from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DataSimConfig:
    # normal data generation mode
    num_modes: int = 10 # will be doubled for transition modes
    num_dimensions: int = 3
    # seed
    random_seed: int = 42
    # distribution of clusters
    means_list: list[tuple[float, float]] | None = None  # if None, use mean_range for all dimensions
    vars_list: list[tuple[float, float]] | None = None  # if None, use var_range for all dimensions
    mean_range: tuple[float, float] | None = (0, 20)
    var_range: tuple[float, float] | None = (0.25, 1.25)
    mode_weights: list[float] = None  # if None, uniform weights are used, also includes transition modes
    # iteration and sampling
    samples_per_iteration: int = 1000
    iterations_during_training: int = 10  # number of iterations through all modes in one complete simulation
    iterations_for_generator: int = 10
    # noise parameters
    noise_student_t_df: float | None = 5.0  # None to disable, low rare but big spikes, high frequent small noise
    noise_student_t_scale: float = 0.05 # low normal values, high chaos
    quant_step: float | None = 0.001 # None to disable
    clip: tuple[float, float] | None = mean_range  # (0.0, 1.0) for clipping between 0 and 1

    def __post_init__(self):
        # TODO: add more checks for future variables?
        if self.num_modes <= 0:
            raise ValueError("num_modes must be positive")
        if self.num_dimensions <= 0:
            raise ValueError("num_dimensions must be positive")
        if self.mean_range is not None and self.var_range is not None:
            if self.mean_range[0] >= self.mean_range[1]:
                raise ValueError("mean_range min must be less than max")
            if self.var_range[0] >= self.var_range[1]:
                raise ValueError("var_range min must be less than max")
        if self.mode_weights is not None:
            if sum(self.mode_weights) != 1:
                raise ValueError(
                    "Sum of weights in mode_weights must be 1")
        both_lists_given = (self.means_list is not None and
                            self.vars_list is not None)
        both_lists_none = (self.means_list is None and
                        self.vars_list is None)

        if not (both_lists_given or both_lists_none):
            raise ValueError(
                "means_list and vars_list must both be given together "
                "or both be None (cannot supply only one of them)."
            )

        if both_lists_given:
            if self.mean_range is not None or self.var_range is not None:
                raise ValueError(
                    "When means_list and vars_list are provided, "
                    "mean_range and var_range must be None."
                )
            
        if not self.samples_per_iteration > 0:
            raise ValueError("samples_per_iteration must be greater than 0")
        if not self.iterations_during_training > 0:
            raise ValueError("iterations must be greater than 0")
        if not self.iterations_for_generator > 0:
            raise ValueError("iterations_for_generator must be greater than 0")
