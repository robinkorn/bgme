from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BGMEConfig:
    # Ensemble size
    n_estimators: int = 5

    # GMM structure
    n_components: int = 10
    covariance_type: str = "diag"  # 'full', 'tied', 'diag', or 'spherical'
    reg_covar: float = 1e-4
    max_iter: int = 200
    tol: float = 1e-3
    random_state: int | None = 42

    # "Boosting-like" sample reweighting
    learning_rate: float = 0.7  # eta in exp(eta * error)
    hard_quantile: float = 0.90  # top-q hardest (lowest ll) get upweighted
    min_weight: float = 1e-6
    max_weight: float = 1e3

    # How to combine estimators at inference
    combine: str = "mean_loglik"  # "mean_loglik", "logmeanexp"

    # Online ensemble maintenance
    # If None: keep legacy behavior (re-balance to equal alphas after adding a model).
    # If set to a value in (0, 1]: multiply existing alphas by this factor before
    # adding the new model, then renormalize. Smaller => faster forgetting.
    alpha_decay_rate: float | None = 0.95

    # Maximum number of models to keep in the ensemble. If None: unlimited.
    # When exceeded, the lowest-alpha models are removed.
    max_models_per_ensemble: int | None = 30
