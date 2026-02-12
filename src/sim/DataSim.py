import numpy as np
from src.sim.DataSimConfig import DataSimConfig
from src.sim.AnomalyConfig import AnomalyConfig
from dataclasses import dataclass

@dataclass(slots=True)
class AnomalyEvent:
    type: str
    iteration: int
    affected_mode: int
    ext_idx: int
    old_mean: np.ndarray
    old_cov: np.ndarray
    new_mean: np.ndarray
    new_cov: np.ndarray
    scale: float

class DataSim:
    def __init__(self, sim_config: DataSimConfig, anomaly_config: AnomalyConfig):
        self.sim_config = sim_config
        self.anomaly_config = anomaly_config
        np.random.seed(self.sim_config.random_seed)

        # store seen anomalies
        self._anomaly_history = {}
        self._stream_anomaly_templates = []

    # Anomaly manipulation
    def _choose_mode_and_anomaly(self, current_components, ext_idx: int, iteration: int):
        cfg = self.anomaly_config

        # only modes can have an anomaly, not transitions
        if ext_idx % 2 != 0:
            return current_components, False

        mode_idx = ext_idx // 2

        # randomly decide whether to trigger an anomaly
        if np.random.rand() >= cfg.stream_anomaly_prob:
            return current_components, False

        templates = self._stream_anomaly_templates
        max_templates = cfg.max_stream_anomaly_templates

        # only look at templates with the same affected_mode
        same_mode_templates = [
            t for t in templates
            if t.get("affected_mode") == mode_idx
        ]

        # choose whether to use existing template or create new one
        prefer_existing = False
        if len(templates) > 0:
            if len(templates) >= max_templates:
                # if cap reached, always prefer existing
                prefer_existing = True
            else:
                prefer_existing = (np.random.rand() <
                                   cfg.stream_use_seen_template_prob)

        # only choose existing if there is at least one for this mode
        use_existing_for_this_mode = prefer_existing and len(
            same_mode_templates) > 0

        # choose template or create new one
        if use_existing_for_this_mode:
            # choose random existing template with matching affected_mode
            template = same_mode_templates[np.random.randint(
                0, len(same_mode_templates))]
            anomaly_type = template["type"]
        else:
            # create new template
            mean_k, cov_k = current_components[mode_idx]
            dim = mean_k.shape[0]

            # choose anomaly type based on configured probabilities
            available_types = cfg.modes
            type_weights = []
            for t in available_types:
                if t == "mean_shift":
                    type_weights.append(cfg.mean_shift_prob)
                elif t == "var_shift":
                    type_weights.append(cfg.var_shift_prob)
                elif t == "new_mode":
                    type_weights.append(cfg.new_mode_prob)
                else:
                    type_weights.append(0.0)

            type_weights = np.asarray(type_weights, dtype=float)
            anomaly_type = np.random.choice(available_types, p=type_weights)

            # random direction
            direction = np.random.randn(dim)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm

            template: dict[str, object] = {
                "type": anomaly_type,
                "affected_mode": mode_idx,
            }

            if anomaly_type == "mean_shift":
                mean_scale = np.random.uniform(
                    cfg.mean_shift_scale_min, cfg.mean_shift_scale_max
                )
                template["mean_direction"] = direction
                template["mean_scale"] = mean_scale
                template["var_scale"] = 1.0

            elif anomaly_type == "var_shift":
                var_scale = np.random.uniform(
                    cfg.var_scale_min, cfg.var_scale_max
                )
                template["mean_direction"] = None
                template["mean_scale"] = 0.0
                template["var_scale"] = var_scale

            elif anomaly_type == "new_mode":
                mean_scale = np.random.uniform(
                    cfg.mean_shift_scale_min, cfg.mean_shift_scale_max
                )
                var_scale = np.random.uniform(
                    cfg.var_scale_min, cfg.var_scale_max
                )
                template["mean_direction"] = direction
                template["mean_scale"] = mean_scale
                template["var_scale"] = var_scale

            else:
                # fallback, should not happen
                return current_components, False

            # save anomalies if limit isnt reached
            if len(templates) < max_templates:
                templates.append(template)
            # NOTE: if limit is reached but no affected_mode matched we still use the template but don't save it

        # use template to modify current_components
        mean_k, cov_k = current_components[mode_idx]
        old_mean = mean_k.copy()
        old_cov = cov_k.copy()

        mean_direction = template.get("mean_direction", None)
        mean_scale = float(template.get("mean_scale", 0.0))
        var_scale = float(template.get("var_scale", 1.0))

        new_mean = old_mean.copy()
        new_cov = old_cov.copy()

        # event_scale
        # mean_shift: mean_scale
        # var_shift: var_scale
        # new_mode: var_scale but using other scale aswell for calculation
        event_scale = 1.0

        if anomaly_type == "mean_shift":
            if mean_direction is not None:
                new_mean = old_mean + mean_direction * mean_scale
            new_cov = old_cov.copy()
            event_scale = mean_scale

        elif anomaly_type == "var_shift":
            new_mean = old_mean
            old_vars = np.diag(old_cov).copy()
            new_vars = np.maximum(old_vars * var_scale, 1e-12)
            new_cov = self._rand_spd_from_variances(new_vars)
            event_scale = var_scale

        elif anomaly_type == "new_mode":
            if mean_direction is not None:
                new_mean = old_mean + mean_direction * mean_scale
            old_vars = np.diag(old_cov).copy()
            new_vars = np.maximum(old_vars * var_scale, 1e-12)
            new_cov = self._rand_spd_from_variances(new_vars)
            event_scale = var_scale

        # new components
        current_components[mode_idx] = (new_mean, new_cov)

        # keep history of anomalies
        event = AnomalyEvent(
            type=anomaly_type,
            iteration=iteration,
            affected_mode=mode_idx,
            ext_idx=ext_idx,
            old_mean=old_mean,
            old_cov=old_cov,
            new_mean=new_mean,
            new_cov=new_cov,
            scale=event_scale,
        )

        if anomaly_type not in self._anomaly_history:
            self._anomaly_history[anomaly_type] = []
        self._anomaly_history[anomaly_type].append(event)

        return current_components, True
    
    # Noise
    def _add_sensor_noise(self, x):
        df = self.sim_config.noise_student_t_df
        t_scale = self.sim_config.noise_student_t_scale
        quant_step = self.sim_config.quant_step
        clip = self.sim_config.clip

        y = x.copy()

        # Student-t via Gaussian scale mixture: N(0, t_scale^2 / G) with G ~ Gamma(df/2, df/2)
        if df is not None:
            G = np.random.gamma(shape=df/2, scale=2/df,
                                size=(len(y), 1))  # mean ~1
            y += np.random.randn(*y.shape) * (t_scale / np.sqrt(G))

        # Quantization (ADC)
        if quant_step is not None and quant_step > 0:
            y = np.round(y / quant_step) * quant_step

        # Clipping (saturation)
        if clip is not None:
            lo, hi = clip
            y = np.clip(y, lo, hi)

        return y

    # random symmetric positive definite matrix
    def _rand_spd(self, dim, eigmin=0.3, eigmax=2.0):
        A = np.random.randn(dim, dim)
        Q, _ = np.linalg.qr(A)  # random orthogonal rotation
        eig = np.random.uniform(eigmin, eigmax, dim)
        return Q @ np.diag(eig) @ Q.T  # symmetric positive-definite matrix

    # for given variances
    def _rand_spd_from_variances(self, var_list):
        dim = len(var_list)
        A = np.random.randn(dim, dim)
        Q, _ = np.linalg.qr(A)      # zufällige Rotation
        return Q @ np.diag(var_list) @ Q.T

    # get components from config
    def _build_components(self):
        dimensions = self.sim_config.num_dimensions
        components = []

        # random means / covs
        if self.sim_config.mean_range is not None and self.sim_config.var_range is not None:
            for _ in range(self.sim_config.num_modes):
                mean = np.random.uniform(
                    self.sim_config.mean_range[0],
                    self.sim_config.mean_range[1],
                    dimensions,
                )
                cov = self._rand_spd(
                    dimensions,
                    eigmin=self.sim_config.var_range[0],
                    eigmax=self.sim_config.var_range[1],
                )
                components.append((mean, cov))

        # provided means / diagonal covs
        elif self.sim_config.means_list is not None and self.sim_config.vars_list is not None:
            for mean_l, var_l in zip(self.sim_config.means_list, self.sim_config.vars_list):
                mean = np.asarray(mean_l, dtype=float)
                var = np.asarray(var_l, dtype=float)
                cov = self._rand_spd_from_variances(var)
                components.append((mean, cov))

        if len(components) != self.sim_config.num_modes:
            raise ValueError("Number of components must equal num_modes.")

        return components

    # generate mode weights over extended indices
    def _mode_weights_array(self):
        extended_num_modes = 2 * self.sim_config.num_modes
        if self.sim_config.mode_weights is not None:
            pi = np.asarray(self.sim_config.mode_weights, dtype=float)
            if pi.shape[0] != extended_num_modes:
                raise ValueError(
                    f"mode_weights length {pi.shape[0]} does not match 2*num_modes={extended_num_modes}"
                )
            pi = pi / (pi.sum() + 1e-12)
        else:
            pi = np.ones(extended_num_modes) / extended_num_modes
        return pi

    # generator used to generate data
    def _sequence_generator(self, components, iterations, *, enable_stream_anomalies: bool = False):
        num_base_modes = len(components)
        assert num_base_modes == self.sim_config.num_modes
        extended_num_modes = 2 * num_base_modes

        pi = self._mode_weights_array()

        current_components = [(m.copy(), c.copy()) for (m, c) in components]

        for it in range(iterations):
            # integer counts for each extended index this iteration
            counts = np.random.multinomial(
                self.sim_config.samples_per_iteration, pi)

            # Simple extended order: 0,1,2,...,2*num_modes-1
            # even -> pure mode, odd -> transition
            for ext_idx in range(extended_num_modes):
                n = counts[ext_idx]

                if n <= 0:
                    continue
                    
                # choose wether to apply anomaly for this mode
                is_anom_for_mode = False
                if enable_stream_anomalies:
                    current_components, is_anom_for_mode = self._choose_mode_and_anomaly(
                        current_components=current_components,
                        ext_idx=ext_idx,
                        iteration=it,
                    )

                if ext_idx % 2 == 0:
                    is_anom_label = bool(is_anom_for_mode)
                else:
                    is_anom_label = False

                if ext_idx % 2 == 0:
                    # Pure mode k
                    k = ext_idx // 2
                    mean_k, cov_k = current_components[k]
                    samples = np.random.multivariate_normal(
                        mean_k, cov_k, size=n)
                    samples = self._add_sensor_noise(samples)
                    labels = np.full(n, ext_idx, dtype=int)

                    for s, l in zip(samples, labels):
                        # Any sample generated while this mode is in anomaly state is anomalous,
                        # independent of proximity to any component.
                        sample_is_anom = bool(is_anom_label)

                        yield s, l, sample_is_anom

                else:
                    # Transition k -> k_next
                    k = (ext_idx - 1) // 2
                    a = k
                    b = (k + 1) % num_base_modes
                    mean_a, cov_a = current_components[a]
                    mean_b, cov_b = current_components[b]

                    # "liquid" transition: (N-1)/N * A + 1/N * B  ...  0/N * A + N/N * B
                    for i in range(n):
                        # weights for this sample
                        w_b = (i + 1) / n
                        w_a = 1.0 - w_b

                        mean_t = w_a * mean_a + w_b * mean_b
                        cov_t = w_a * cov_a + w_b * cov_b  # linear interpolation of spread

                        s = np.random.multivariate_normal(mean_t, cov_t)
                        s = self._add_sensor_noise(s[np.newaxis, :])[0]
                        sample_is_anom = bool(is_anom_label)

                        yield s, ext_idx, sample_is_anom

    # public api simulate_data for "training_data"
    def simulate_data(self):
        components = self._build_components()

        total_samples = self.sim_config.samples_per_iteration * self.sim_config.iterations_during_training

        data = np.empty((total_samples, self.sim_config.num_dimensions))
        labels = np.empty(total_samples, dtype=int)

        gen = self._sequence_generator(components, self.sim_config.iterations_during_training, enable_stream_anomalies=False)
        for i, (sample, label, _) in enumerate(gen):
            data[i] = sample
            labels[i] = label

        return data, labels, components

    # public api simulate_continuous_data for "streaming data"
    def simulate_continuous_data(self, components):
        iterations = int(self.sim_config.iterations_for_generator)
        return self._sequence_generator(components, iterations, enable_stream_anomalies=True)


    def generate_samples_from_distribution(self, mean: np.ndarray, var: np.ndarray, n: int):
        """Generate n samples from specified Gaussian distribution with noise."""
        cov = self._rand_spd_from_variances(var)
        samples = np.random.multivariate_normal(mean, cov, size=n)
        samples = self._add_sensor_noise(samples)
        return samples


    # Visualization
    def visualize_data(self, data, labels, *, ax=None, show: bool = True, title: str | None = None):
        """Visualize (N, D) data colored by extended labels (modes + transitions).
        """
        import matplotlib.pyplot as plt

        dimension = int(data.shape[1])
        num_labels = 2 * self.sim_config.num_modes

        if dimension == 2:
            if ax is None:
                fig, ax = plt.subplots()
            for m in range(num_labels):
                sel = (labels == m)
                if np.any(sel):
                    ax.scatter(data[sel, 0], data[sel, 1],
                               label=f'Comp {m}', alpha=0.6)
            ax.set_title(title or '2D Scatter Plot of Simulated Data (modes + transitions)')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.legend()

        elif dimension == 3:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            for m in range(num_labels):
                ax.computed_zorder = False
                is_transition = (m % 2 == 1)
                z = 1 if is_transition else 10  # transitions unten, modes oben
                sel = (labels == m)
                if np.any(sel):
                    ax.scatter(data[sel, 0], data[sel, 1], data[sel, 2],
                               label=f'Comp {m}', alpha=0.6, zorder=z)
            ax.set_title(title or '3D Scatter Plot of Simulated Data (modes + transitions)')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')
            ax.legend()

        elif dimension == 4:
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            fig = ax.figure
            cmap = plt.get_cmap('viridis')
            scatter = ax.scatter(
                data[:, 0], data[:, 1], data[:, 2],
                c=data[:, 3], cmap=cmap, alpha=0.6,
            )
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
            cbar.set_label('Dimension 4 Value')

            ax.set_title(title or '4D Scatter Plot of Simulated Data (Color = 4th Dimension)')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')

        else:
            raise ValueError(f"Unsupported dimension for visualization: {dimension}")

        if show:
            plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sim_config = DataSimConfig(random_seed=123,
                               mean_range=None, var_range=None,
                               means_list=[(1, 1, 1), (1, 5, 10), (10, 10, 10)],
                               vars_list=[(0.5, 0.2, 0.7), (0.1, 0.3, 0.5), (0.2, 0.1, 0.5)],
                               num_dimensions=3,
                               num_modes=3,
                               mode_weights=[0.5, 0.05, 0.3, 0.03, 0.11, 0.01],
                               noise_student_t_df=5.0, noise_student_t_scale=0.2,
                               samples_per_iteration=1000,
                               iterations_during_training=10,
                               iterations_for_generator=10)
    
    anomaly_config = AnomalyConfig(stream_anomaly_prob=0.1)
                             
    simulator = DataSim(sim_config=sim_config, anomaly_config=anomaly_config)

    data, labels, components = simulator.simulate_data()

    stream = simulator.simulate_continuous_data(components=components)

    g_samples = []
    g_labels = []

    for i in range(10000):
        sample, mode_idx, is_anomaly = next(stream)
        if i % 5 == 0:
            print(f"Sample {i+1}: {sample}, Mode: {mode_idx+1}, Anomaly: {is_anomaly}")
        g_samples.append(sample)
        g_labels.append(mode_idx)

    # Show training + streaming data side-by-side (single window).
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    simulator.visualize_data(data, labels, ax=ax1, show=False, title="Training data")
    simulator.visualize_data(np.array(g_samples), np.array(g_labels), ax=ax2, show=False, title="Streaming data")
    fig.tight_layout()
    plt.show()
    # print(simulator._anomaly_history)
    # print(simulator._stream_anomaly_templates)