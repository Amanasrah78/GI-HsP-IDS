import math

import torch


class StreamingFeatureNormalizer:
    def __init__(
        self,
        feature_names,
        binary_feature_names=(),
        epsilon=1.0e-8,
    ):
        self.feature_names = tuple(feature_names)
        self.binary_feature_names = tuple(
            binary_feature_names
        )
        self.epsilon = float(epsilon)

        if not self.feature_names:
            raise ValueError(
                "At least one feature name is required"
            )

        if len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("Feature names must be unique")

        unknown_binary = (
            set(self.binary_feature_names)
            - set(self.feature_names)
        )

        if unknown_binary:
            raise ValueError(
                "Unknown binary features: "
                f"{sorted(unknown_binary)}"
            )

        if (
            not math.isfinite(self.epsilon)
            or self.epsilon <= 0
        ):
            raise ValueError("epsilon must be positive")

        self.binary_indices = tuple(
            self.feature_names.index(name)
            for name in self.binary_feature_names
        )
        self.magnitude_indices = tuple(
            index
            for index in range(len(self.feature_names))
            if index not in self.binary_indices
        )

        width = len(self.feature_names)
        self.count = 0
        self.mean = torch.zeros(
            width,
            dtype=torch.float64,
        )
        self.m2 = torch.zeros(
            width,
            dtype=torch.float64,
        )
        self.scale = None

    @property
    def fitted(self):
        return self.scale is not None

    def _selected_rows(self, values, active_mask):
        values = torch.as_tensor(values)

        if values.ndim < 2:
            raise ValueError(
                "values must have at least two dimensions"
            )

        if values.shape[-1] != len(self.feature_names):
            raise ValueError(
                "Feature width does not match feature names"
            )

        active_mask = torch.as_tensor(
            active_mask,
            dtype=torch.bool,
            device=values.device,
        )

        if active_mask.shape != values.shape[:-1]:
            raise ValueError(
                "active_mask shape does not match values"
            )

        rows = values[active_mask].to(
            dtype=torch.float64,
            device="cpu",
        )

        if rows.numel() == 0:
            return rows.reshape(
                0,
                len(self.feature_names),
            )

        if not torch.isfinite(rows).all():
            raise ValueError(
                "Feature values must be finite"
            )

        if self.magnitude_indices:
            magnitude = rows[
                :,
                list(self.magnitude_indices),
            ]

            if torch.any(magnitude < 0):
                raise ValueError(
                    "Magnitude features must be nonnegative"
                )

            rows[
                :,
                list(self.magnitude_indices),
            ] = torch.log1p(magnitude)

        return rows

    def update(self, values, active_mask):
        if self.fitted:
            raise RuntimeError(
                "Cannot update finalized statistics"
            )

        rows = self._selected_rows(
            values,
            active_mask,
        )
        batch_count = rows.shape[0]

        if batch_count == 0:
            return

        batch_mean = rows.mean(dim=0)
        centered = rows - batch_mean
        batch_m2 = torch.sum(
            centered * centered,
            dim=0,
        )

        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
            self.count = batch_count
            return

        total = self.count + batch_count
        delta = batch_mean - self.mean

        self.mean = (
            self.mean
            + delta * (batch_count / total)
        )
        self.m2 = (
            self.m2
            + batch_m2
            + delta.pow(2)
            * self.count
            * batch_count
            / total
        )
        self.count = total

    def finalize(self):
        if self.count == 0:
            raise RuntimeError(
                "Cannot finalize empty statistics"
            )

        standard_deviation = torch.sqrt(
            self.m2 / self.count
        )

        self.scale = torch.where(
            standard_deviation > self.epsilon,
            standard_deviation,
            torch.ones_like(standard_deviation),
        )

        return self

    def transform(self, values, active_mask):
        if not self.fitted:
            raise RuntimeError(
                "Statistics must be finalized first"
            )

        values = torch.as_tensor(values)
        active_mask = torch.as_tensor(
            active_mask,
            dtype=torch.bool,
            device=values.device,
        )

        if values.shape[-1] != len(self.feature_names):
            raise ValueError(
                "Feature width does not match feature names"
            )

        if active_mask.shape != values.shape[:-1]:
            raise ValueError(
                "active_mask shape does not match values"
            )

        output = torch.zeros_like(values)

        if not torch.any(active_mask):
            return output

        selected = values[active_mask]

        if not torch.isfinite(selected).all():
            raise ValueError(
                "Feature values must be finite"
            )

        if self.binary_indices:
            indices = list(self.binary_indices)
            output_rows = output[active_mask]
            output_rows[:, indices] = selected[:, indices]
            output[active_mask] = output_rows

        if self.magnitude_indices:
            indices = list(self.magnitude_indices)
            magnitude = selected[:, indices]

            if torch.any(magnitude < 0):
                raise ValueError(
                    "Magnitude features must be nonnegative"
                )

            transformed = torch.log1p(magnitude)
            mean = self.mean[indices].to(
                dtype=values.dtype,
                device=values.device,
            )
            scale = self.scale[indices].to(
                dtype=values.dtype,
                device=values.device,
            )

            output_rows = output[active_mask]
            output_rows[:, indices] = (
                transformed - mean
            ) / scale
            output[active_mask] = output_rows

        return output

    def state_dict(self):
        if not self.fitted:
            raise RuntimeError(
                "Statistics must be finalized first"
            )

        return {
            "feature_names": list(self.feature_names),
            "binary_feature_names": list(
                self.binary_feature_names
            ),
            "epsilon": self.epsilon,
            "count": int(self.count),
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state):
        normalizer = cls(
            state["feature_names"],
            state["binary_feature_names"],
            state["epsilon"],
        )
        normalizer.count = int(state["count"])
        normalizer.mean = torch.tensor(
            state["mean"],
            dtype=torch.float64,
        )
        normalizer.m2 = torch.tensor(
            state["m2"],
            dtype=torch.float64,
        )
        normalizer.scale = torch.tensor(
            state["scale"],
            dtype=torch.float64,
        )
        return normalizer
