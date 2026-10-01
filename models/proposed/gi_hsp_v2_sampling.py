import math
import random

from torch.utils.data import Sampler


class RotatingBalancedBatchSampler(Sampler):
    def __init__(
        self,
        targets,
        batch_size,
        seed=0,
    ):
        self.targets = [
            int(target)
            for target in targets
        ]
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        if self.batch_size % 2 != 0:
            raise ValueError(
                "batch_size must be even for a 1:1 ratio"
            )

        unsupported = set(self.targets) - {0, 1}

        if unsupported:
            raise ValueError(
                f"Unsupported targets: {sorted(unsupported)}"
            )

        class_indices = {
            label: [
                index
                for index, target in enumerate(self.targets)
                if target == label
            ]
            for label in (0, 1)
        }

        if not class_indices[0] or not class_indices[1]:
            raise ValueError(
                "Both binary classes must be present"
            )

        counts = {
            label: len(indices)
            for label, indices in class_indices.items()
        }
        self.minority_label = min(
            counts,
            key=lambda label: (
                counts[label],
                label,
            ),
        )
        self.majority_label = 1 - self.minority_label
        self.minority_indices = class_indices[
            self.minority_label
        ]
        self.majority_indices = class_indices[
            self.majority_label
        ]
        self.samples_per_class = len(
            self.minority_indices
        )
        self.per_class_batch_size = (
            self.batch_size // 2
        )

        majority_order = list(self.majority_indices)
        random.Random(self.seed).shuffle(
            majority_order
        )
        self.majority_order = majority_order

    def set_epoch(self, epoch):
        epoch = int(epoch)

        if epoch < 0:
            raise ValueError(
                "epoch must be nonnegative"
            )

        self.epoch = epoch

    def _majority_epoch_indices(self):
        count = len(self.majority_order)
        start = (
            self.epoch * self.samples_per_class
        ) % count

        return [
            self.majority_order[
                (start + offset) % count
            ]
            for offset in range(self.samples_per_class)
        ]

    def __iter__(self):
        generator = random.Random(
            self.seed + self.epoch
        )

        minority = list(self.minority_indices)
        generator.shuffle(minority)
        majority = self._majority_epoch_indices()

        for start in range(
            0,
            self.samples_per_class,
            self.per_class_batch_size,
        ):
            minority_batch = minority[
                start:
                start + self.per_class_batch_size
            ]
            majority_batch = majority[
                start:
                start + self.per_class_batch_size
            ]
            batch = minority_batch + majority_batch
            generator.shuffle(batch)
            yield batch

    def __len__(self):
        return math.ceil(
            self.samples_per_class
            / self.per_class_batch_size
        )
