import os
from pathlib import Path

from torch.utils.data import Dataset

from models.proposed.gi_hsp_v2_feature_contract import (
    validate_graph_view,
)
from models.proposed.gi_hsp_v2_tensor_conversion import (
    temporal_sequence_to_tensors,
)
from preprocessing.gi_hsp_v2.flow_store import (
    open_flow_store,
)
from preprocessing.gi_hsp_v2.sequence_index import (
    PARTITIONS,
    get_metadata,
    open_sequence_index,
)
from preprocessing.gi_hsp_v2.sequence_loader import (
    load_assembled_window,
)


class GIHSPV2SequenceDataset(Dataset):
    def __init__(
        self,
        flow_store_path,
        sequence_index_path,
        dataset,
        fold,
        partition_name,
        graph_view,
        normalizer=None,
    ):
        if partition_name not in PARTITIONS:
            raise ValueError(
                f"Unsupported partition: {partition_name!r}"
            )

        self.flow_store_path = str(
            Path(flow_store_path)
        )
        self.sequence_index_path = str(
            Path(sequence_index_path)
        )
        self.dataset = str(dataset)
        self.fold = int(fold)
        self.partition_name = partition_name
        self.graph_view = validate_graph_view(
            graph_view
        )
        self.normalizer = normalizer

        if self.fold <= 0:
            raise ValueError("fold must be positive")

        self._flow_connection = None
        self._index_connection = None
        self._connection_pid = None

        index_connection = open_sequence_index(
            self.sequence_index_path
        )

        try:
            build_contract = get_metadata(
                index_connection,
                "build_contract",
            )
            try:
                self.bin_seconds = int(
                    build_contract["bin_seconds"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Sequence index build contract must define "
                    "bin_seconds"
                ) from exc

            if self.bin_seconds <= 0:
                raise ValueError(
                    "Sequence index bin_seconds must be positive"
                )

            rows = index_connection.execute(
                """
                SELECT
                    windows.window_id,
                    windows.binary_label
                FROM windows
                INNER JOIN capture_partitions
                    ON capture_partitions.capture_id
                        = windows.capture_id
                    AND capture_partitions.stride_seconds
                        = windows.stride_seconds
                WHERE capture_partitions.fold = ?
                  AND capture_partitions.partition_name = ?
                ORDER BY
                    windows.capture_id,
                    windows.start_second,
                    windows.window_id
                """,
                (
                    self.fold,
                    self.partition_name,
                ),
            ).fetchall()
        finally:
            index_connection.close()

        self.window_ids = [
            row[0]
            for row in rows
        ]
        self.targets = [
            int(row[1])
            for row in rows
        ]

    def __len__(self):
        return len(self.window_ids)

    def _ensure_connections(self):
        current_pid = os.getpid()

        if (
            self._connection_pid is not None
            and self._connection_pid != current_pid
        ):
            self.close()

        if self._flow_connection is None:
            self._flow_connection = open_flow_store(
                self.flow_store_path
            )

        if self._index_connection is None:
            self._index_connection = open_sequence_index(
                self.sequence_index_path
            )

        self._connection_pid = current_pid

    def __getitem__(self, index):
        if index < 0:
            index += len(self)

        if index < 0 or index >= len(self):
            raise IndexError(index)

        self._ensure_connections()
        window_id = self.window_ids[index]

        sequence = load_assembled_window(
            self._flow_connection,
            self._index_connection,
            dataset=self.dataset,
            window_id=window_id,
            graph_view=self.graph_view,
            bin_seconds=self.bin_seconds,
        )

        tensors = temporal_sequence_to_tensors(sequence)

        if self.normalizer is not None:
            tensors = self.normalizer.transform(tensors)

        return tensors

    def close(self):
        if self._flow_connection is not None:
            self._flow_connection.close()
            self._flow_connection = None

        if self._index_connection is not None:
            self._index_connection.close()
            self._index_connection = None

        self._connection_pid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_flow_connection"] = None
        state["_index_connection"] = None
        state["_connection_pid"] = None
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
