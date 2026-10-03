from preprocessing.gi_hsp_v2.sequence_assembly import (
    assemble_temporal_sequence,
)
from preprocessing.gi_hsp_v2.sequence_index import (
    PARTITIONS,
)


def get_window(connection, window_id):
    cursor = connection.execute(
        """
        SELECT
            window_id,
            capture_id,
            source_label,
            binary_label,
            start_second,
            end_second_exclusive,
            active_second_count,
            stride_seconds
        FROM windows
        WHERE window_id = ?
        """,
        (str(window_id),),
    )
    row = cursor.fetchone()

    if row is None:
        raise KeyError(f"Unknown window: {window_id}")

    columns = [
        description[0]
        for description in cursor.description
    ]
    return dict(zip(columns, row))


def list_partition_windows(
    connection,
    fold,
    partition_name,
):
    if partition_name not in PARTITIONS:
        raise ValueError(
            f"Unsupported partition: {partition_name!r}"
        )

    cursor = connection.execute(
        """
        SELECT
            windows.window_id,
            windows.capture_id,
            windows.source_label,
            windows.binary_label,
            windows.start_second,
            windows.end_second_exclusive,
            windows.active_second_count,
            windows.stride_seconds
        FROM windows
        INNER JOIN capture_partitions
            ON capture_partitions.capture_id = windows.capture_id
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
            int(fold),
            partition_name,
        ),
    )
    columns = [
        description[0]
        for description in cursor.description
    ]

    return [
        dict(zip(columns, row))
        for row in cursor
    ]


def load_window_records(
    flow_connection,
    dataset,
    window,
):
    rows = flow_connection.execute(
        """
        SELECT *
        FROM flows
        WHERE dataset = ?
          AND capture_id = ?
          AND timestamp >= ?
          AND timestamp < ?
        ORDER BY timestamp, record_id
        """,
        (
            str(dataset),
            window["capture_id"],
            int(window["start_second"]),
            int(window["end_second_exclusive"]),
        ),
    )

    return [dict(row) for row in rows]


def load_assembled_window(
    flow_connection,
    index_connection,
    dataset,
    window_id,
    graph_view,
    bin_seconds=1,
    graph_attribute_mode="full",
):
    window = get_window(
        index_connection,
        window_id,
    )
    records = load_window_records(
        flow_connection,
        dataset,
        window,
    )
    sequence = assemble_temporal_sequence(
        records,
        window,
        graph_view,
        bin_seconds=bin_seconds,
        graph_attribute_mode=graph_attribute_mode,
    )
    sequence["dataset"] = str(dataset)
    sequence["stride_seconds"] = int(
        window["stride_seconds"]
    )

    return sequence
