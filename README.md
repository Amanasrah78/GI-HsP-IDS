# GI-HsP IDS

This repository currently implements reproducible traffic
preprocessing, graph construction, and aligned packet-derived
training windows for the GI-HsP intrusion-detection pipeline.

## Process an experiment capture

The command expects these files:

- `capture/pcap/<experiment_id>.pcap`
- `capture/pcap/<experiment_id>.timing.json`
- `experiments/<experiment_id>.yaml`

Run:

```bash
./scripts/process_pcap.sh \
  capture/pcap/<experiment_id>.pcap \
  <experiment_id>
```

The pipeline runs Zeek, creates the flow and MQTT CSV files, builds
the static and five-second dynamic graphs, extracts packet records,
and writes aligned training windows.

Generated artifacts are placed under `results/` and `graph/output/`.
These directories are excluded from version control.

## Packet training-window format

The default output is:

```text
results/processed/<experiment_id>.windows.jsonl
```

Each JSONL line represents one non-overlapping dynamic-graph window
and contains:

- `experiment_id`, for provenance.
- `window_index`, `start_ts`, and `end_ts`, for alignment metadata.
- `packet_features`, containing the predictive packet aggregates.
- `label`, containing the supervised targets.

Windows use half-open boundaries: `[start_ts, end_ts)`.

The packet feature vector contains:

- `packet_count`
- `frame_bytes`
- `tcp_payload_bytes`
- `mean_frame_len`
- `mean_tcp_payload_len`
- `mean_interarrival_seconds`
- `std_interarrival_seconds`
- `max_interarrival_seconds`
- `suspected_retransmission_count`
- `previous_segment_not_captured_count`

Packet timestamps are sorted before inter-arrival statistics are
calculated. Empty and single-packet windows use zero for undefined
inter-arrival statistics.

The two TShark analysis indicators are not ground truth.
`suspected_retransmission_count` uses
`tcp.analysis.retransmission`. The
`previous_segment_not_captured_count` feature uses
`tcp.analysis.lost_segment`, which can be triggered at capture start.

See the
[Wireshark TCP field reference](https://www.wireshark.org/docs/dfref/t/tcp.html)
for the authoritative definitions.

## Predictive-feature boundary

Only values nested under `packet_features` belong to the packet
branch's predictive vector.

The following fields are metadata or targets and must not be supplied
as predictive inputs:

- `experiment_id`
- `window_index`
- `start_ts`
- `end_ts`
- `label`

Raw IP addresses, ports, and TShark stream indices may remain in
auditable intermediate or graph artifacts. They are not copied into
`packet_features`.

This exclusion reduces shortcut-learning risk. Destination port alone
has been shown to separate multiple benchmark IDS datasets for
dataset-specific rather than transferable reasons. See
[D'hooge et al., DIMVA 2022](https://doi.org/10.1007/978-3-031-09484-2_2).

## Evaluation constraints

The current five-second duration matches the dynamic graph. It remains
an experimental parameter and should later be evaluated with a
window-size ablation.

Training, validation, and test partitions should be grouped by complete
experiment. Adjacent windows from one capture must not be randomly
distributed across partitions because that permits temporal and
experiment-specific leakage.

The packet branch complements the flow and graph representations.
MQTT intrusion-detection results indicate that flow-level features can
provide stronger discrimination than packet-only features for
MQTT-specific attacks. See
[Hindy et al., MQTT-IoT-IDS2020](https://arxiv.org/abs/2006.15340).

## Tests

Run:

```bash
python3 -m unittest discover -s tests -v
```
