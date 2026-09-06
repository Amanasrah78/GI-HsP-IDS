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
and writes aligned training windows. When at least ten complete
windows are available, it also writes unpadded temporal sequences.

Generated artifacts are placed under `results/` and `graph/output/`.
These directories are excluded from version control.

## Packet training-window format

The default output is:

```text
results/processed/<experiment_id>.windows.jsonl
```

Each JSONL line represents one non-overlapping dynamic-graph window
and contains:

- `schema_version`, for compatibility checks.
- `experiment_id`, for provenance.
- `window_seconds`, for window-size compatibility.
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

## Encoded graph topology

Each window record also contains a `graph` object with:

- `node_count`, the stable spatial-axis size for the experiment.
- `active_node_indices`, the nodes active in the current window.
- `edges`, containing `source_index`, `target_index`,
  `event_count`, and `payload_bytes`.
- `node_features`, an ordered feature vector for each stable node index.
  Each node contains `active`, `in_neighbor_count`,
  `out_neighbor_count`, `in_event_count`, `out_event_count`,
  `in_payload_bytes`, and `out_payload_bytes`.

The node-local features are identifier-free. Neighbor counts use
distinct directed neighbors, while event and payload fields aggregate
the directed edges incident on each node within the current window.

Node indices are generated once per experiment and remain stable
across its windows. This preserves temporal node correspondence.
The indices are topology coordinates only. They must not be treated
as ordered numeric node features or learned identity values.

Raw endpoint identifiers remain available in the dynamic graph for
auditability, but the IP-to-index mapping is not copied into the
training-window JSONL.

This stable spatial axis follows the source architecture's
spatio-temporal formulation, in which each node has a feature sequence
over successive time steps. See
[Aljuhani et al., MFTST](https://doi.org/10.1016/j.cose.2026.104999).

## Temporal sequence format

The default sequence output is:

```text
results/processed/<experiment_id>.sequences.jsonl
```

Each sequence contains ten consecutive five-second window records,
giving a 50-second temporal context. The default stride is one window,
so adjacent sequences overlap by nine windows. Sequences are never
padded. Captures with fewer than ten complete windows retain their
window JSONL but do not produce a sequence JSONL.

Each sequence record contains:

- `schema_version` and `window_schema_version`
- `experiment_id` and `window_seconds`
- `sequence_index` and `sequence_length`
- `start_window_index` and `end_window_index`
- `start_ts` and `end_ts`
- `steps`, containing the ordered packet features and encoded graph
  for each window
- `label`, stored once at sequence level

The sequence builder validates contiguous indices and timestamps,
constant labels, a constant window duration, and a stable graph-node
axis before serialization. The default length of ten and stride of one
match the current MFTST target configuration. Short diagnostic captures
can be processed without weakening this fixed-length model contract.

To generate sequences explicitly with different research parameters,
run:

```bash
python3 preprocessing/build_training_sequences.py \
  results/processed/<experiment_id>.windows.jsonl \
  --sequence-length 10 \
  --stride 1
```

Dataset partitions must be assigned by complete experiment before
overlapping sequences are used for model training. Splitting individual
sequences would place shared windows in different partitions.

Use the deterministic experiment-level partitioner to assign complete
experiments to train, validation, and test sets:

```bash
python3 preprocessing/build_dataset_partitions.py \
  <experiment_id_1> <experiment_id_2> <experiment_id_3> ... \
  --seed 0 \
  --output datasets/processed/partitions.json
```

The default requested fractions are 0.70 train, 0.15 validation, and
0.15 test. At least three experiment IDs are required, and every
experiment is assigned to exactly one partition.

The partitioner reads each experiment's class label from its manifest
and uses deterministic class-aware assignment so that class coverage is
preserved across train, validation, and test whenever the available
experiment counts and requested partition sizes make that possible.

## Predictive-feature boundary

Only values nested under `packet_features` belong to the packet
branch's predictive vector.

The following fields are metadata or targets and must not be supplied
as predictive inputs:

- `schema_version`
- `experiment_id`
- `window_seconds`
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

## Model training and ablation comparison

The proposed GI-HsP model combines a flow-temporal encoder, a dynamic
topology encoder, and gated cross-view fusion. Two ablation baselines
reuse the same data pipeline and training procedure:

- `flow_only`: flow-temporal branch only.
- `topology_only`: dynamic-topology branch only.

Train an individual model with its reproducible configuration:

```bash
python -m models.proposed.train_gi_hsp \
  --config configs/gi_hsp_training.yaml
```

Equivalent configurations are available at
`configs/flow_only_training.yaml` and
`configs/topology_only_training.yaml`.

Run all three models sequentially on the same dataset partitions with:

```bash
bash scripts/run_model_comparison.sh
```

Evaluation reports loss, accuracy, precision, recall, F1 score, a
binary confusion matrix, and sample count. Checkpoints are written to
separate directories so baseline runs do not overwrite the proposed
model checkpoint.


For repeated training-seed evaluation across the proposed model and
both ablations, run:

    python scripts/run_repeated_evaluation.py --seeds 0 1 2 3 4 --output results/evaluation/repeated_evaluation.json

Each model is trained independently for every requested seed with an
isolated checkpoint directory. The resulting JSON records each test run
and reports the mean and sample standard deviation of loss, accuracy,
precision, recall, and F1 score. Generated evaluation outputs under
`results/evaluation/` are ignored by Git.

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

