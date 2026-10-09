# GI-HSP

**Leakage-controlled dual-view intrusion detection under held-out attacks and
distribution shift**

[![Release](https://img.shields.io/badge/release-v1.2.0-blue)](https://github.com/Amanasrah78/GI-HsP-IDS/releases/tag/v1.2.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23261916.svg)](https://doi.org/10.5281/zenodo.23261916)

This repository contains the code, frozen configurations, protocol checksums,
tests, and result summarizers for the final GI-HSP experimental evaluation.
GI-HSP studies whether synchronized temporal flow statistics and dynamic
communication topology provide complementary evidence for MQTT intrusion
detection.

The primary evaluation uses capture-disjoint MQTTset partitions. Each of four
folds holds out one attack scenario for testing and another for validation.
The confirmatory replication uses training seeds 5 through 14. Preprocessing
is fitted on the training partition only, while validation and test retain
their observed class prevalence.

Release `v1.2.0` corresponds to commit
`13119bd4af0f12637a9210d50e9d4e9d3a7dcfb8`.

## Experimental design

Each default observation contains ten synchronized five-second steps, giving a
50-second context. The flow branch encodes temporal flow features with a
Transformer. The topology branch applies direction-aware graph operations and
recurrent temporal aggregation. GI-HSP combines both embeddings through a
feature-wise gate.

The frozen confirmatory protocol is defined in
[`configs/gi_hsp_v2_confirmatory_replication.yaml`](configs/gi_hsp_v2_confirmatory_replication.yaml).
It evaluates four folds and ten confirmatory seeds for six conditions:

| Condition | Purpose |
| --- | --- |
| `fused_identity` | Full GI-HSP with endpoint-identity topology |
| `fused_role_control` | Fusion after collapsing endpoints to client and broker roles |
| `flow_transformer` | Flow-only Transformer ablation |
| `topology_only` | Topology-only ablation |
| `flow_mlp_matched` | Capacity-matched flow MLP control |
| `flow_gru_matched` | Capacity-matched flow GRU control |

The design treats the training seed as the independent inference unit. Metrics
are macro-averaged across folds within each seed. The prespecified primary
metrics are balanced accuracy and Matthews correlation coefficient. Planned
contrasts use exact two-sided paired sign-flip tests, Student *t* 95% confidence
intervals, and Holm correction within each protocol and metric family.

## Included evaluations

The release contains the following reproducibility components:

- capture-aware MQTTset preprocessing and held-out-attack evaluation;
- identity, role-collapsed, flow-only, topology-only, and capacity-matched
  controls;
- protocol-matched adaptations of GraphIDS and E-GraphSAGE;
- a structure-only graph ablation that retains adjacency while zeroing
  prespecified node and edge attributes;
- fusion controls using score averaging and capacity-matched concatenation;
- a common-support observation-duration study at 5, 10, 20, 30, 40, and
  50 seconds;
- zero-shot external-evaluation pipelines for X-IIoTID, CSE-CIC-IDS2018,
  CICIoT2023, and CIC-BCCC-NRC TabularIoTAttack-2024;
- generated host-space-perturbation evaluation, including the expanded
  60-capture protocol;
- statistical aggregation, calibration analysis, paired-effect summaries,
  efficiency measurement, and protocol-integrity tests.

GraphIDS and E-GraphSAGE are protocol-matched adaptations, not exact executions
of their source repositories. Their frozen protocol files document the retained
model semantics, necessary task adaptations, source commits, and limitations.

## Repository layout

```text
configs/                 Frozen data, training, comparison, and ablation protocols
datasets/                Expected raw and generated processed-data locations
docs/                    Recorded environment and supplementary result notes
experiments/             Capture manifests
graph/                   Static and dynamic graph construction
hsp/                     Host-space-perturbation capture tooling and taxonomy
models/baselines/        Earlier and contextual baseline implementations
models/proposed/         GI-HSP V2 models, runners, evaluators, and summarizers
preprocessing/gi_hsp_v2/ Canonical adapters, stores, partitions, and sequence indexes
results/                 Versioned summaries and generated-output locations
scripts/                 Capture, preprocessing, monitoring, and orchestration scripts
tests/gi_hsp_v2/         Protocol, model, adapter, evaluator, and summary tests
```

## Environment

The recorded release environment is Linux on ARM64 with Python 3.12.3, Docker
29.8.0, and Docker Compose 5.5.1. See
[`docs/environment.txt`](docs/environment.txt) for the complete record.

The core Python code uses PyTorch, NumPy, scikit-learn, PyYAML, and pytest.
Several data-preparation paths additionally require Zeek, TShark, Docker, or
dataset-specific source files. `requirements-dev.txt` pins the pytest version
used for development tests; it is not a complete runtime lockfile.

Run commands from the repository root so that module imports and relative
protocol paths resolve consistently.

## Data and artifact policy

Raw datasets, restricted captures, generated per-run results, and model
checkpoints are not redistributed. Obtain each dataset from its authoritative
source and place it at the path expected by the relevant frozen configuration.
The repository supplies acquisition assumptions, canonical adapters, expected
paths, cardinality checks, and SHA-256 sidecars for protocol verification.

The principal generated paths are excluded from version control:

```text
datasets/processed/gi_hsp_v2/
results/gi_hsp_v2/comparisons/experiments/
results/evaluation/
graph/output/
```

Do not fit normalization, select features, tune thresholds, or modify model
parameters using validation-external or test-external labels. External
evaluation reuses the corresponding MQTTset training-fold normalizer and the
frozen decision rule.

## Reproducing the experiment matrices

All matrix runners accept `--dry-run`. Use it first to verify the protocol
sidecar, enumerate the jobs, and inspect the commands without starting model
training.

### Confirmatory replication

```bash
python3 -m models.proposed.run_gi_hsp_v2_confirmatory_matrix --dry-run
```

Without `--dry-run`, this executes the complete six-condition, four-fold,
ten-seed matrix. Existing complete jobs are validated and skipped. An existing
incomplete output directory causes the runner to stop instead of silently
overwriting it.

### Structure-only ablation

```bash
python3 -m models.proposed.run_gi_hsp_v2_structure_ablation_matrix --dry-run
```

The protocol and checksum are
[`configs/gi_hsp_v2_structure_ablation.yaml`](configs/gi_hsp_v2_structure_ablation.yaml)
and
[`configs/gi_hsp_v2_structure_ablation.yaml.sha256`](configs/gi_hsp_v2_structure_ablation.yaml.sha256).

### Fusion controls

```bash
python3 -m models.proposed.run_gi_hsp_v2_fusion_control_matrix --dry-run
```

The frozen controls are defined in
[`configs/gi_hsp_v2_fusion_control.yaml`](configs/gi_hsp_v2_fusion_control.yaml).

### Observation-duration study

```bash
python3 -m models.proposed.run_gi_hsp_v2_horizon_matrix --dry-run
```

This runner evaluates sequence lengths 1, 2, 4, 6, 8, and 10, corresponding to
5, 10, 20, 30, 40, and 50 seconds. It holds the evaluated population fixed by
requiring activity in the first five-second prefix. The study measures
performance under different observation durations; it is not an attack-delay
or early-detection experiment.

### Protocol-matched graph comparators

```bash
python3 -m models.proposed.run_gi_hsp_v2_e_graphsage_matrix --dry-run
```

The E-GraphSAGE matrix contains 40 fold-seed jobs. The GraphIDS release includes
resumable orchestration scripts because its 40-job matrix was executed in
stages:

```bash
bash scripts/run_graphids_original_resume.sh
```

Review the protocol files before interpreting comparator results:

- [`configs/gi_hsp_v2_comparison_e_graphsage.yaml`](configs/gi_hsp_v2_comparison_e_graphsage.yaml)
- [`configs/gi_hsp_v2_comparison_graphids.yaml`](configs/gi_hsp_v2_comparison_graphids.yaml)

### External and expanded HsP evaluation

After confirmatory checkpoints and the required canonical stores are present,
enumerate the primary external jobs with:

```bash
python3 -m models.proposed.evaluate_gi_hsp_v2_confirmatory_external_matrix --dry-run
```

The additional dataset-specific matrix evaluators are:

```text
models.proposed.evaluate_gi_hsp_v2_cse_cic_ids2018_matrix
models.proposed.evaluate_gi_hsp_v2_ciciot2023_matrix
models.proposed.evaluate_gi_hsp_v2_cic_bccc_matrix
```

Enumerate the expanded 60-capture HsP evaluation with:

```bash
python3 -m models.proposed.evaluate_gi_hsp_v2_generated_hsp_expanded_matrix --dry-run
```

## Processing an individual capture

The general capture-processing pipeline expects:

- `capture/pcap/<experiment_id>.pcap`;
- `capture/pcap/<experiment_id>.timing.json`;
- `experiments/<experiment_id>.yaml`.

Run:

```bash
./scripts/process_pcap.sh \
  capture/pcap/<experiment_id>.pcap \
  <experiment_id>
```

The script runs Zeek, creates flow and MQTT tables, builds static and
five-second dynamic graphs, extracts packet records, and writes aligned windows
and sequences. Generated files are placed under `results/` and `graph/output/`.

Sequences use half-open step boundaries, `[start_ts, end_ts)`. Node indices are
stable within an experiment but are topology coordinates, not ordinal numeric
features. Raw endpoint identifiers may remain in auditable intermediate
artifacts but are excluded from the packet-feature vector.

## Tests

Run the complete test suite with:

```bash
python3 -m pytest
```

For the final protocol-specific suite only:

```bash
python3 -m pytest tests/gi_hsp_v2
```

The tests cover protocol hashes and invariants, capture-disjoint partitions,
normalization boundaries, graph semantics, dense and sparse topology paths,
external adapters, matrix enumeration, result aggregation, and statistical
summaries. They do not replace dataset license checks or independent validation
of externally obtained raw files.

## Interpreting the release

This repository supports controlled reproduction of the implemented study. It
does not imply that every external source uses an identical capture process or
feature schema. Cross-dataset results combine changes in environment,
acquisition, labels, prevalence, and available fields. Accordingly, they should
be interpreted as distribution-shift evaluations rather than additional random
test splits.

The observation-duration analysis uses common support and should not be
reported as detection latency. The generated HsP protocol evaluates the listed
tools, goals, schedules, and environments; it does not establish invariance to
all possible host-space perturbations.

## Release and citation

- GitHub release: [GI-HSP reproducibility release v1.2.0](https://github.com/Amanasrah78/GI-HsP-IDS/releases/tag/v1.2.0)
- Archived release: [https://doi.org/10.5281/zenodo.23261916](https://doi.org/10.5281/zenodo.23261916)

When using the software or protocols, cite the archived release DOI above and
the associated manuscript. For exact reproduction, record the release tag,
commit, protocol-file SHA-256 values, dataset versions, and local environment.
