from models.proposed.evaluate_gi_hsp_v2_structure_ablation_external_matrix import (
    build_jobs,
    experiment_directory,
)
from models.proposed.gi_hsp_v2_structure_ablation_protocol import (
    load_structure_ablation_protocol,
)


PROTOCOL = "configs/gi_hsp_v2_structure_ablation.yaml"


def test_external_matrix_contains_160_unique_jobs():
    protocol = load_structure_ablation_protocol(PROTOCOL)
    jobs = build_jobs(protocol)
    assert len(jobs) == 160
    identities = {
        (
            job["evaluation_protocol"],
            job["condition"],
            job["seed"],
            job["fold"],
        )
        for job in jobs
    }
    assert len(identities) == 160
    assert {job["evaluation_protocol"] for job in jobs} == {
        "xiiotid", "generated_hsp"
    }
    assert {job["graph_attribute_mode"] for job in jobs} == {
        "structure_only"
    }


def test_matrix_can_select_one_external_protocol():
    protocol = load_structure_ablation_protocol(PROTOCOL)
    jobs = build_jobs(
        protocol,
        selected_protocols=("xiiotid",),
    )
    assert len(jobs) == 80
    assert {job["evaluation_protocol"] for job in jobs} == {
        "xiiotid"
    }


def test_experiment_directory_uses_secondary_namespace():
    path = experiment_directory(
        "results/gi_hsp_v2/experiments",
        "flow_structure_only_graph",
        5,
        1,
    )
    assert path.name == (
        "mqttset-structure-ablation-fold-1-"
        "flow-structure-only-graph-seed-5"
    )


def test_generated_hsp_uses_expanded_60_capture_evaluation():
    protocol = load_structure_ablation_protocol(PROTOCOL)
    jobs = build_jobs(
        protocol,
        selected_protocols=("generated_hsp",),
    )

    assert len(jobs) == 80
    assert {job["expected_window_count"] for job in jobs} == {60}
    assert {job["expected_dataset"] for job in jobs} == {
        "generated_hsp_expanded"
    }
    assert {job["result_path"].name for job in jobs} == {
        "generated_hsp_expanded_metrics.json"
    }
    assert {job["module"] for job in jobs} == {
        (
            "models.proposed."
            "evaluate_gi_hsp_v2_generated_hsp_expanded"
        )
    }
    assert {job["processing_protocol_path"] for job in jobs} == {
        (
            "configs/"
            "gi_hsp_v2_generated_hsp_expanded_processing.yaml"
        )
    }

