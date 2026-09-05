import json
import unittest
from pathlib import Path

from models.proposed.gi_hsp_model import GIHSPModel
from models.proposed.input_assembly import assemble_sequence_inputs
from models.proposed.tensor_conversion import sequence_inputs_to_tensors


class RealSequenceModelIntegrationTests(unittest.TestCase):
    def test_real_sequence_reaches_full_model(self):
        path = Path(
            "results/processed/"
            "benign-mqtt-e2e-013.sequences.jsonl"
        )

        if not path.exists():
            self.skipTest("Real processed sequence artifact is unavailable")

        with path.open("r", encoding="utf-8") as handle:
            record = json.loads(next(handle))

        assembled = assemble_sequence_inputs(record)
        tensors = sequence_inputs_to_tensors(assembled)

        model = GIHSPModel()
        model.eval()

        outputs = model(
            tensors["packet_features"],
            tensors["node_features"],
            tensors["adjacency"],
        )

        self.assertEqual(
            tuple(tensors["packet_features"].shape),
            (1, 10, 10),
        )
        self.assertEqual(
            tuple(tensors["node_features"].shape),
            (1, 10, 5, 7),
        )
        self.assertEqual(
            tuple(tensors["adjacency"].shape),
            (1, 10, 5, 5),
        )
        self.assertEqual(
            tuple(outputs["logits"].shape),
            (1, 2),
        )
        self.assertEqual(
            tuple(outputs["flow_embedding"].shape),
            (1, 64),
        )
        self.assertEqual(
            tuple(outputs["topology_embedding"].shape),
            (1, 64),
        )
        self.assertEqual(
            tuple(outputs["fused_embedding"].shape),
            (1, 64),
        )
        self.assertEqual(
            tensors["label"]["class"],
            "benign",
        )


if __name__ == "__main__":
    unittest.main()
