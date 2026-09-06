import unittest

import torch

from models.proposed.gated_fusion import GatedCrossViewFusion


class GatedCrossViewFusionTests(unittest.TestCase):
    def test_output_shape(self):
        model = GatedCrossViewFusion(
            flow_dim=32,
            topology_dim=48,
            fusion_dim=24,
            dropout=0.0,
        )

        flow = torch.randn(4, 32)
        topology = torch.randn(4, 48)

        output = model(flow, topology)

        self.assertEqual(
            tuple(output.shape),
            (4, 24),
        )

    def test_rejects_batch_mismatch(self):
        model = GatedCrossViewFusion(
            dropout=0.0,
        )

        flow = torch.randn(2, 64)
        topology = torch.randn(3, 64)

        with self.assertRaisesRegex(
            ValueError,
            "Flow and topology batch sizes must match",
        ):
            model(flow, topology)

    def test_flow_view_affects_output(self):
        model = GatedCrossViewFusion(
            fusion_dim=16,
            dropout=0.0,
        )
        model.eval()

        flow_a = torch.zeros(2, 64)
        flow_b = torch.ones(2, 64)
        topology = torch.randn(2, 64)

        with torch.no_grad():
            output_a = model(flow_a, topology)
            output_b = model(flow_b, topology)

        self.assertFalse(
            torch.allclose(output_a, output_b)
        )

    def test_topology_view_affects_output(self):
        model = GatedCrossViewFusion(
            fusion_dim=16,
            dropout=0.0,
        )
        model.eval()

        flow = torch.randn(2, 64)
        topology_a = torch.zeros(2, 64)
        topology_b = torch.ones(2, 64)

        with torch.no_grad():
            output_a = model(flow, topology_a)
            output_b = model(flow, topology_b)

        self.assertFalse(
            torch.allclose(output_a, output_b)
        )


    def test_can_return_gate_for_diagnostics(self):
        model = GatedCrossViewFusion(
            flow_dim=32,
            topology_dim=48,
            fusion_dim=24,
            dropout=0.0,
        )

        flow = torch.randn(4, 32)
        topology = torch.randn(4, 48)

        output, gate = model(
            flow,
            topology,
            return_gate=True,
        )

        self.assertEqual(tuple(output.shape), (4, 24))
        self.assertEqual(tuple(gate.shape), (4, 24))
        self.assertTrue(torch.all(gate >= 0.0))
        self.assertTrue(torch.all(gate <= 1.0))


if __name__ == "__main__":
    unittest.main()
