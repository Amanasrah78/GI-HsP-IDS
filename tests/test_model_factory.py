import unittest

from models.baselines.flow_only import FlowOnlyModel
from models.baselines.topology_only import TopologyOnlyModel
from models.proposed.gi_hsp_model import GIHSPModel
from models.proposed.model_factory import build_model


class ModelFactoryTests(unittest.TestCase):
    def make_config(self, name):
        return {
            "name": name,
            "flow_dim": 16,
            "topology_dim": 16,
            "fusion_dim": 16,
            "num_classes": 2,
            "dropout": 0.0,
        }

    def test_builds_gi_hsp_model(self):
        model = build_model(self.make_config("gi_hsp"))
        self.assertIsInstance(model, GIHSPModel)

    def test_builds_flow_only_model(self):
        model = build_model(self.make_config("flow_only"))
        self.assertIsInstance(model, FlowOnlyModel)

    def test_builds_topology_only_model(self):
        model = build_model(self.make_config("topology_only"))
        self.assertIsInstance(model, TopologyOnlyModel)


if __name__ == "__main__":
    unittest.main()
