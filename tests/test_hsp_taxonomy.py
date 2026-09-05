import unittest

from hsp.taxonomy import (
    load_hsp_taxonomy,
    validate_hsp_label,
)


class HSPTaxonomyTests(unittest.TestCase):
    def test_loads_taxonomy(self):
        taxonomy = load_hsp_taxonomy()

        self.assertEqual(
            taxonomy["definition"]["name"],
            "Host-Space Perturbation",
        )

    def test_accepts_benign_label(self):
        validate_hsp_label({
            "class": "benign",
            "attack_goal": "none",
            "hsp_family": "none",
        })

    def test_accepts_valid_attack_label(self):
        validate_hsp_label({
            "class": "attack",
            "attack_goal": "authentication",
            "hsp_family": "hydra",
        })

    def test_rejects_unknown_attack_goal(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unknown HsP attack goal",
        ):
            validate_hsp_label({
                "class": "attack",
                "attack_goal": "unknown",
                "hsp_family": "hydra",
            })

    def test_rejects_wrong_family_for_goal(self):
        with self.assertRaisesRegex(
            ValueError,
            "HsP family does not belong to attack goal",
        ):
            validate_hsp_label({
                "class": "attack",
                "attack_goal": "reconnaissance",
                "hsp_family": "hydra",
            })


if __name__ == "__main__":
    unittest.main()
