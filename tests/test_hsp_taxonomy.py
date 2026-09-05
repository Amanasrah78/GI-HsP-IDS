import unittest

from hsp.taxonomy import (
    load_hsp_taxonomy,
    validate_attack_provenance,
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


    def test_accepts_benign_without_attack_provenance(self):
        validate_attack_provenance(
            {
                "class": "benign",
                "attack_goal": "none",
                "hsp_family": "none",
            },
            {
                "attacker_container": "none",
                "attack_command": "none",
            },
        )

    def test_accepts_attack_with_provenance(self):
        validate_attack_provenance(
            {
                "class": "attack",
                "attack_goal": "reconnaissance",
                "hsp_family": "nmap",
            },
            {
                "attacker_container": "hsp-attacker",
                "attack_command": "nmap -sT 172.30.0.10",
            },
        )

    def test_rejects_attack_without_provenance(self):
        with self.assertRaisesRegex(
            ValueError,
            "requires attacker container provenance",
        ):
            validate_attack_provenance(
                {
                    "class": "attack",
                    "attack_goal": "reconnaissance",
                    "hsp_family": "nmap",
                },
                {
                    "attacker_container": "none",
                    "attack_command": "none",
                },
            )



if __name__ == "__main__":
    unittest.main()
