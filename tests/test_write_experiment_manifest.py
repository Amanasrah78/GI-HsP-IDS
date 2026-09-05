import subprocess
import sys
import unittest


class WriteExperimentManifestTests(unittest.TestCase):
    def test_cli_exposes_hsp_label_arguments(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/write_experiment_manifest.py",
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--class", result.stdout)
        self.assertIn("--attack-goal", result.stdout)
        self.assertIn("--hsp-family", result.stdout)


if __name__ == "__main__":
    unittest.main()
