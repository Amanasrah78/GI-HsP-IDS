import unittest

from models.proposed.label_encoding import (
    CLASS_TO_INDEX,
    decode_class_index,
    encode_class_label,
)


class LabelEncodingTests(unittest.TestCase):
    def test_binary_class_mapping(self):
        self.assertEqual(
            CLASS_TO_INDEX,
            {
                "benign": 0,
                "attack": 1,
            },
        )

    def test_encode_class_label(self):
        self.assertEqual(
            encode_class_label({"class": "benign"}),
            0,
        )
        self.assertEqual(
            encode_class_label({"class": "attack"}),
            1,
        )

    def test_decode_class_index(self):
        self.assertEqual(decode_class_index(0), "benign")
        self.assertEqual(decode_class_index(1), "attack")

    def test_rejects_unknown_label_class(self):
        with self.assertRaises(ValueError):
            encode_class_label({"class": "unknown"})

    def test_rejects_unknown_class_index(self):
        with self.assertRaises(ValueError):
            decode_class_index(2)


if __name__ == "__main__":
    unittest.main()
