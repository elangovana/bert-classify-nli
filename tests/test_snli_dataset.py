
import os
from unittest import TestCase

from snli_dataset import SnliDataset


class TestSnliDataset(TestCase):
    def test___getitem__(self):
        input_file = os.path.join(os.path.dirname(__file__), "sample_data", "snli_train.jsonl")
        expected_y = 1
        expected_x_prem = "A person on a horse jumps over a broken down airplane."
        expected_x_hyp = """A person is outdoors, on a horse."""

        sut = SnliDataset(input_file)

        # Act

        (actual_x_prem, actual_x_hyp), actual_y = sut.__getitem__(2)

        # Assert
        self.assertEqual(expected_y, actual_y)
        self.assertEqual(expected_x_prem, actual_x_prem)
        self.assertEqual(expected_x_hyp, actual_x_hyp)

