
import os
from unittest import TestCase

from imdb_dataset import ImdbDataset
from snli_dataset import SnliDataset


class TestSnliDataset(TestCase):
    def test___getitem__(self):
        input_file = os.path.join(os.path.dirname(__file__), "sample_data", "snli_test.json")
        expected_y = 2
        expected_x_prem = "This church choir sings to the masses as they sing joyous songs from the book at a church."
        expected_x_hyp = """A choir singing at a baseball game."""

        sut = SnliDataset(input_file)

        # Act
        (actual_x_prem, actual_x_hyp), actual_y = sut.__getitem__(2)

        # Assert
        self.assertEqual(expected_y, actual_y)
        self.assertEqual(expected_x_prem, actual_x_prem)
        self.assertEqual(expected_x_hyp, actual_x_hyp)

