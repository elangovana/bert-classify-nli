import csv
import json
import logging

from torch.utils.data import Dataset

from imdb_dataset_label_mapper import ImdbLabelMapper
from snli_dataset_label_mapper import SnliLabelMapper


class SnliDataset(Dataset):
    """
    The snli  dataset as formatted by huggingface dataset
    """

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __init__(self, json_file: str, preprocessor=None):
        self.preprocessor = preprocessor
        self._file = json_file
        self._label_mapper = SnliLabelMapper()

        with open(json_file) as f:
            data = json.load(f)

            self._items = data["rows"]

        self.logger.info("Loaded {} records from the dataset".format(len(self)))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        row = self._items[idx]
        x_prem, x_hype, y_raw = row["row"]["premise"],row["row"]["hypothesis"], row["row"]["label"]

        # The original label is 1 indexed, this needs to be converted to zero index
        y = self._label_mapper.map(y_raw)

        x = (x_prem, x_hype)
        if self.preprocessor:
            x = self.preprocessor((x_prem, x_hype))

        return x, y
