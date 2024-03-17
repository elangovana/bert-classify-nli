
import csv
import logging

from torch.utils.data import Dataset

from imdb_dataset_label_mapper import ImdbLabelMapper


class ImdbDataset(Dataset):
    """
    Imdb  dataset
    """

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __init__(self, file: str, preprocessor=None):
        self.preprocessor = preprocessor
        self._file = file
        self._label_mapper = ImdbLabelMapper()

        with open(file) as f:
            reader = csv.reader(f, delimiter='\t', quotechar='"')
            # Skip header
            next(reader)
            self._items = [(r[1], r[0]) for r in reader]

        self.logger.info("Loaded {} records from the dataset".format(len(self)))

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        x, y_raw = self._items[idx]

        # The original label is 1 indexed, this needs to be converted to zero index
        y =  self._label_mapper.map(y_raw)

        if self.preprocessor:
            x = self.preprocessor(x)

        return x, y
