
from label_mapper_base import LabelMapperBase


class SnliLabelMapper(LabelMapperBase):
    """
    Maps string labels to integers for DBPedia dataset.
    """

    def __init__(self):
        self._raw_labels = ["neutral", "entailment", "contradiction"]
        self._map = {v: i for i, v in enumerate(self._raw_labels)}

        self._reverse_map = {i: v for i, v in enumerate(self._raw_labels)}

    def map(self, item) -> int:
        return self._map[item]

    def reverse_map(self, item: int):
        return self._reverse_map[item]

    @property
    def num_classes(self) -> int:
        return len(self._reverse_map)

    @property
    def positive_label(self):
        return self.reverse_map(1)

    @property
    def positive_label_index(self) -> int:
        return self.map(self.positive_label)
