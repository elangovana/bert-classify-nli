
class LabelMapperBase:
    """
    Base class for mapping labels to zero indexed integers
    """

    def map(self, item) -> int:
        """
        Maps the raw label to corresponding zero indexed integer. E.g. if the raw labels are "Positive" & "Negative", then the corresponding integers would be 0,1
        :param item: The raw label to map. e.g. "positive"
        :return: returns the corresponding zero indexed integer, e.g. 1
        """
        raise NotImplementedError

    def reverse_map(self, item: int):
        """
        Reverse maps the integer label to corresponding raw labels. E.g. if the integer labels are 0,1, then the corresponding raw labels are "Positive" & "Negative"
        :param item: The int label to map. e.g. 1
        :return: returns the corresponding raw label , e.g. "Positive"
        """
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        """
        The total number of unique classes. E.g. if you are performing sentiment analysis for positive, negative & neutral, then you would return 3
        :return: The total number of unique classes
        """
        raise NotImplementedError

    @property
    def positive_label(self):
        """
        The raw positive label. Useful for unbalanced dataset when you want to use F-score as the measure
        :return: The raw positive label , e.g. "positive"
        """
        raise NotImplementedError

    @property
    def positive_label_index(self) -> int:
        """
        The raw positive label index
        :return: The integer index corresponding to the raw positive_label
        """
        raise NotImplementedError
