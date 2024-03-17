import torch


class PreprocessorNliBertTokeniser:
    """
    Text to an array of indices using the BERT tokeniser
    """

    def __init__(self, max_feature_len, tokeniser):
        self.max_feature_len = max_feature_len
        self.tokeniser = tokeniser

    @staticmethod
    def pad_token():
        return "[PAD]"

    @staticmethod
    def eos_token():
        return "<EOS>"

    @staticmethod
    def unk_token():
        return "[UNK]"

    def __call__(self, item):
        prem, hyp = item[0], item[1]
        prem, hyp = self.tokenise(prem, hyp)
        item = self.sequence_pad(prem, hyp)
        item = self.token_to_index(item)
        item = self.to_tensor(item)

        return item

    def tokenise(self, prem, hyp):
        """
        Converts text to tokens, e.g. "The dog" would return ["The", "dog"]
        """
        prem = self.tokeniser.tokenize(prem)
        hyp = self.tokeniser.tokenize(hyp)
        return prem, hyp

    def token_to_index(self, item):
        """
        Converts a string of token to corresponding indices. e.g. ["The", "dog"] would return [2,3]
        :return: self
        """
        item = self.tokeniser.convert_tokens_to_ids(item)
        return item

    def sequence_pad(self, prem, hyp):
        """
        Converts the tokens to fixed size and formats it according to bert
        :return: self
        """
        prem = prem[:self.max_feature_len // 2 - 2]
        hyp = hyp[:self.max_feature_len // 2 - 1]
        token_len = len(prem) + len(hyp)
        pad_tokens = [self.pad_token()] * (self.max_feature_len - 3 - token_len)
        result = ['[CLS]'] + prem + ['[SEP]'] + hyp + pad_tokens + ['[SEP]']

        return result

    def to_tensor(self, item):
        """
        Converts list of int to tensor
        :return: self
        """

        return torch.tensor(item)
