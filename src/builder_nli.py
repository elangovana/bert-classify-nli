import logging
import os

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from bert_model import BertModel
from bert_train import Train
from preprocessor_nli_bert_tokeniser import PreprocessorNliBertTokeniser
from snli_dataset import SnliDataset
from snli_dataset_label_mapper import SnliLabelMapper


class BuilderNli:

    def __init__(self, train_data, val_data, model_dir, num_workers=None, checkpoint_dir=None, epochs=10,
                 early_stopping_patience=10, checkpoint_frequency=1, grad_accumulation_steps=8, batch_size=8,
                 max_seq_len=512, learning_rate=0.00001, fine_tune=True):
        self.model_dir = model_dir
        self.fine_tune = fine_tune
        self.learning_rate = learning_rate
        self.checkpoint_frequency = checkpoint_frequency
        self.grad_accumulation_steps = grad_accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        # Note: Since the max seq len for pos embedding is 512 , in the pretrained  bert this must be less than eq to 512
        # Also note increasing the length greater also will create GPU out of mememory error
        self._max_seq_len = max_seq_len
        if num_workers is None:
            self.num_workers = os.cpu_count() - 1
        else:
            self.num_workers = num_workers

        if self.num_workers <= 0:
            self.num_workers = 0

        self._network = None
        self._train_dataloader = None
        self._train_dataset = None
        self._val_dataset = None
        self._val_dataloader = None
        self._trainer = None
        self._lossfunc = None
        self._optimiser = None
        self._label_mapper = None

        self._bert_model_name = "bert-base-cased"
        self._token_lower_case = False

        self._bert_config = None
        self._tokenisor = None

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def set_bert_config(self, value):
        self._bert_config = value

    def set_tokensior(self, value):
        self._tokenisor = value

    def get_preprocessor(self):
        self._logger.info("Retrieving Tokeniser")

        if self._tokenisor is None:
            self._tokenisor = BertTokenizer.from_pretrained(self._bert_model_name, do_lower_case=self._token_lower_case)

        preprocessor = PreprocessorNliBertTokeniser(max_feature_len=self._max_seq_len, tokeniser=self._tokenisor)
        self._logger.info("Completed retrieving Tokeniser")

        return preprocessor

    def get_network(self):
        # If network already loaded simply return
        if self._network is not None: return self._network

        self._logger.info("Retrieving model")

        # If checkpoint file is available, load from checkpoint
        state_dict = self.get_trainer().try_load_statedict_from_checkpoint()

        self._network = BertModel(self._bert_model_name, self.get_label_mapper().num_classes,
                                  fine_tune=self.fine_tune, bert_config=self._bert_config)

        if state_dict is not None:
            # Only load from BERT pretrained when no checkpoint is available
            self._logger.info("checkpoint models found")
            self._network.load_state_dict(state_dict)

        self._logger.info("Retrieving model complete")

        return self._network

    def get_train_dataset(self):
        if self._train_dataset is None:
            self._train_dataset = SnliDataset(self.train_data, preprocessor=self.get_preprocessor())

        return self._train_dataset

    def get_val_dataset(self):
        if self._val_dataset is None:
            self._val_dataset = SnliDataset(self.val_data, preprocessor=self.get_preprocessor())

        return self._val_dataset

    def get_label_mapper(self):
        if self._label_mapper is None:
            self._label_mapper = SnliLabelMapper()

        return self._label_mapper

    def get_pos_label_index(self):
        return self.get_label_mapper().positive_label_index

    def get_train_val_dataloader(self):
        if self._train_dataloader is None:
            self._train_dataloader = DataLoader(dataset=self.get_train_dataset(), num_workers=self.num_workers,
                                                batch_size=self.batch_size, shuffle=True)

        if self._val_dataloader is None:
            self._val_dataloader = DataLoader(dataset=self.get_val_dataset(), num_workers=self.num_workers,
                                              batch_size=self.batch_size, shuffle=False)

        return self._train_dataloader, self._val_dataloader

    def get_loss_function(self):
        if self._lossfunc is None:
            self._lossfunc = nn.CrossEntropyLoss()
        return self._lossfunc

    def get_optimiser(self):
        if self._optimiser is None:
            self._optimiser = Adam(params=self.get_network().parameters(), lr=self.learning_rate)
        return self._optimiser

    def get_trainer(self):
        if self._trainer is None:
            self._trainer = Train(model_dir=self.model_dir, epochs=self.epochs,
                                  early_stopping_patience=self.early_stopping_patience,
                                  checkpoint_frequency=self.checkpoint_frequency,
                                  checkpoint_dir=self.checkpoint_dir,
                                  accumulation_steps=self.grad_accumulation_steps)

        return self._trainer
