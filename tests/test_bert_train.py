
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock

import torch

from bert_train import Train


class TestBertTrain(TestCase):

    def test_run_train(self):
        """
        Test case  run train without exception
        :return:
        """
        tmp_dir = tempfile.mkdtemp()

        sut = Train(model_dir=tmp_dir, epochs=1)
        batch_size = 10
        sequence_len = 20
        vocab_size = 5
        num_classes = 3

        # Mock loss function to return a scalar value
        mock_loss = MagicMock()
        mock_loss.return_value = torch.tensor(0.0, requires_grad=True)

        # Mock model call for classification to return a tuple tensor that is shaped ([input_size, num_classes],)
        mock_network = MagicMock()
        mock_network.side_effect = lambda x: (torch.rand(size=(x.shape[0], num_classes)),)

        # Mock optimiser
        mock_optmiser = MagicMock()

        train = [self._generate_random_train_batch(batch_size, num_classes, sequence_len, vocab_size) for _ in
                 range(10)]
        val = [self._generate_random_train_batch(batch_size, num_classes, sequence_len, vocab_size) for _ in range(10)]

        # mock out pickling, to avoid raising pickling error for mock objects..
        sut.snapshot = MagicMock()
        sut.create_checkpoint = MagicMock()

        # Act
        actual = sut.run_train(train, val, loss_function=mock_loss, model_network=mock_network,
                               optimizer=mock_optmiser, pos_label=0)

        # Assert
        self.assertIsNotNone(actual)

    def _generate_random_train_batch(self, batch_size, num_classes, sequence_len, vocab_size):
        x = torch.randint(high=vocab_size, size=(batch_size, sequence_len))
        y = torch.randint(high=num_classes, size=(batch_size,))

        return x, y
