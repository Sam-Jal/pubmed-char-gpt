import unittest

import torch

from model import GPT


class GPTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = GPT(
            vocab_size=19,
            block_size=8,
            n_emb=16,
            n_head=4,
            n_layer=2,
            dropout=0.0,
        )
        self.model.eval()

    def test_forward_and_backward(self):
        inputs = torch.randint(0, 19, (2, 8))
        targets = torch.randint(0, 19, (2, 8))
        logits, loss = self.model(inputs, targets)

        self.assertEqual(logits.shape, (2, 8, 19))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.parameters()))

    def test_attention_is_causal(self):
        first = torch.tensor([[1, 2, 3, 4]])
        second = torch.tensor([[1, 8, 9, 10]])
        first_logits, _ = self.model(first)
        second_logits, _ = self.model(second)

        torch.testing.assert_close(first_logits[:, 0], second_logits[:, 0])

    def test_generation_length(self):
        context = torch.tensor([[1, 2]])
        output = self.model.generate(context, max_new_tokens=5, temperature=0.8)
        self.assertEqual(output.shape, (1, 7))

    def test_invalid_generation_temperature(self):
        with self.assertRaises(ValueError):
            self.model.generate(torch.tensor([[1]]), max_new_tokens=1, temperature=0)


if __name__ == "__main__":
    unittest.main()
