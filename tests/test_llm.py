"""Tests for dkmem.backends.llm that need no GPU or model download.

Run: python -m unittest discover -s tests
"""

import importlib.util
import json
import unittest

from dkmem.backends.llm import (
    DEFAULT_MODEL_ID,
    GenerationParams,
    HFGenerator,
    ModelConfig,
    batched,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_HF = HAS_TORCH and importlib.util.find_spec("transformers") is not None


class FakeChatTokenizer:
    """Records chat-template calls; renders messages as 'role:content|...'."""

    chat_template = "{{ fake template }}"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.calls.append((messages, tokenize, add_generation_prompt))
        return "|".join(f"{m['role']}:{m['content']}" for m in messages) + "|assistant:"


class TestModelConfig(unittest.TestCase):
    def test_defaults_target_qwen_on_t4(self):
        c = ModelConfig()
        self.assertEqual(c.model_id, DEFAULT_MODEL_ID)
        self.assertEqual(c.model_id, "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(c.dtype, "float16")
        self.assertEqual(c.device, "cuda:0")

    def test_invalid(self):
        for kw in (
            {"model_id": ""},
            {"dtype": "fp16"},
            {"device": ""},
            {"revision": 123},
            {"use_chat_template": 1},
        ):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                ModelConfig(**kw)


class TestGenerationParams(unittest.TestCase):
    def test_greedy_kwargs_omit_sampling_fields(self):
        kw = GenerationParams(max_new_tokens=32).to_hf_kwargs()
        self.assertEqual(
            kw, {"max_new_tokens": 32, "do_sample": False, "repetition_penalty": 1.0}
        )

    def test_sampling_kwargs(self):
        kw = GenerationParams(do_sample=True, temperature=0.7, top_p=0.9, top_k=20).to_hf_kwargs()
        self.assertEqual(kw["temperature"], 0.7)
        self.assertEqual(kw["top_p"], 0.9)
        self.assertEqual(kw["top_k"], 20)
        self.assertTrue(kw["do_sample"])

    def test_invalid(self):
        for kw in (
            {"max_new_tokens": 0},
            {"max_new_tokens": True},
            {"temperature": 0},
            {"top_p": 0},
            {"top_p": 1.5},
            {"top_k": -1},
            {"repetition_penalty": 0},
            {"seed": -1},
            {"seed": 2**32},
            {"seed": 1.0},
            {"batch_size": 0},
            {"do_sample": "yes"},
        ):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                GenerationParams(**kw)


class TestBatched(unittest.TestCase):
    def test_chunks_in_order(self):
        self.assertEqual(list(batched([1, 2, 3, 4, 5], 2)), [[1, 2], [3, 4], [5]])
        self.assertEqual(list(batched([], 3)), [])

    def test_invalid_size(self):
        with self.assertRaises(ValueError):
            list(batched([1], 0))


class TestFormatting(unittest.TestCase):
    def test_chat_template_wraps_strings_as_user_messages(self):
        tok = FakeChatTokenizer()
        gen = HFGenerator(model=None, tokenizer=tok, config=ModelConfig())
        out = gen.format_prompts(["hello", [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]])
        self.assertEqual(out, ["user:hello|assistant:", "system:s|user:u|assistant:"])
        self.assertTrue(all(c[1] is False and c[2] is True for c in tok.calls))

    def test_raw_mode_passes_strings_through(self):
        gen = HFGenerator(None, FakeChatTokenizer(), ModelConfig(use_chat_template=False))
        self.assertEqual(gen.format_prompts(["a", "b"]), ["a", "b"])
        with self.assertRaises(TypeError):
            gen.format_prompts([[{"role": "user", "content": "x"}]])

    def test_single_string_rejected(self):
        gen = HFGenerator(None, FakeChatTokenizer(), ModelConfig())
        with self.assertRaises(TypeError):
            gen.generate("not a list")

    def test_empty_prompts(self):
        gen = HFGenerator(None, FakeChatTokenizer(), ModelConfig())
        self.assertEqual(gen.generate([]), [])


class TestRunInfo(unittest.TestCase):
    def test_json_serializable_and_complete(self):
        gen = HFGenerator(None, FakeChatTokenizer(), ModelConfig(), device_name="Tesla T4")
        info = gen.run_info(GenerationParams(seed=3))
        json.dumps(info)
        self.assertEqual(info["backbone"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(gen.backbone, info["backbone"])
        self.assertEqual(info["generation_params"]["seed"], 3)
        self.assertEqual(info["model_config"]["dtype"], "float16")
        self.assertEqual(info["device_name"], "Tesla T4")
        self.assertEqual(len(info["chat_template_sha256"]), 64)

    def test_no_template_hash_in_raw_mode(self):
        gen = HFGenerator(None, FakeChatTokenizer(), ModelConfig(use_chat_template=False))
        self.assertIsNone(gen.run_info()["chat_template_sha256"])


@unittest.skipUnless(HAS_HF, "requires torch and transformers")
class TestGenerateWithFakeModel(unittest.TestCase):
    """Exercises the batching/decoding loop with a fake model (CPU, no download)."""

    def setUp(self):
        import torch

        class Enc(dict):
            def to(self, device):
                return self

        class Tok(FakeChatTokenizer):
            def __init__(self):
                super().__init__()
                self.batches = []

            def __call__(self, batch, return_tensors, padding, add_special_tokens):
                self.batches.append(list(batch))
                lens = [len(s) for s in batch]
                width = max(lens)
                ids = [[0] * (width - n) + [1] * n for n in lens]  # left padded
                return Enc(input_ids=torch.tensor(ids))

            def batch_decode(self, rows, skip_special_tokens):
                return [",".join(str(t) for t in r.tolist()) for r in rows]

        class Model:
            device = "cpu"

            def __init__(self):
                self.calls = []

            def generate(self, input_ids, **kwargs):
                self.calls.append(kwargs)
                n = input_ids.shape[0]
                new = torch.arange(n).unsqueeze(1).repeat(1, 2) + 7  # row i -> [7+i, 7+i]
                return torch.cat([input_ids, new], dim=1)

        self.tok, self.model = Tok(), Model()
        self.gen = HFGenerator(self.model, self.tok, ModelConfig(device="cpu"))

    def test_batched_generation_returns_only_new_tokens_in_order(self):
        out = self.gen.generate(["a", "bbb", "cc"], GenerationParams(batch_size=2, max_new_tokens=2))
        self.assertEqual(out, ["7,7", "8,8", "7,7"])
        self.assertEqual([len(b) for b in self.tok.batches], [2, 1])
        self.assertEqual(self.model.calls[0]["max_new_tokens"], 2)
        self.assertFalse(self.model.calls[0]["do_sample"])


if __name__ == "__main__":
    unittest.main()
