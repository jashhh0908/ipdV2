"""Hugging Face Transformers text-generation backend.

Loads one causal LM (default: Qwen2.5-3B-Instruct) and returns raw generated
text for batches of prompts. Loading (``HFGenerator.load``) is separate from
generation (``HFGenerator.generate``). No prompt design, output parsing, or
experiment logic lives here.

``torch``/``transformers`` are imported lazily, so the configs and helpers can
be used and tested without them installed.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Any, Iterator, Sequence, Union

__all__ = [
    "DEFAULT_MODEL_ID",
    "ModelConfig",
    "GenerationParams",
    "HFGenerator",
    "Prompt",
    "batched",
]

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DTYPES = ("float16", "bfloat16", "float32")

# A prompt is either a plain user message or a full chat (list of
# {"role": ..., "content": ...} dicts) for callers that need a system message.
Prompt = Union[str, Sequence[dict]]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class ModelConfig:
    """Which model to load, where, and how prompts are formatted.

    Defaults target a single Kaggle T4: fp16 (the T4 has no native bf16) on
    ``cuda:0``. Set ``revision`` to a commit hash to pin exact weights.
    With ``use_chat_template=True`` each prompt is wrapped in the tokenizer's
    chat template; note Qwen2.5's template inserts a default system message
    when the prompt has none.
    """

    model_id: str = DEFAULT_MODEL_ID
    revision: str | None = None
    device: str = "cuda:0"
    dtype: str = "float16"
    use_chat_template: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if self.revision is not None and not isinstance(self.revision, str):
            raise ValueError("revision must be a string or None")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {self.dtype!r}")
        if not isinstance(self.use_chat_template, bool):
            raise ValueError("use_chat_template must be bool")


@dataclass(frozen=True)
class GenerationParams:
    """Decoding parameters for one ``generate`` call.

    Greedy by default. ``temperature``/``top_p``/``top_k`` only apply when
    ``do_sample=True`` (``top_k=0`` disables top-k). ``seed`` is set once at
    the start of each ``generate`` call; results are reproducible for the same
    prompts, params (including ``batch_size``), model, and hardware.
    """

    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    seed: int = 0
    batch_size: int = 8

    def __post_init__(self) -> None:
        if not _is_int(self.max_new_tokens) or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be an int >= 1")
        if not isinstance(self.do_sample, bool):
            raise ValueError("do_sample must be bool")
        if not _is_num(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not _is_num(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not _is_int(self.top_k) or self.top_k < 0:
            raise ValueError("top_k must be an int >= 0")
        if not _is_num(self.repetition_penalty) or self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if not _is_int(self.seed) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an int in [0, 2**32)")
        if not _is_int(self.batch_size) or self.batch_size < 1:
            raise ValueError("batch_size must be an int >= 1")

    def to_hf_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``model.generate``.

        Sampling-only fields are omitted for greedy decoding so transformers
        does not warn about (or silently apply) unused settings.
        """
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": float(self.repetition_penalty),
        }
        if self.do_sample:
            kwargs.update(
                temperature=float(self.temperature),
                top_p=float(self.top_p),
                top_k=self.top_k,
            )
        return kwargs


def batched(items: Sequence[Any], size: int) -> Iterator[list[Any]]:
    """Yield consecutive chunks of ``items`` of at most ``size``, in order."""
    if not _is_int(size) or size < 1:
        raise ValueError("size must be an int >= 1")
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _dtype_kwarg(torch_dtype: Any) -> dict[str, Any]:
    """``from_pretrained`` dtype kwarg: renamed ``torch_dtype`` -> ``dtype`` in transformers 4.56."""
    import transformers
    from packaging.version import Version

    new = Version(transformers.__version__) >= Version("4.56")
    return {"dtype" if new else "torch_dtype": torch_dtype}


class HFGenerator:
    """Batched raw-text generation with a loaded Hugging Face causal LM.

    Build with ``HFGenerator.load(ModelConfig(...))``. The constructor takes an
    already-loaded model/tokenizer (useful for tests or custom loading); the
    tokenizer must pad on the left.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: ModelConfig,
        *,
        device_name: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device_name = device_name

    @classmethod
    def load(cls, config: ModelConfig | None = None) -> "HFGenerator":
        """Download/load tokenizer and model onto ``config.device`` in eval mode."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import GenerationConfig as HFGenerationConfig

        config = config or ModelConfig()
        is_cuda = config.device.startswith("cuda")
        if is_cuda and not torch.cuda.is_available():
            raise RuntimeError(f"device {config.device!r} requested but CUDA is not available")

        tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision)
        tokenizer.padding_side = "left"  # decoder-only batching
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            **_dtype_kwarg(getattr(torch, config.dtype)),
        )
        model.to(config.device)
        model.eval()

        # Replace the checkpoint's generation defaults (Qwen2.5-Instruct ships
        # temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05) with
        # neutral ones, so only GenerationParams controls decoding.
        shipped = model.generation_config
        model.generation_config = HFGenerationConfig(
            bos_token_id=shipped.bos_token_id,
            eos_token_id=shipped.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        device_name = torch.cuda.get_device_name(torch.device(config.device)) if is_cuda else "cpu"
        return cls(model, tokenizer, config, device_name=device_name)

    @property
    def backbone(self) -> str:
        """Backbone identifier for ``Extraction.backbone`` / ``MergeEvent.backbone``."""
        return self.config.model_id

    def format_prompts(self, prompts: Sequence[Prompt]) -> list[str]:
        """Render prompts to the exact strings fed to the tokenizer."""
        if isinstance(prompts, str):
            raise TypeError("prompts must be a list of prompts, not a single string")
        if not self.config.use_chat_template:
            if not all(isinstance(p, str) for p in prompts):
                raise TypeError("chat-message prompts require use_chat_template=True")
            return list(prompts)
        rendered = []
        for p in prompts:
            messages = [{"role": "user", "content": p}] if isinstance(p, str) else list(p)
            rendered.append(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        return rendered

    def generate(
        self, prompts: Sequence[Prompt], params: GenerationParams | None = None
    ) -> list[str]:
        """Generate one raw completion per prompt, in input order.

        Returns only the newly generated text (prompt excluded, special tokens
        stripped); no parsing or post-processing.
        """
        params = params or GenerationParams()
        texts = self.format_prompts(prompts)
        if not texts:
            return []

        import torch
        from transformers import set_seed

        set_seed(params.seed)
        hf_kwargs = params.to_hf_kwargs()
        outputs: list[str] = []
        with torch.inference_mode():
            for batch in batched(texts, params.batch_size):
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    # the chat template already contains the special tokens
                    add_special_tokens=not self.config.use_chat_template,
                ).to(self.model.device)
                out = self.model.generate(**enc, **hf_kwargs)
                new_tokens = out[:, enc["input_ids"].shape[1] :]
                outputs.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return outputs

    def run_info(self, params: GenerationParams | None = None) -> dict[str, Any]:
        """JSON-serializable provenance for a generation run with ``params``."""
        params = params or GenerationParams()
        template = getattr(self.tokenizer, "chat_template", None)
        return {
            "backbone": self.backbone,
            "model_config": asdict(self.config),
            "resolved_revision": getattr(getattr(self.model, "config", None), "_commit_hash", None),
            "generation_params": asdict(params),
            "chat_template_sha256": (
                hashlib.sha256(template.encode("utf-8")).hexdigest()
                if self.config.use_chat_template and isinstance(template, str)
                else None
            ),
            "device_name": self.device_name,
            "torch_version": _package_version("torch"),
            "transformers_version": _package_version("transformers"),
        }
