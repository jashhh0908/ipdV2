"""bge-m3 dense embeddings for Config D (embedding-threshold merge).

Loading (``BgeM3Embedder.load``) is separate from embedding
(``BgeM3Embedder.embed``). ``torch``/``transformers`` are imported lazily, so
the helpers and ``cosine_metric`` work (and are tested) without them.

The score is the cosine similarity of L2-normalised CLS embeddings (bge-m3's
dense mode). ``cosine_metric`` returns it as a ``SimilarityMetric``, which
requires scores in [0, 1] and never clips: a negative cosine (not expected for
natural-language sentence pairs) raises ``SimilarityError`` instead of being
hidden. It is a raw cosine, not a remapped ``(cos + 1) / 2``, so a threshold
tau means "cosine above tau".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

from dkmem.backends.llm import _dtype_kwarg, _package_version, batched
from dkmem.memory.similarity import SimilarityMetric

__all__ = [
    "DEFAULT_EMBEDDING_MODEL_ID",
    "EmbeddingConfig",
    "BgeM3Embedder",
    "cosine",
    "cosine_metric",
]

DEFAULT_EMBEDDING_MODEL_ID = "BAAI/bge-m3"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Which embedding model to load and how. ``revision`` pins exact weights."""

    model_id: str = DEFAULT_EMBEDDING_MODEL_ID
    revision: str | None = None
    device: str = "cuda:0"
    dtype: str = "float16"
    max_length: int = 512
    batch_size: int = 32

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if self.dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError(f"dtype must be float16, bfloat16 or float32, got {self.dtype!r}")
        for name in ("max_length", "batch_size"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be an int >= 1")


class BgeM3Embedder:
    """Dense bge-m3 embeddings. Build with ``BgeM3Embedder.load(EmbeddingConfig(...))``."""

    def __init__(self, model: Any, tokenizer: Any, config: EmbeddingConfig, *, device_name: str | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device_name = device_name

    @classmethod
    def load(cls, config: EmbeddingConfig | None = None) -> "BgeM3Embedder":
        import torch
        from transformers import AutoModel, AutoTokenizer

        config = config or EmbeddingConfig()
        is_cuda = config.device.startswith("cuda")
        if is_cuda and not torch.cuda.is_available():
            raise RuntimeError(f"device {config.device!r} requested but CUDA is not available")
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision)
        model = AutoModel.from_pretrained(
            config.model_id, revision=config.revision, **_dtype_kwarg(getattr(torch, config.dtype))
        )
        model.to(config.device)
        model.eval()
        device_name = torch.cuda.get_device_name(torch.device(config.device)) if is_cuda else "cpu"
        return cls(model, tokenizer, config, device_name=device_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One L2-normalised CLS embedding per text, in order."""
        import torch

        out: list[list[float]] = []
        with torch.inference_mode():
            for batch in batched(list(texts), self.config.batch_size):
                enc = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_length
                ).to(self.model.device)
                cls_vec = self.model(**enc).last_hidden_state[:, 0]
                cls_vec = torch.nn.functional.normalize(cls_vec.float(), p=2, dim=1)
                out.extend(cls_vec.cpu().tolist())
        return out

    def run_info(self) -> dict[str, Any]:
        """JSON-serializable provenance of the embedding model."""
        return {
            "model_config": asdict(self.config),
            "resolved_revision": getattr(getattr(self.model, "config", None), "_commit_hash", None),
            "pooling": "cls",
            "normalized": True,
            "device_name": self.device_name,
            "torch_version": _package_version("torch"),
            "transformers_version": _package_version("transformers"),
        }


def cosine(u: Sequence[float], v: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is all zeros)."""
    if len(u) != len(v):
        raise ValueError(f"vectors differ in length: {len(u)} vs {len(v)}")
    dot = sum(x * y for x, y in zip(u, v))
    nu, nv = sum(x * x for x in u) ** 0.5, sum(y * y for y in v) ** 0.5
    if not (nu and nv):
        return 0.0
    r = dot / (nu * nv)
    # floating-point noise can put the cosine of identical vectors a hair above 1
    return 1.0 if 1.0 < r <= 1.0 + 1e-6 else r


def cosine_metric(
    embed: Callable[[Sequence[str]], list[list[float]]], name: str
) -> SimilarityMetric:
    """A ``SimilarityMetric`` that embeds both texts with ``embed`` and returns their cosine.

    ``name`` should identify the model (e.g. ``"bge_m3_dense_cosine_v1"``).
    Embeddings are memoised per text, so repeated texts are embedded once.
    The score is symmetric by construction (cosine of the two vectors).
    """
    memo: dict[str, list[float]] = {}

    def vector(text: str) -> list[float]:
        if text not in memo:
            memo[text] = embed([text])[0]
        return memo[text]

    return SimilarityMetric(name, lambda a, b: cosine(vector(a), vector(b)))
