import numpy as np
import open_clip
import torch
from PIL import Image

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


def _check_finite(vec: np.ndarray, source: str) -> None:
    """A degenerate zero-norm feature vector divides to NaN/Inf silently -- this
    turns that into a loud, specific failure (caught by indexer.py's per-image
    try/except, same as any other embedding failure) instead of a corrupt
    embedding being stored and silently poisoning every future similarity score
    that touches it."""
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{source} produced non-finite values (degenerate zero-norm output)")


class Embedder:
    # Output dimensionality of MODEL_NAME/PRETRAINED. A class attribute (not just an
    # instance property) so consumers needing the shape — e.g. an empty result matrix —
    # can reference it without loading the CLIP model. Single source of truth: update
    # this if the model changes; nothing else should hardcode 512.
    dim: int = 512

    def __init__(self, model_name: str = MODEL_NAME, pretrained: str = PRETRAINED):
        self.device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(self.device).eval()

    def embed_image(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        result = features.cpu().numpy()[0]
        _check_finite(result, "embed_image")
        return result

    def embed_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        result = features.cpu().numpy()[0]
        _check_finite(result, "embed_text")
        return result
