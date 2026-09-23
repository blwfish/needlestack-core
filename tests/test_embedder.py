"""Real (non-mocked) contract tests for Embedder.

Every consumer of Embedder in both this repo and needlestack always mocks the
class out (it requires loading a real CLIP model via torch/open_clip), so no
CI-run test ever verified the real class's actual behavior: output shape,
normalization, or that its declared `dim` class attribute matches what the
model genuinely produces. A future open_clip/model upgrade could silently
change any of these and every mocked test would stay green.

These tests construct the real class. Model weights are typically already
cached locally (via HuggingFace Hub) once anyone has run needlestack for real;
if they aren't cached and no network is available to fetch them, the session
fixture skips the whole module rather than failing the build outright.
"""
import numpy as np
import pytest
from PIL import Image

from needlestack_core.embedder import Embedder


@pytest.fixture(scope="module")
def real_embedder():
    try:
        e = Embedder()
    except Exception as exc:  # model weights unavailable (no cache, no network)
        pytest.skip(f"Real Embedder unavailable in this environment: {exc}")
    return e


def test_embed_text_returns_correct_shape(real_embedder):
    vec = real_embedder.embed_text("a red locomotive")
    assert vec.shape == (Embedder.dim,)


def test_embed_image_returns_correct_shape(real_embedder):
    img = Image.new("RGB", (64, 64), color=(200, 30, 30))
    vec = real_embedder.embed_image(img)
    assert vec.shape == (Embedder.dim,)


def test_dim_class_attribute_matches_real_output_shape(real_embedder):
    """Regression: Embedder.dim=512 was hardcoded and never checked against what
    the real model actually produces -- store.py trusts it as the single source
    of truth for the embedding column's shape."""
    text_vec = real_embedder.embed_text("a caboose")
    image_vec = real_embedder.embed_image(Image.new("RGB", (32, 32)))
    assert text_vec.shape[0] == Embedder.dim
    assert image_vec.shape[0] == Embedder.dim


def test_embed_text_is_l2_normalized(real_embedder):
    vec = real_embedder.embed_text("a steam locomotive on the mainline")
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-4)


def test_embed_image_is_l2_normalized(real_embedder):
    img = Image.new("RGB", (64, 64), color=(30, 80, 200))
    vec = real_embedder.embed_image(img)
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-4)


def test_embed_text_is_deterministic(real_embedder):
    a = real_embedder.embed_text("a yellow caboose")
    b = real_embedder.embed_text("a yellow caboose")
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_embed_text_different_inputs_produce_different_vectors(real_embedder):
    """Not a semantic-correctness test (that needs curated real photos this repo
    doesn't have) -- just proof the real model responds to its input at all,
    rather than e.g. a preprocessing bug that collapses every input to the same
    output vector."""
    a = real_embedder.embed_text("a steam locomotive")
    b = real_embedder.embed_text("a flock of seagulls at the beach")
    cosine = float(np.dot(a, b))
    assert cosine < 0.99


def test_embed_image_values_are_finite(real_embedder):
    """Regression guard for the unchecked-zero-norm-division finding: a
    degenerate normalization would produce NaN/Inf rather than crashing."""
    img = Image.new("RGB", (64, 64), color=(0, 0, 0))
    vec = real_embedder.embed_image(img)
    assert np.all(np.isfinite(vec))
