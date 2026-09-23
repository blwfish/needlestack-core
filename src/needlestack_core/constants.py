"""Lightweight, dependency-free constants.

Kept separate from captioner.py so the CLI can import the default model name without
pulling in PIL/httpx at startup (the CLI lazy-imports heavy modules inside commands).
Single source of truth for these values — every other module imports from here.
"""

DEFAULT_MODEL = "qwen2.5vl:7b"
OLLAMA_URL = "http://localhost:11434"

# Named model presets — convenience aliases over --model.
# fast:     low-RAM / CPU-only machines; good for type + identifier, approximate components
# balanced: default; good GPU/Apple-Silicon fit; rough estimate ~4s/photo
# quality:  high-VRAM machines; best component/armament detail; rough estimate ~90-120s/photo
#           — warn users
# These per-tier numbers are rough guesses, not measured. For real measured
# throughput on your hardware, see the "captioning: Xs/call" line `needlestack
# index` prints after a run (Captioner.stats, driven by Ollama's own per-request
# timing telemetry).
MODEL_PRESETS: dict[str, str] = {
    "fast":     "minicpm-v:latest",
    "balanced": "qwen2.5vl:7b",
    "quality":  "qwen3-vl:32b",
}

# Reverse map: model name → tier label (for doctor display)
MODEL_TIERS: dict[str, str] = {v: k for k, v in MODEL_PRESETS.items()}

# Bump whenever the caption PROMPT, the JSON schema, or caption synthesis changes in a
# way that should invalidate existing captions. Combined with the model name into the
# per-image caption_version so an upgrade auto-re-captions.
PROMPT_SCHEMA_VERSION = "v2"


def model_names_from_tags_response(data: dict) -> list[str]:
    """Extract model names from Ollama's /api/tags response body.

    Single source of truth so captioner.py's check() and doctor.py's Ollama
    reachability check can't independently drift on how they parse this
    (previously both used unguarded `m["name"]` dict indexing, duplicated).
    Uses .get() rather than direct indexing: a models entry missing "name", or
    a non-dict entry, is skipped rather than raising -- consistent with this
    being a best-effort diagnostic/availability check, not a hard schema
    contract with Ollama.
    """
    models = data.get("models", [])
    if not isinstance(models, list):
        return []
    return [m["name"] for m in models if isinstance(m, dict) and "name" in m]


def caption_version(model: str, domain: str) -> str:
    """Canonical per-image caption-version string. Single source of truth for the
    format so the indexer (which writes it) and the server (which counts staleness)
    never disagree on how model+schema+domain map to a version.

    `domain` is part of the version -- not just model+schema -- because a domain
    change (e.g. re-running `needlestack index <dir> --domain naval` on a
    directory previously indexed as railroad) changes what fields the model is
    asked to return and how they should be interpreted, but the file's hash and
    the model producing it may be unchanged. Without domain in the version, the
    indexer's skip-check ("already captioned under the current version") would
    treat every file as up to date and silently skip re-captioning, leaving
    stale railroad-domain structured data in a database that now reports itself
    as the naval domain for that root.
    """
    return f"{model}:{PROMPT_SCHEMA_VERSION}:{domain}"
