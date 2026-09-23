import base64
import io
import json
import logging
from dataclasses import dataclass, field

import httpx
from PIL import Image

from . import taxonomy
from .taxonomy import Domain
from .constants import DEFAULT_MODEL, OLLAMA_URL, MODEL_TIERS, model_names_from_tags_response

_log = logging.getLogger(__name__)

@dataclass
class CaptionStats:
    """Aggregate Ollama call timing across a Captioner's lifetime, sourced from
    telemetry (total_duration, eval_count, eval_duration, load_duration,
    prompt_eval_count) that Ollama already returns on every /api/generate call.
    Previously this data was read only for done_reason and otherwise discarded —
    the hardcoded tier-speed estimates in constants.py had no measured data behind
    them despite this being available for free. Accumulates across every call a
    Captioner makes (the structured caption call, an optional OCR pass, and the
    plain-text fallback), so avg_seconds_per_call reflects real per-call cost
    including thorough=True's extra OCR call.
    """
    calls: int = 0
    total_duration_ns: int = 0
    total_eval_count: int = 0
    total_eval_duration_ns: int = 0
    total_load_duration_ns: int = 0
    total_prompt_eval_count: int = 0
    # Aggregate visibility into how often the model returns a field outside the
    # domain's known vocabulary (unrecognized setting/view/type) -- previously
    # each occurrence was only a per-item debug/info log line, with no run-level
    # signal that would surface a systematic problem (e.g. a domain whose
    # vocabulary the model consistently misses).
    captions_with_unknown_fields: int = 0
    total_unknown_fields: int = 0
    total_dropped_items: int = 0  # non-dict entries in the model's items array, discarded

    @property
    def avg_seconds_per_call(self) -> float:
        return (self.total_duration_ns / self.calls / 1e9) if self.calls else 0.0

    @property
    def tokens_per_second(self) -> float:
        return (
            (self.total_eval_count / (self.total_eval_duration_ns / 1e9))
            if self.total_eval_duration_ns else 0.0
        )


_OCR_PROMPT = (
    "List every piece of text legible in this image — identifiers, names, numbers, "
    "heralds, plates, signs, any lettering. "
    "One item per line. Transcribe exactly what you can read; do not guess or invent. "
    "If no text is legible, reply with nothing."
)


def _make_schema(domain: Domain) -> dict:
    """Build the JSON schema Ollama enforces for this domain's caption output."""
    item_props = {f: {"type": "string"} for f, _ in domain.item_fields}
    return {
        "type": "object",
        "properties": {
            domain.subject_field: {"type": "boolean"},
            "description": {"type": "string"},
            "setting": {"type": "string"},
            "era": {"type": "string"},
            "view": {"type": "string"},
            domain.items_field: {
                "type": "array",
                "items": {"type": "object", "properties": item_props},
            },
            "visible_text": {"type": "array", "items": {"type": "string"}},
        },
        "required": [domain.subject_field, "description"],
    }


def _make_prompt(domain: Domain) -> str:
    """Build the caption prompt for this domain."""
    frags = domain.prompt_fragments
    preamble = frags["preamble"]
    subject_qualifier = frags["subject_qualifier"]
    item_singular = frags["item_singular"]
    id_instruction = frags["id_instruction"]
    era_examples = frags["era_examples"]
    view_instruction = ", ".join(domain.views)
    type_note = frags.get("type_note", "")

    type_instruction = (
        f"Use exact terminology for `type` from this list when it applies: "
        f"{domain.subject_types_prompt()}."
    )
    if type_note:
        type_instruction = f"{type_instruction} {type_note}"

    return (
        f"{preamble} Analyze it for a searchable photo index and return JSON matching "
        "the schema.\n"
        f"- {domain.subject_field}: {subject_qualifier}.\n"
        "- description: plain-sentence description with specific detail. Name only what "
        "you can visually confirm; use 'appears to be' when uncertain. If this is not a "
        "matching photo, describe what it actually shows.\n"
        f"- {domain.items_field}: one entry per distinct {item_singular} visible. "
        f"{type_instruction} {id_instruction}\n"
        f"- setting: one of, or similar to: {domain.settings_prompt()}.\n"
        f"- era: approximate period if inferable (e.g. {era_examples}).\n"
        f"- view: camera perspective — one of: {view_instruction}.\n"
        "- visible_text: EVERY piece of text you can read anywhere in the image. "
        "Transcribe exactly what you can read; do not guess or invent."
    )


@dataclass
class CaptionResult:
    """Structured caption output. `caption` is the synthesized FTS text; the other
    fields map to dedicated store columns (see Store.upsert)."""
    caption: str
    description: str = ""
    is_railroad: bool = False   # semantically: "is_subject" — true if on-topic for the domain
    reporting_marks: str = ""   # flattened high-priority identifiers (FTS-weighted high)
    equipment: str = ""         # flattened subject types + class/road names (FTS-weighted mid)
    structured_json: str = ""   # raw model JSON, so nothing is ever silently dropped
    view: str = ""              # camera perspective (broadside, bow quarter, etc.)
    truncated: bool = False     # Ollama's done_reason=="length" -- caption was cut off
    unknown_field_count: int = 0  # settings/view/type values outside the domain vocabulary
    dropped_item_count: int = 0   # non-dict entries in the model's items array, discarded


_DEFAULT_TIMEOUT_S = 120.0
# constants.py documents the "quality" tier as running ~90-120s/photo -- a fixed
# 120s timeout applied to every tier left quality-tier calls with no headroom at
# all, so a call landing near that documented ceiling would time out and be
# treated as an ordinary caption failure (logged, not distinguished as an
# expected-slow-tier timeout) rather than actually failing.
_TIMEOUT_S_BY_TIER = {"quality": 240.0}


class Captioner:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_URL,
        domain: Domain = taxonomy.RAILROAD,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._domain = domain
        tier = MODEL_TIERS.get(model, "custom")
        timeout_s = _TIMEOUT_S_BY_TIER.get(tier, _DEFAULT_TIMEOUT_S)
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=5.0))
        self.stats = CaptionStats()

    @property
    def domain(self) -> Domain:
        """The domain this Captioner was constructed with -- public so callers
        (e.g. current_caption_version) can build a version string that changes
        when the domain does, without reaching into a private attribute."""
        return self._domain

    # -- public API ---------------------------------------------------------------

    def caption(self, image: Image.Image, thorough: bool = False) -> CaptionResult:
        """Caption an image, returning structured fields.

        Default is a single JSON-schema-constrained call. With `thorough=True`, a
        second dedicated OCR pass is merged in to maximize identifier recall.
        """
        b64 = self._encode(image)
        schema = _make_schema(self._domain)
        prompt = _make_prompt(self._domain)
        try:
            data = self._generate(prompt, b64, schema=schema)
            parsed = json.loads(data["response"])
            if not isinstance(parsed, dict):
                raise ValueError("model returned non-object JSON")
        except (json.JSONDecodeError, KeyError, ValueError, httpx.HTTPError) as e:
            _log.warning("Structured caption failed (%s); falling back to plain text", e)
            return self._plain_caption(b64)

        truncated = data.get("done_reason") == "length"
        if thorough:
            self._merge_ocr_pass(parsed, b64)

        result = self._build_result(parsed, self._domain, truncated=truncated)
        if result.unknown_field_count:
            self.stats.captions_with_unknown_fields += 1
            self.stats.total_unknown_fields += result.unknown_field_count
        self.stats.total_dropped_items += result.dropped_item_count
        return result

    def check(self) -> tuple[bool, str]:
        """Return (ok, message). Checks Ollama is running and model is available.

        Broad catch is intentional and matches the return contract: this method's
        entire purpose is to turn "is Ollama reachable" into a status tuple for the
        caller to display, never to raise — any failure to reach or parse the
        response means "not reachable," regardless of the specific exception type.
        The whole reach-and-parse sequence is inside one try (previously the JSON
        parse/field-access after raise_for_status() sat outside it, so malformed
        JSON or a models entry missing "name" raised uncaught instead of
        degrading per this docstring's own stated contract).
        """
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            models = model_names_from_tags_response(resp.json())
        except Exception:
            return False, f"Ollama not reachable at {self.base_url}"

        base = self.model.split(":")[0]
        # Accept exact match or the untagged pull (stored as :latest by Ollama).
        model_found = self.model in models or f"{base}:latest" in models
        if not model_found:
            available = ", ".join(models) or "none"
            return False, (
                f"Model '{self.model}' not found in Ollama. "
                f"Available: {available}. "
                f"Run: ollama pull {self.model}"
            )
        return True, "ok"

    def close(self) -> None:
        self._client.close()

    # -- internals ----------------------------------------------------------------

    def _encode(self, image: Image.Image) -> str:
        img = image.copy()
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    def _generate(self, prompt: str, b64: str, schema: dict | None = None) -> dict:
        body = {"model": self.model, "prompt": prompt, "images": [b64], "stream": False}
        if schema is not None:
            body["format"] = schema
        resp = self._client.post(f"{self.base_url}/api/generate", json=body)
        resp.raise_for_status()
        data = resp.json()
        self._record_stats(data)
        done_reason = data.get("done_reason")
        if done_reason == "length":
            _log.warning("Caption truncated at token limit (model=%s)", self.model)
        elif done_reason not in ("stop", None):
            _log.warning("Unexpected done_reason=%r (model=%s)", done_reason, self.model)
        return data

    def _record_stats(self, data: dict) -> None:
        """Accumulate Ollama's per-request timing telemetry into self.stats.
        Fields missing from the response (older Ollama versions, or any response
        shape that omits them) contribute zero rather than raising."""
        self.stats.calls += 1
        self.stats.total_duration_ns += data.get("total_duration") or 0
        self.stats.total_eval_count += data.get("eval_count") or 0
        self.stats.total_eval_duration_ns += data.get("eval_duration") or 0
        self.stats.total_load_duration_ns += data.get("load_duration") or 0
        self.stats.total_prompt_eval_count += data.get("prompt_eval_count") or 0

    def _plain_caption(self, b64: str) -> CaptionResult:
        """Old single-call behavior, used when structured parsing fails."""
        fallback_prompt = self._domain.prompt_fragments.get("fallback_preamble", "")
        try:
            data = self._generate(fallback_prompt, b64)
            # str(x or "") rather than data["response"].strip(): a non-string
            # response (e.g. None, or a type Ollama's schema-less plain-text
            # path doesn't guarantee) would otherwise raise AttributeError here
            # uncaught -- this is already the fallback path for a first failure,
            # so a second, different-shaped failure here must still degrade to
            # the documented empty-caption contract, not stack a third one.
            text = str(data["response"] or "").strip()
            truncated = data.get("done_reason") == "length"
        except (KeyError, httpx.HTTPError) as e:
            _log.warning("Plain caption also failed: %s", e)
            text = ""
            truncated = False
        return CaptionResult(caption=text, description=text, truncated=truncated)

    def _merge_ocr_pass(self, parsed: dict, b64: str) -> None:
        """Add a dedicated OCR pass's lines into parsed['visible_text'] (deduped)."""
        try:
            data = self._generate(_OCR_PROMPT, b64)
            # str(x or "") for the same reason as _plain_caption above: a
            # non-string response must degrade to "no OCR text found", not
            # raise AttributeError from an unguarded .splitlines().
            lines = [ln.strip(" -•\t") for ln in str(data["response"] or "").splitlines()]
        except (KeyError, httpx.HTTPError) as e:
            _log.warning("OCR pass failed: %s", e)
            return
        existing = parsed.get("visible_text") or []
        if not isinstance(existing, list):
            existing = []
        seen = {str(t).lower() for t in existing}
        for ln in lines:
            if ln and ln.lower() not in seen:
                existing.append(ln)
                seen.add(ln.lower())
        parsed["visible_text"] = existing

    def _build_result(self, parsed: dict, domain: Domain, truncated: bool = False) -> CaptionResult:
        description = str(parsed.get("description") or "").strip()
        setting = str(parsed.get("setting") or "").strip()
        era = str(parsed.get("era") or "").strip()
        view = str(parsed.get("view") or "").strip()
        is_subject = bool(parsed.get(domain.subject_field))
        unknown_field_count = 0

        # setting/view are drawn from bounded vocabularies (domain.settings / domain.views);
        # era is deliberately freeform (era_examples are illustrative, not exhaustive), so
        # it has no vocabulary to check against. Kept regardless — an out-of-vocabulary
        # value is model drift worth knowing about, not a reason to drop the field.
        if setting and setting.lower() not in {s.lower() for s in domain.settings}:
            _log.info("Unknown %s setting from model (kept): %r", domain.name, setting)
            unknown_field_count += 1
        if view and view.lower() not in {v.lower() for v in domain.views}:
            _log.info("Unknown %s view from model (kept): %r", domain.name, view)
            unknown_field_count += 1

        items = parsed.get(domain.items_field)
        items = items if isinstance(items, list) else []
        visible_text = parsed.get("visible_text")
        visible_text = [str(t).strip() for t in visible_text if str(t).strip()] \
            if isinstance(visible_text, list) else []

        mark_tokens: list[str] = []      # high-value identifiers (hull numbers, marks)
        equip_tokens: list[str] = []     # subject types + class/road names
        equip_phrases: list[str] = []    # human-readable per-item phrases for the caption
        # Lowercased once per call, not per item: domain.valid_subject_types holds the
        # dict's original-case keys, so comparing against it directly with a lowercased
        # `etype` silently never matched any mixed-case canonical type (e.g. motorsports'
        # "GT3 car", "NASCAR Cup car") -- every correctly-typed item in those domains was
        # misreported as "unknown" below, while an all-lowercase domain like railroad
        # happened to work by accident.
        valid_types_lower = {t.lower() for t in domain.valid_subject_types}
        dropped_item_count = 0

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                _log.debug("Skipping non-dict item at index %d in model output", idx)
                dropped_item_count += 1
                continue
            etype = str(item.get("type") or "").strip()
            if etype and etype.lower() not in valid_types_lower:
                _log.info("Unknown %s type from model (kept): %r", domain.name, etype)
                unknown_field_count += 1

            phrase_parts: list[str] = []
            for field_name, fts_weight in domain.item_fields:
                value = str(item.get(field_name) or "").strip()
                if not value:
                    continue
                if fts_weight == "high":
                    mark_tokens.append(value)
                elif fts_weight == "mid":
                    equip_tokens.append(value)
                phrase_parts.append(value)

            phrase = " ".join(phrase_parts)
            if phrase:
                equip_phrases.append(phrase)

        # visible_text is the OCR catch-all — every legible token, weighted as a mark.
        mark_tokens.extend(visible_text)

        caption = self._synthesize(description, equip_phrases, setting, era, view, visible_text)
        return CaptionResult(
            caption=caption,
            description=description,
            is_railroad=is_subject,
            reporting_marks=" ".join(dict.fromkeys(mark_tokens)),
            equipment=" ".join(dict.fromkeys(equip_tokens)),
            structured_json=json.dumps(parsed, ensure_ascii=False),
            view=view,
            truncated=truncated,
            unknown_field_count=unknown_field_count,
            dropped_item_count=dropped_item_count,
        )

    @staticmethod
    def _synthesize(
        description: str,
        equip_phrases: list[str],
        setting: str,
        era: str,
        view: str,
        visible_text: list[str],
    ) -> str:
        parts: list[str] = []
        if description:
            parts.append(description)
        if equip_phrases:
            parts.append("Equipment: " + "; ".join(equip_phrases) + ".")
        if setting:
            parts.append(f"Setting: {setting}.")
        if era:
            parts.append(f"Era: {era}.")
        if view:
            parts.append(f"View: {view}.")
        if visible_text:
            parts.append("Visible text: " + ", ".join(visible_text) + ".")
        return "\n".join(parts).strip()
