"""Reasoning-effort normalization shared by local and API backends."""

from __future__ import annotations


# Five active tiers used by current reasoning-capable APIs. ``auto`` and
# ``off`` are separate controls rather than pretending that either is an
# effort tier.
REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")
REASONING_CHOICES = ("auto", "off", *REASONING_EFFORTS)

_LEGACY_CHOICES = {
    "backend_default": "auto",
    "thinking": "medium",
    "instruct": "off",
    "none": "off",
    "disabled": "off",
}

# Qwen3.8's official template accepts only low, medium and xhigh. Keep the
# universal five-tier UI while resolving unsupported local/API Qwen3.8 values
# to the nearest supported tier at or below the requested effort.
_QWEN38_EFFORT_MAP = {
    "low": "low",
    "medium": "medium",
    "high": "medium",
    "xhigh": "xhigh",
    "max": "xhigh",
}


def normalize_reasoning_choice(value: str | None) -> str:
    """Return a current reasoning choice while accepting old workflows."""
    choice = str(value or "auto").strip().lower()
    choice = _LEGACY_CHOICES.get(choice, choice)
    return choice if choice in REASONING_CHOICES else "auto"


def validate_reasoning_choice(value: str | None) -> bool | str:
    """Allow legacy API prompts through ComfyUI's combo validation."""
    if value is None or str(value).strip().lower() in {*REASONING_CHOICES, *_LEGACY_CHOICES}:
        return True
    return f"Unknown reasoning setting: {value}"


def is_qwen38_model(model: str | None) -> bool:
    """Best-effort model-id check for Qwen3.8's three native effort tiers."""
    compact = str(model or "").lower().replace("_", "").replace("-", "").replace(".", "")
    return "qwen38" in compact


def resolve_qwen38_effort(value: str | None) -> tuple[str, bool | None, str | None]:
    """Resolve a UI choice for Qwen3.8.

    Returns ``(choice, enabled, effective_effort)``. ``enabled`` is ``None``
    for auto so API backends can omit all reasoning fields. Local inference
    deliberately treats auto as its historical non-thinking behavior.
    """
    choice = normalize_reasoning_choice(value)
    if choice == "auto":
        return choice, None, None
    if choice == "off":
        return choice, False, None
    return choice, True, _QWEN38_EFFORT_MAP[choice]


def effective_reasoning_effort(value: str | None, model: str | None = None) -> str | None:
    """Return the effort sent on the wire, including Qwen3.8 clamping."""
    choice = normalize_reasoning_choice(value)
    if choice in {"auto", "off"}:
        return None
    if is_qwen38_model(model):
        return _QWEN38_EFFORT_MAP[choice]
    return choice
