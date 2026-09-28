from __future__ import annotations

import json
import math
from pathlib import Path

from .contract import Cost, fingerprint


def load_pricing() -> dict:
    prices = json.loads(Path(__file__).with_name("pricing.json").read_text())
    for model, rate in prices.items():
        if model.startswith("_") or rate is None:
            continue
        if not (
            isinstance(rate, list)
            and len(rate) == 2
            and all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in rate)
        ):
            raise ValueError(f"Invalid input/output price for {model}")
    return prices


def cost_usd(
    prices: dict,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    """Uncached USD, or None when rates or billable usage are unknown."""
    rate = prices.get(model)
    if rate is None:
        return None
    if input_tokens is None and rate[0] == 0:
        input_tokens = 0
    if output_tokens is None and rate[1] == 0:
        output_tokens = 0
    if input_tokens is None or output_tokens is None:
        return None
    return (input_tokens * rate[0] + output_tokens * rate[1]) / 1e6


def estimated_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    prices: dict,
) -> Cost | None:
    usd = cost_usd(prices, model, input_tokens, output_tokens)
    return (
        None
        if usd is None
        else Cost(usd=usd, basis="estimated_uncached", pricing_revision=fingerprint(prices))
    )
