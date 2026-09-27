"""
ollama_json.py
==============
One small helper shared by the post-extraction LLM passes (query-view briefs and
the strict violation adjudication): a schema-constrained Ollama chat that returns
a parsed dict, with reasoning mode off and a caller-chosen temperature.

Kept separate from hfacs_extractor._call_ollama on purpose. That function reads
module-level generation options pinned to temperature 0, which is right for the
committed extraction and wrong for consensus sampling, where temperature must vary
per call.
"""

import json
import logging

DEFAULT_MODEL = "qwen3.8:27b"


def chat_json(system: str, user: str, schema: dict, model: str = DEFAULT_MODEL,
              temperature: float = 0.0, num_ctx: int = 8192, num_predict: int = 400,
              seed: int | None = None, retries: int = 3) -> dict | None:
    """Schema-constrained chat -> dict, or None after `retries` failures."""
    import ollama
    options = {"temperature": float(temperature), "num_ctx": int(num_ctx),
               "num_predict": int(num_predict)}
    if seed is not None:
        options["seed"] = int(seed)
    kwargs = {"format": schema, "think": False}
    for attempt in range(1, retries + 1):
        try:
            resp = ollama.chat(model=model, options=options,
                               messages=[{"role": "system", "content": system},
                                         {"role": "user", "content": user}], **kwargs)
            return json.loads(resp["message"]["content"])
        except Exception as exc:                       # noqa: BLE001
            err = str(exc).lower()
            if "connection" in err or "refused" in err:
                raise SystemExit("Cannot reach Ollama. Start it with `ollama serve`.")
            if "think" in err and "think" in kwargs:   # model without a think switch
                kwargs.pop("think")
                continue
            logging.warning("chat_json attempt %d/%d failed: %s", attempt, retries, exc)
    return None


def head_tail(text: str, head: int, tail: int) -> str:
    """Keep the first `head` and last `tail` characters of a long narrative."""
    text = str(text or "")
    if len(text) <= head + tail:
        return text
    return text[:head] + "\n[...]\n" + text[-tail:]
