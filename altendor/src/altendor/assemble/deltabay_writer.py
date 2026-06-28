"""Serialize :class:`IntermediateDebate` to disk (S17).

The writer is intentionally thin: pydantic owns the JSON shape (field order
and value coercion), and this module just commits the resulting bytes to a
file. Round-trippable via :meth:`IntermediateDebate.model_validate_json`.
"""

from __future__ import annotations

from pathlib import Path

from altendor.assemble.intermediate import IntermediateDebate


def write_debate_json(idb: IntermediateDebate, out_path: Path) -> None:
    """Write *idb* to *out_path* as pretty-printed UTF-8 JSON.

    Field order matches the pydantic model declaration (stable across runs).
    Parent directories are created if missing. A trailing newline is appended
    so the file is well-behaved under POSIX tooling.

    The output is round-trippable: ``IntermediateDebate.model_validate_json``
    on the bytes written here returns an equal model instance.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ``by_alias=False`` is the default; we use snake_case end-to-end and the
    # DeltaBay loader (S26) handles its own renaming to JSON-LD camelCase.
    payload = idb.model_dump_json(indent=2)
    if not payload.endswith("\n"):
        payload = payload + "\n"
    out_path.write_text(payload, encoding="utf-8")


__all__ = ["write_debate_json"]
