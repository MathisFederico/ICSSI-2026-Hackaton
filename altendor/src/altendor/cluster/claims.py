"""Haiku one-shot claim clustering (S14).

Each Endorsement carries a free-text ``claim_text`` extracted by the
classifier (S10). Multiple posts about the same paper will often endorse
semantically similar claims worded differently. This stage collapses them
into a small canonical set of "claims this paper makes" via a single
Claude Haiku tool call per paper.

The public entry point is :func:`cluster_claims`. Behaviour:

* If there are at most :data:`MIN_CLUSTERS` input claims, skip the LLM and
  return one cluster per input verbatim.
* Otherwise call Claude with a forced tool call asking for between
  :data:`MIN_CLUSTERS` and :data:`MAX_CLUSTERS` clusters.
* Post-hoc clamp the model's output: collapse to a single mega-cluster
  when too few are returned, merge the smallest into the largest when too
  many are returned, and rescue any input ids the model dropped by
  appending them to the smallest cluster.
* On any Anthropic API error, also fall back to the mega-cluster.

All errors are absorbed: this stage never raises — the worst case is the
mega-cluster, which preserves correctness at the cost of granularity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import anthropic

TOOL_NAME: str = "cluster_claims"
"""Name of the Anthropic tool S14 forces the model to call."""

DEFAULT_MODEL: str = "claude-haiku-4-5"
"""Cheap, fast default. Clustering paraphrases doesn't need Sonnet."""

DEFAULT_MAX_TOKENS: int = 1024
"""Output is a small JSON object; 1024 leaves slack for longer canonical texts."""

MIN_CLUSTERS: int = 3
"""Lower bound on returned cluster count after clamping."""

MAX_CLUSTERS: int = 7
"""Upper bound on returned cluster count after clamping."""

_MEGA_CLUSTER_TEXT: str = "Various claims about this paper."
"""Fallback canonical_text when we collapse to a single mega-cluster."""

_CANONICAL_TEXT_MAX_LEN: int = 300
"""Hard cap on canonical_text length after merging."""


@dataclass(frozen=True)
class ClaimCluster:
    """One canonical claim with the post ids endorsing paraphrases of it.

    Attributes
    ----------
    canonical_text:
        Neutral one-sentence restatement of the cluster's shared idea.
    member_post_ids:
        Subset of the input ``claim_texts`` keys whose paraphrases belong
        to this cluster.
    """

    canonical_text: str
    member_post_ids: list[str]


def _build_system_prompt() -> str:
    """Build the deterministic clustering system prompt."""
    return (
        "You group similar scientific-claim paraphrases together so a downstream UI\n"
        "can show one canonical claim per cluster instead of dozens of near-duplicate\n"
        "restatements.\n"
        "\n"
        "You will be given a list of claim paraphrases extracted from social-media\n"
        "posts about ONE paper. Each paraphrase has a stable id. Cluster them into\n"
        f"between {MIN_CLUSTERS} and {MAX_CLUSTERS} canonical claims based on semantic similarity. Every input\n"
        "id must appear in exactly one cluster. Each canonical_text is YOUR\n"
        "re-statement of the cluster's shared idea (one sentence, neutral phrasing).\n"
        "\n"
        f"Call the `{TOOL_NAME}` tool with the resulting clusters."
    )


def _build_user_message(claim_texts: dict[str, str]) -> str:
    """Render the per-paper user message listing every (id, claim_text) pair."""
    lines = ["## Claim paraphrases for this paper", ""]
    for post_id in sorted(claim_texts):
        # Newlines inside individual claim_texts would corrupt the bullet list;
        # collapse them so each paraphrase stays on its own line.
        text = claim_texts[post_id].replace("\n", " ").strip()
        lines.append(f"- {post_id}: {text}")
    lines.append("")
    lines.append("## Task")
    lines.append(f"Cluster these into {MIN_CLUSTERS} to {MAX_CLUSTERS} groups. Cover every id exactly once.")
    return "\n".join(lines)


def _build_tool_schema() -> dict[str, Any]:
    """JSON schema for the ``cluster_claims`` tool's ``input_schema``."""
    return {
        "type": "object",
        "properties": {
            "clusters": {
                "type": "array",
                "minItems": MIN_CLUSTERS,
                "maxItems": MAX_CLUSTERS,
                "items": {
                    "type": "object",
                    "properties": {
                        "canonical_text": {"type": "string", "minLength": 1},
                        "member_post_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                    },
                    "required": ["canonical_text", "member_post_ids"],
                },
            },
        },
        "required": ["clusters"],
    }


def _extract_tool_input(response: object) -> dict[str, Any]:
    """Pull the ``cluster_claims`` tool_use block's ``input`` from the Anthropic message."""
    content = getattr(response, "content", None)
    if not content:
        raise ValueError("Anthropic response had no content blocks.")
    for block in content:
        if getattr(block, "type", None) != "tool_use":
            continue
        if getattr(block, "name", None) != TOOL_NAME:
            continue
        payload = getattr(block, "input", None)
        if not isinstance(payload, dict):
            raise ValueError(f"tool_use block had non-dict input: {payload!r}")
        return cast("dict[str, Any]", payload)
    raise ValueError(f"No tool_use block named {TOOL_NAME!r} in Anthropic response.")


def _truncate(text: str) -> str:
    """Truncate a canonical_text to at most :data:`_CANONICAL_TEXT_MAX_LEN` chars."""
    if len(text) <= _CANONICAL_TEXT_MAX_LEN:
        return text
    return text[: _CANONICAL_TEXT_MAX_LEN - 3] + "..."


def _mega_cluster(claim_texts: dict[str, str]) -> list[ClaimCluster]:
    """Return the single-cluster fallback covering every input id."""
    return [
        ClaimCluster(
            canonical_text=_MEGA_CLUSTER_TEXT,
            member_post_ids=sorted(claim_texts),
        )
    ]


def _identity_clusters(claim_texts: dict[str, str]) -> list[ClaimCluster]:
    """One cluster per input claim, used when the input is too small to merit the LLM call."""
    return [
        ClaimCluster(canonical_text=claim_texts[post_id], member_post_ids=[post_id])
        for post_id in sorted(claim_texts)
    ]


def _parse_clusters(payload: dict[str, Any], valid_ids: set[str]) -> list[tuple[str, list[str]]]:
    """Pull the raw (canonical_text, member_post_ids) pairs out of the tool input.

    Filters member ids to those in ``valid_ids`` (the model may hallucinate),
    deduplicates ids within each cluster while preserving first-seen order,
    and drops clusters whose member list is empty after filtering.
    """
    raw_clusters = payload.get("clusters")
    if not isinstance(raw_clusters, list):
        raise ValueError(f"clusters field is not a list: {raw_clusters!r}")

    parsed: list[tuple[str, list[str]]] = []
    for raw in raw_clusters:
        if not isinstance(raw, dict):
            continue
        canonical = raw.get("canonical_text")
        members = raw.get("member_post_ids")
        if not isinstance(canonical, str) or not canonical.strip():
            continue
        if not isinstance(members, list):
            continue
        seen: set[str] = set()
        deduped: list[str] = []
        for m in members:
            if not isinstance(m, str):
                continue
            if m not in valid_ids or m in seen:
                continue
            seen.add(m)
            deduped.append(m)
        if not deduped:
            continue
        parsed.append((canonical.strip(), deduped))
    return parsed


def _merge_excess_clusters(clusters: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    """Merge the smallest cluster into the largest until ``len <= MAX_CLUSTERS``.

    The merged cluster keeps the largest cluster's index/position. Canonical
    text becomes ``"<largest>; <smallest>"`` (truncated to 300 chars) and
    member ids are unioned (largest's order first, then any new ids from the
    smallest in their original order).
    """
    while len(clusters) > MAX_CLUSTERS:
        # Find the smallest by member count (ties broken by current order).
        smallest_idx = min(range(len(clusters)), key=lambda i: (len(clusters[i][1]), i))
        smallest_text, smallest_members = clusters.pop(smallest_idx)

        # Find the largest among the remaining clusters.
        largest_idx = max(range(len(clusters)), key=lambda i: (len(clusters[i][1]), -i))
        largest_text, largest_members = clusters[largest_idx]

        merged_text = _truncate(f"{largest_text}; {smallest_text}")
        seen = set(largest_members)
        merged_members = list(largest_members)
        for m in smallest_members:
            if m not in seen:
                seen.add(m)
                merged_members.append(m)
        clusters[largest_idx] = (merged_text, merged_members)
    return clusters


def _attach_missing_ids(
    clusters: list[tuple[str, list[str]]],
    claim_texts: dict[str, str],
) -> list[tuple[str, list[str]]]:
    """Append any ids missing from every cluster to the smallest cluster.

    This must be called AFTER clamping so the smallest cluster is identified
    post-clamp.
    """
    covered: set[str] = set()
    for _, members in clusters:
        covered.update(members)
    missing = sorted(set(claim_texts) - covered)
    if not missing:
        return clusters

    smallest_idx = min(range(len(clusters)), key=lambda i: (len(clusters[i][1]), i))
    text, members = clusters[smallest_idx]
    seen = set(members)
    appended = list(members)
    for m in missing:
        if m not in seen:
            seen.add(m)
            appended.append(m)
    clusters[smallest_idx] = (text, appended)
    return clusters


def _sort_clusters(clusters: list[ClaimCluster]) -> list[ClaimCluster]:
    """Sort clusters by descending member count, ties broken by ``canonical_text`` ascending."""
    return sorted(clusters, key=lambda c: (-len(c.member_post_ids), c.canonical_text))


def cluster_claims(
    client: anthropic.Anthropic,
    claim_texts: dict[str, str],
    *,
    k_hint: int = 5,
    model: str = DEFAULT_MODEL,
) -> list[ClaimCluster]:
    """Cluster the provided ``claim_texts`` into 3..7 canonical claims via one Haiku call.

    Parameters
    ----------
    client:
        An ``anthropic.Anthropic`` client (or duck-typed stand-in exposing
        ``client.messages.create``).
    claim_texts:
        ``{post_id: claim_text}``. Every key is preserved across the
        returned clusters' ``member_post_ids``.
    k_hint:
        Advisory target cluster count. Currently unused by the prompt
        beyond the ``[MIN_CLUSTERS, MAX_CLUSTERS]`` window; reserved for
        future tuning without breaking the public signature.
    model:
        Anthropic model id. Defaults to :data:`DEFAULT_MODEL` (Haiku).

    Returns
    -------
    list[ClaimCluster]
        Clusters in descending order by ``len(member_post_ids)``, ties
        broken by ``canonical_text`` ascending for determinism.
    """
    del k_hint  # advisory only; prompt fixes the [MIN, MAX] window directly.

    if not claim_texts:
        return []

    if len(claim_texts) <= MIN_CLUSTERS:
        return _sort_clusters(_identity_clusters(claim_texts))

    system_text = _build_system_prompt()
    user_text = _build_user_message(claim_texts)
    tool_schema = _build_tool_schema()

    try:
        response = client.messages.create(
            model=model,
            max_tokens=DEFAULT_MAX_TOKENS,
            system=system_text,
            messages=[{"role": "user", "content": user_text}],
            tools=[
                {
                    "name": TOOL_NAME,
                    "description": (
                        f"Cluster the supplied claim paraphrases into between "
                        f"{MIN_CLUSTERS} and {MAX_CLUSTERS} canonical claims."
                    ),
                    "input_schema": tool_schema,
                }
            ],
            tool_choice={"type": "tool", "name": TOOL_NAME},
        )
        payload = _extract_tool_input(response)
        parsed = _parse_clusters(payload, valid_ids=set(claim_texts))
    except anthropic.AnthropicError:
        return _mega_cluster(claim_texts)
    except Exception:
        # Any malformed payload, missing tool_use block, etc. — same fallback.
        return _mega_cluster(claim_texts)

    if len(parsed) < MIN_CLUSTERS:
        return _mega_cluster(claim_texts)

    if len(parsed) > MAX_CLUSTERS:
        parsed = _merge_excess_clusters(parsed)

    parsed = _attach_missing_ids(parsed, claim_texts)

    clusters = [ClaimCluster(canonical_text=text, member_post_ids=members) for text, members in parsed]
    return _sort_clusters(clusters)


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "MAX_CLUSTERS",
    "MIN_CLUSTERS",
    "TOOL_NAME",
    "ClaimCluster",
    "cluster_claims",
]
