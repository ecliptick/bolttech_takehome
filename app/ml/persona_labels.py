"""Canonical insurance-claimant persona labels (aligned with Notebook 01 / LLM batching)."""

from __future__ import annotations

PERSONA_CANONICAL: tuple[str, ...] = (
    "The Victim (Theft/Loss)",
    "The Unlucky Spiller (Liquid Damage)",
    "The Active/Sporty (Action Damage)",
    "The Family/Pet Owner (Chaos Damage)",
    "The Commuter (Transit Damage)",
    "The Professional (Workplace Accident)",
    "The Clumsy Dropper (Standard Accidental)",
)

DEFAULT_PERSONA = PERSONA_CANONICAL[-1]


def normalize_persona_label(raw: object) -> str:
    """Map LLM/rule drift (missing parentheses, casing, extra words) to ``PERSONA_CANONICAL``."""
    if raw is None:
        return DEFAULT_PERSONA
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return DEFAULT_PERSONA
    cf = s.casefold()

    for label in PERSONA_CANONICAL:
        if cf == label.casefold():
            return label

    heads: list[tuple[str, str]] = []
    inners: list[tuple[str, str]] = []
    for label in PERSONA_CANONICAL:
        if "(" in label:
            head, _, tail = label.partition("(")
            head_cf = head.strip().casefold()
            inner = tail.rstrip(")").strip().casefold()
            heads.append((head_cf, label))
            if inner:
                inners.append((inner, label))
        else:
            heads.append((label.casefold(), label))

    for head_cf, label in sorted(heads, key=lambda h: len(h[0]), reverse=True):
        if cf == head_cf or cf.startswith(head_cf + " "):
            return label

    for inner, label in sorted(inners, key=lambda t: len(t[0]), reverse=True):
        if inner and inner in cf:
            return label

    hints: tuple[tuple[tuple[str, ...], str], ...] = (
        (("theft", "loss", "victim"), PERSONA_CANONICAL[0]),
        (("liquid", "spiller"), PERSONA_CANONICAL[1]),
        (("sporty", "action damage", "sport", "gym"), PERSONA_CANONICAL[2]),
        (("family", "pet owner", "chaos"), PERSONA_CANONICAL[3]),
        (("commuter", "transit", "train", "commute"), PERSONA_CANONICAL[4]),
        (("professional", "workplace", "office"), PERSONA_CANONICAL[5]),
        (("clumsy", "dropper", "standard accidental"), PERSONA_CANONICAL[6]),
    )
    best: str | None = None
    best_score = 0
    for keys, lbl in hints:
        score = sum(1 for k in keys if k in cf)
        if score > best_score:
            best_score = score
            best = lbl
    if best is not None and best_score >= 1:
        return best
    return DEFAULT_PERSONA
