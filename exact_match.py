import unicodedata
from typing import List, Tuple

def _normalize_drop_punct_ws(text: str) -> str:
    if text is None:
        return ""
    out_chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("P", "Z", "C"):
            continue
        out_chars.append(ch.lower())
    return "".join(out_chars)

def exact_match_no_punct(pred: str, ref: str) -> bool:
    return _normalize_drop_punct_ws(pred) == _normalize_drop_punct_ws(ref)

def em_compute(predictions: List[str], references: List[str]) -> dict:
    if len(predictions) != len(references):
        raise ValueError(f"Mismatch between predictions ({len(predictions)}) and references ({len(references)})")
    total = len(predictions)
    if total == 0:
        return 0.0, 0, 0
    matches = 0
    for p, r in zip(predictions, references):
        if exact_match_no_punct(p, r):
            matches += 1
    return {"em_ratio": matches / total, "matches": matches, "total": total}