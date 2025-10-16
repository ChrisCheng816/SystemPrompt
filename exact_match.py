import unicodedata
from typing import List, Tuple

def _normalize_drop_punct_ws(text: str) -> str:
    """
    丢弃所有标点、空白/分隔、控制字符，保留字母数字和汉字等可见字符，
    并将结果小写。None 会视为空字符串。
    """
    if text is None:
        return ""
    out_chars = []
    for ch in text:
        # unicodedata.category 返回两字符类别，例如 'Lu','Ll','Nd','Zs','Pc','Po','Cc' 等
        cat = unicodedata.category(ch)
        # 丢弃以 P（Punctuation）、Z（Separator/空白）或 C（Other/控制）开头的字符
        if cat[0] in ("P", "Z", "C"):
            continue
        out_chars.append(ch.lower())
    return "".join(out_chars)

def exact_match_no_punct(pred: str, ref: str) -> bool:
    """单样本比较：归一化后是否完全相等"""
    return _normalize_drop_punct_ws(pred) == _normalize_drop_punct_ws(ref)

def em_compute(predictions: List[str], references: List[str]) -> dict:
    """
    计算 EM：
    返回 (em_ratio, matches, total)
    - em_ratio: float, 命中比例（0~1）
    - matches: int, 命中数
    - total: int, 样本总数
    要求 predictions 和 references 长度一致，否则抛错。
    """
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