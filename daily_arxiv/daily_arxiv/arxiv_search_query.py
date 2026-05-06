"""
Build arXiv API search_query strings for export.arxiv.org/api/query.
Aligned with ai/aggregate_keywords.py phrase / all_words / any_word semantics.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import List


def parse_keywords(raw: str | None) -> List[str]:
    if not raw or not str(raw).strip():
        return []
    parts = re.split(r"[,，]", str(raw))
    return [p.strip() for p in parts if p.strip()]


def _escape_atom_word(word: str) -> str:
    """Avoid breaking API query; keep letters/digits/dot/hyphen."""
    w = word.strip()
    return re.sub(r"[^\w.\-]", "", w, flags=re.UNICODE) or w


def phrase_to_api_clause(phrase: str, mode: str) -> str:
    """
    One comma-separated phrase -> API sub-expression using ti/abs/all fields.
    """
    phrase = phrase.strip()
    if not phrase:
        return ""
    words = [t for t in phrase.lower().split() if t]
    if len(words) <= 1:
        w = _escape_atom_word(words[0]) if words else ""
        if not w:
            return ""
        return f"all:{w}"
    if mode == "phrase":
        q = phrase.replace('"', '\\"')
        return f'(all:"{q}")'
    if mode == "any_word":
        parts = [f"all:{_escape_atom_word(w)}" for w in words if _escape_atom_word(w)]
        if not parts:
            return ""
        return "(" + " OR ".join(parts) + ")"
    # all_words (default): 多词 AND，短语之间 OR 时再由 combine 包括号
    parts = [f"all:{_escape_atom_word(w)}" for w in words if _escape_atom_word(w)]
    if not parts:
        return ""
    return " AND ".join(parts)


def combine_keyword_clauses(keywords: List[str], mode: str) -> str:
    """逗号分隔的多个短语之间为 OR；含 AND 的子式在 OR 分支外加一层括号。"""
    clauses = []
    for k in keywords:
        c = phrase_to_api_clause(k, mode)
        if c:
            clauses.append(c)
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]
    wrapped = []
    for c in clauses:
        if " OR " in c or " AND " in c:
            wrapped.append(f"({c})")
        else:
            wrapped.append(c)
    return "(" + " OR ".join(wrapped) + ")"


def submitted_date_range(date_yyyy_mm_dd: str) -> tuple[str, str]:
    """
    arXiv submittedDate uses YYYYMMDDHHMM (UTC).
    Single calendar day in UTC.
    """
    dt = datetime.strptime(date_yyyy_mm_dd.strip(), "%Y-%m-%d")
    start = dt.strftime("%Y%m%d") + "0000"
    end = dt.strftime("%Y%m%d") + "2359"
    return start, end


def submitted_date_clause(date_yyyy_mm_dd: str) -> str:
    s, e = submitted_date_range(date_yyyy_mm_dd)
    return f"submittedDate:[{s}+TO+{e}]"


def build_search_query_for_category(
    category: str,
    keywords: List[str],
    match_mode: str,
) -> str:
    """
    Full search_query for one primary category (export.arxiv.org API).

    注意：截至 2026-05，在 search_query 中加入 submittedDate:[...] 区间会导致
    export API 返回 HTTP 500（与 cat/关键词组合与否无关）。当日范围改由蜘蛛在解析
    Atom 时用 entry 的 published（UTC 日期）与 ARXIV_CRAWL_DATE 比对过滤。
    官方文档仍见 https://info.arxiv.org/help/api/user-manual.html
    """
    cat = category.strip()
    parts = [f"cat:{cat}"]
    kw_expr = combine_keyword_clauses(keywords, match_mode)
    if kw_expr:
        if " OR " in kw_expr:
            parts.append(f"({kw_expr})")
        else:
            parts.append(kw_expr)
    return " AND ".join(parts)
