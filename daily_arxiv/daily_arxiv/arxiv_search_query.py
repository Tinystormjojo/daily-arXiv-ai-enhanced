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
        return f"(all:{w})"
    if mode == "phrase":
        q = phrase.replace('"', '\\"')
        return f'(all:"{q}")'
    if mode == "any_word":
        parts = [f"all:{_escape_atom_word(w)}" for w in words if _escape_atom_word(w)]
        if not parts:
            return ""
        return "(" + " OR ".join(parts) + ")"
    # all_words (default)
    parts = [f"all:{_escape_atom_word(w)}" for w in words if _escape_atom_word(w)]
    if not parts:
        return ""
    return "(" + " AND ".join(parts) + ")"


def combine_keyword_clauses(keywords: List[str], mode: str) -> str:
    """Comma-separated phrases -> OR between phrases."""
    clauses = []
    for k in keywords:
        c = phrase_to_api_clause(k, mode)
        if c:
            clauses.append(c)
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]
    return "(" + " OR ".join(clauses) + ")"


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
    date_yyyy_mm_dd: str | None,
) -> str:
    """
    Full search_query for one primary category.
    """
    cat = category.strip()
    parts = [f"cat:{cat}"]
    kw_expr = combine_keyword_clauses(keywords, match_mode)
    if kw_expr:
        parts.append(kw_expr)
    if date_yyyy_mm_dd:
        parts.append(submitted_date_clause(date_yyyy_mm_dd.strip()))
    return " AND ".join(parts)
