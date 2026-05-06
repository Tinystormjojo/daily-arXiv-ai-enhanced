"""
Filter AI-enhanced JSONL to at most 5 most keyword-relevant papers per primary category,
and write per-category aggregate summaries.

When KEYWORDS is set, the crawler already restricts papers via the arXiv API; this script
does not re-filter by title/abstract. It still ranks by keyword relevance for the top-K pick.

Category summaries optionally incorporate the latest prior day's summary for the same
category (from *_category_meta_*.json) for thematic continuity.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


TOP_K = 5
# 避免单次分类汇总上下文过长导致 API 失败
MAX_PAPERS_PER_CATEGORY_SUMMARY = 60


def parse_keywords(raw: str | None) -> List[str]:
    if not raw or not str(raw).strip():
        return []
    parts = re.split(r"[,，]", str(raw))
    return [p.strip() for p in parts if p.strip()]


def paper_search_blob(item: Dict[str, Any]) -> str:
    title = (item.get("title") or "").lower()
    summary = (item.get("summary") or "").lower()
    return f"{title} {summary}"


def _phrase_tokens(phrase: str) -> List[str]:
    """拆成非空词（用于 all_words / any_word）。"""
    return [t for t in phrase.lower().split() if t]


def phrase_matches_blob(blob: str, phrase: str, mode: str) -> bool:
    """
    mode:
      - phrase: 整段短语必须连续出现在 title+摘要 中（最严，易零命中）
      - all_words: 短语中的每个词都要出现（默认，适合英文「generative recommendation」）
      - any_word: 短语中任一词出现即可（较松）
    """
    phrase = phrase.strip()
    if not phrase:
        return False
    pl = phrase.lower()
    blob_l = blob.lower()
    words = _phrase_tokens(phrase)
    if len(words) <= 1:
        return words[0] in blob_l if words else False
    if mode == "phrase":
        return pl in blob_l
    if mode == "any_word":
        return any(w in blob_l for w in words)
    # all_words
    return all(w in blob_l for w in words)


def matches_keywords(
    item: Dict[str, Any], keywords: List[str], mode: str
) -> bool:
    """逗号分隔的多个短语之间为 OR：任一短语命中即匹配。"""
    if not keywords:
        return True
    blob = paper_search_blob(item)
    return any(phrase_matches_blob(blob, k, mode) for k in keywords)


def relevance_score(
    item: Dict[str, Any], keywords: List[str], mode: str
) -> int:
    if not keywords:
        return 0
    title = (item.get("title") or "").lower()
    summary = (item.get("summary") or "").lower()
    score = 0
    for phrase in keywords:
        pl = phrase.strip().lower()
        if not pl:
            continue
        words = _phrase_tokens(phrase)
        if mode == "phrase" or len(words) <= 1:
            score += title.count(pl) * 3 + summary.count(pl)
        else:
            for w in words:
                score += title.count(w) * 3 + summary.count(w)
    return score


def primary_category(item: Dict[str, Any]) -> str | None:
    cats = item.get("categories")
    if not cats:
        return None
    if isinstance(cats, list) and len(cats) > 0:
        return cats[0]
    if isinstance(cats, str):
        return cats
    return None


def _is_substantive_summary(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 20:
        return False
    if "无与关键词匹配" in t:
        return False
    if "生成失败" in t:
        return False
    return True


def load_latest_prior_category_summary(
    out_dir: str,
    current_date: str,
    lang_key: str,
    category: str,
    max_lookback_days: int = 90,
) -> Tuple[Optional[str], Optional[str]]:
    """
    从 current_date 往前找最近一天，其 category_meta 中该分区有可用 summary。
    返回 (summary, 该 meta 对应的日期 YYYY-MM-DD)。
    """
    try:
        anchor = datetime.strptime(current_date, "%Y-%m-%d").date()
    except ValueError:
        return None, None
    for i in range(1, max_lookback_days + 1):
        d = anchor - timedelta(days=i)
        prev = d.strftime("%Y-%m-%d")
        meta_f = os.path.join(out_dir, f"{prev}_category_meta_{lang_key}.json")
        if not os.path.isfile(meta_f):
            continue
        try:
            with open(meta_f, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        cat_info = (doc.get("categories") or {}).get(category)
        if not cat_info:
            continue
        s = (cat_info.get("summary") or "").strip()
        if _is_substantive_summary(s):
            return s, prev
    return None, None


def summarize_category(
    llm: ChatOpenAI,
    category: str,
    papers: List[Dict[str, Any]],
    keywords: List[str],
    language: str,
    prior_summary: Optional[str] = None,
    prior_summary_date: Optional[str] = None,
) -> str:
    lines = []
    for p in papers:
        tid = p.get("id", "")
        title = p.get("title", "")
        ai = p.get("AI") or {}
        tldr = ai.get("tldr") or ""
        lines.append(f"- id={tid} | {title}\n  TL;DR: {tldr}")

    body = "\n".join(lines)
    sys_msg = (
        "You write concise, accurate academic digests. "
        "Output only the summary text requested, no preamble."
    )
    kw_desc = ", ".join(keywords) if keywords else "(all papers in this category)"
    scope = (
        "These papers are already from the repository's keyword-targeted crawl (no extra text filter)."
        if keywords
        else "These are all papers in this category in today's batch."
    )
    human = f"""Repository keywords (comma-separated): {kw_desc}
arXiv primary category: {category}
Target language for your answer: {language}

{scope}
Write a single cohesive summary (2–4 short paragraphs) that synthesizes themes, methods, and trends across these papers.
If the list is empty, reply with one sentence saying there are no papers.

Papers:
{body}
"""
    if prior_summary and prior_summary_date:
        human += f"""

Below is the most recent prior summary for this same category (from {prior_summary_date}), when that day had data.
Use it only for continuity: relate emerging vs continuing themes, do not copy verbatim, and highlight what is new or different in today's papers relative to that narrative.

Prior summary:
{prior_summary}
"""
    messages = [SystemMessage(content=sys_msg), HumanMessage(content=human)]
    out = llm.invoke(messages)
    return (out.content or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to *_AI_enhanced_*.jsonl")
    args = parser.parse_args()

    path = args.data
    language = (os.environ.get("LANGUAGE") or "").strip() or "Chinese"
    keywords = parse_keywords(os.environ.get("KEYWORDS"))
    match_mode = (
        (os.environ.get("KEYWORD_MATCH_MODE") or "all_words").strip().lower()
    )
    if match_mode not in ("phrase", "all_words", "any_word"):
        print(
            f"Unknown KEYWORD_MATCH_MODE={match_mode}, using all_words",
            file=sys.stderr,
        )
        match_mode = "all_words"
    model_name = os.environ.get("MODEL_NAME", "deepseek-chat")

    if not os.path.isfile(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    stem = os.path.basename(path)
    if "_AI_enhanced_" in stem:
        date_part = stem.split("_AI_enhanced_")[0]
        lang_part = stem.split("_AI_enhanced_")[1].replace(".jsonl", "")
    else:
        date_part = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        lang_part = re.sub(r"[^\w\-]+", "_", language.strip() or "Chinese")

    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        cat = primary_category(item)
        if not cat:
            continue
        by_cat[cat].append(item)

    out_lines: List[Dict[str, Any]] = []
    meta_categories: Dict[str, Any] = {}

    llm: ChatOpenAI | None = None
    # 与爬虫一致：设置了 KEYWORDS 时论文已在检索阶段限定，此处不再按标题/摘要二次过滤
    crawl_prefiltered_by_keywords = bool(keywords)

    for cat in sorted(by_cat.keys()):
        papers = by_cat[cat]
        if crawl_prefiltered_by_keywords:
            matched = list(papers)
        else:
            matched = [p for p in papers if matches_keywords(p, keywords, match_mode)]

        if keywords and not matched:
            meta_categories[cat] = {
                "summary": "当日该分区无与关键词匹配的论文。",
                "matched_count": 0,
                "titles_in_category": [p.get("title", "") for p in papers],
                "continuity_from_date": None,
            }
            continue

        pool = matched if keywords else papers
        if keywords:
            ranked = sorted(
                pool,
                key=lambda p: (
                    relevance_score(p, keywords, match_mode),
                    p.get("id", ""),
                ),
                reverse=True,
            )
        else:
            ranked = list(pool)
        top = ranked[:TOP_K]
        out_lines.extend(top)

        to_summarize = matched if keywords else papers
        if not to_summarize:
            meta_categories[cat] = {
                "summary": "",
                "matched_count": 0,
                "titles_in_category": [p.get("title", "") for p in papers],
                "continuity_from_date": None,
            }
            continue

        if len(to_summarize) > MAX_PAPERS_PER_CATEGORY_SUMMARY:
            to_summarize = to_summarize[:MAX_PAPERS_PER_CATEGORY_SUMMARY]

        if llm is None:
            llm = ChatOpenAI(model=model_name)

        prior_text, prior_dt = load_latest_prior_category_summary(
            out_dir, date_part, lang_part, cat
        )

        try:
            summary_text = summarize_category(
                llm,
                cat,
                to_summarize,
                keywords,
                language,
                prior_summary=prior_text,
                prior_summary_date=prior_dt,
            )
        except Exception as e:
            print(f"Category summary failed for {cat}: {e}", file=sys.stderr)
            summary_text = "（分类汇总生成失败，请稍后重试或检查 API 配置。）"
        meta_categories[cat] = {
            "summary": summary_text,
            "matched_count": len(matched) if keywords else len(papers),
            "titles_in_category": [p.get("title", "") for p in papers],
            "continuity_from_date": prior_dt,
        }

    # e.g. 2025-01-01_AI_enhanced_Chinese.jsonl -> 2025-01-01_category_meta_Chinese.json
    if "_AI_enhanced_" in stem:
        meta_name = f"{date_part}_category_meta_{lang_part}.json"
    else:
        meta_name = stem.replace(".jsonl", "") + "_category_meta.json"

    meta_path = os.path.join(out_dir, meta_name)

    inventory = []
    for item in items:
        cat = primary_category(item)
        inventory.append(
            {
                "id": item.get("id"),
                "title": item.get("title"),
                "primary_category": cat,
            }
        )

    meta_doc = {
        "keywords": keywords,
        "keyword_match_mode": match_mode,
        "keyword_match_mode_note": (
            "phrase=整句连续子串(最严); "
            "all_words=短语内每个词都出现(默认); "
            "any_word=短语内任一词出现"
        ),
        "language": language,
        "total_papers_before_filter": len(items),
        "all_papers_inventory": inventory,
        "categories": meta_categories,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_doc, f, ensure_ascii=False, indent=2)

    with open(path, "w", encoding="utf-8") as f:
        for item in out_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote filtered JSONL ({len(out_lines)} lines): {path}", file=sys.stderr)
    print(f"Wrote category meta: {meta_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
