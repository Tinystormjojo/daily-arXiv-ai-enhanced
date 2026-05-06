"""
Filter AI-enhanced JSONL to at most 5 most keyword-relevant papers per primary category,
and write per-category aggregate summaries for all keyword-matched papers in that category.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List

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


def summarize_category(
    llm: ChatOpenAI,
    category: str,
    papers: List[Dict[str, Any]],
    keywords: List[str],
    language: str,
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
    human = f"""Repository keywords (comma-separated): {kw_desc}
arXiv primary category: {category}
Target language for your answer: {language}

Below are ALL papers in this category that match the keyword filter (or all papers if no keywords were set).
Write a single cohesive summary (2–4 short paragraphs) that synthesizes themes, methods, and trends across these papers.
If the list is empty, reply with one sentence saying there are no papers.

Papers:
{body}
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

    for cat in sorted(by_cat.keys()):
        papers = by_cat[cat]
        matched = [p for p in papers if matches_keywords(p, keywords, match_mode)]

        if keywords and not matched:
            meta_categories[cat] = {
                "summary": "当日该分区无与关键词匹配的论文。",
                "matched_count": 0,
                "titles_in_category": [p.get("title", "") for p in papers],
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
            }
            continue

        if len(to_summarize) > MAX_PAPERS_PER_CATEGORY_SUMMARY:
            to_summarize = to_summarize[:MAX_PAPERS_PER_CATEGORY_SUMMARY]

        if llm is None:
            llm = ChatOpenAI(model=model_name)

        try:
            summary_text = summarize_category(llm, cat, to_summarize, keywords, language)
        except Exception as e:
            print(f"Category summary failed for {cat}: {e}", file=sys.stderr)
            summary_text = "（分类汇总生成失败，请稍后重试或检查 API 配置。）"
        meta_categories[cat] = {
            "summary": summary_text,
            "matched_count": len(matched) if keywords else len(papers),
            "titles_in_category": [p.get("title", "") for p in papers],
        }

    stem = os.path.basename(path)
    # e.g. 2025-01-01_AI_enhanced_Chinese.jsonl -> 2025-01-01_category_meta_Chinese.json
    if "_AI_enhanced_" in stem:
        date_part = stem.split("_AI_enhanced_")[0]
        lang_part = stem.split("_AI_enhanced_")[1].replace(".jsonl", "")
        meta_name = f"{date_part}_category_meta_{lang_part}.json"
    else:
        meta_name = stem.replace(".jsonl", "") + "_category_meta.json"

    out_dir = os.path.dirname(path) or "."
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
