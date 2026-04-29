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


def matches_keywords(item: Dict[str, Any], keywords: List[str]) -> bool:
    if not keywords:
        return True
    blob = paper_search_blob(item)
    return any(k.lower() in blob for k in keywords)


def relevance_score(item: Dict[str, Any], keywords: List[str]) -> int:
    if not keywords:
        return 0
    title = (item.get("title") or "").lower()
    summary = (item.get("summary") or "").lower()
    score = 0
    for kw in keywords:
        k = kw.lower()
        score += title.count(k) * 3 + summary.count(k)
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
    language = os.environ.get("LANGUAGE", "Chinese")
    keywords = parse_keywords(os.environ.get("KEYWORDS"))
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
        matched = [p for p in papers if matches_keywords(p, keywords)]

        if keywords and not matched:
            meta_categories[cat] = {
                "summary": "当日该分区无与关键词匹配的论文。",
                "matched_count": 0,
            }
            continue

        pool = matched if keywords else papers
        if keywords:
            ranked = sorted(
                pool,
                key=lambda p: (relevance_score(p, keywords), p.get("id", "")),
                reverse=True,
            )
        else:
            ranked = list(pool)
        top = ranked[:TOP_K]
        out_lines.extend(top)

        to_summarize = matched if keywords else papers
        if not to_summarize:
            meta_categories[cat] = {"summary": "", "matched_count": 0}
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

    meta_doc = {
        "keywords": keywords,
        "language": language,
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
