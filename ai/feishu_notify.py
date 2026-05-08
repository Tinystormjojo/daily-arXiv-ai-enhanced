"""
在 AI 增强完成后、aggregate_keywords 覆盖 JSONL 之前调用。
根据当日 *_AI_enhanced_*.jsonl 生成飞书机器人消息（可选，需 FEISHU_WEBHOOK_URL）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# 大厂 / 知名工业界主体（摘要或机构字段中可能出现的中英文关键词，仅作辅助强调）
BIG_TECH_PATTERNS = (
    "google",
    "deepmind",
    "meta",
    "facebook",
    "microsoft",
    "amazon",
    "apple",
    "nvidia",
    "openai",
    "anthropic",
    "bytedance",
    "tiktok",
    "字节",
    "alibaba",
    "阿里",
    "tencent",
    "腾讯",
    "baidu",
    "百度",
    "huawei",
    "华为",
    "meituan",
    "美团",
    "jd.com",
    "京东",
    "netflix",
    "spotify",
    "salesforce",
    "adobe",
    "intel",
    "ibm",
    "oracle",
    "uber",
    "airbnb",
    "linkedin",
    "snap",
    "pinterest",
    "twitter",
    "x corp",
)

PRODUCTION_PATTERNS = (
    "生产",
    "线上",
    "部署",
    "deployment",
    "production",
    "a/b",
    "ab test",
    "live system",
    "industrial",
    "工业界",
    "real-world",
    "real world",
    "in the wild",
    "online serving",
    "线上服务",
)

GEN_REC_POSITIVE = (
    "强相关",
    "密切相关",
    "高度相关",
    "substantive",
    "substantially",
    "end-to-end",
    "端到端",
    "yes",
    "是",
    "相关",
)

GEN_REC_NEGATIVE = (
    "不相关",
    "无关",
    "not related",
    "not substantively",
    "unrelated",
    "否",
)


class DigestOut(BaseModel):
    overall_summary: str = Field(
        description="2–4 short paragraphs: themes and trends across ALL papers today."
    )
    per_paper: List[str] = Field(
        description=(
            "Exactly one short paragraph per paper, in the SAME order as given in the user message "
            "(same count as papers listed). Each blurb: 1–3 sentences."
        )
    )


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _affiliation_blob(p: Dict[str, Any]) -> str:
    ai = p.get("AI") or {}
    aff = (ai.get("author_affiliations") or "").strip()
    authors = p.get("authors")
    if isinstance(authors, list):
        auth = ", ".join(str(a) for a in authors)
    else:
        auth = str(authors or "")
    return f"{auth}\n{aff}"


def _production_blob(p: Dict[str, Any]) -> str:
    ai = p.get("AI") or {}
    return (ai.get("production_deployment") or "").strip()


def _genrec_blob(p: Dict[str, Any]) -> str:
    ai = p.get("AI") or {}
    return (ai.get("generative_recommendation") or "").strip()


def _text_has_any(hay: str, needles: Tuple[str, ...]) -> bool:
    h = hay.lower()
    return any(n in h for n in needles)


def _likely_bigtech(aff_blob: str) -> bool:
    h = aff_blob.lower()
    return any(p in h for p in BIG_TECH_PATTERNS)


def _likely_production(prod: str) -> bool:
    if not prod:
        return False
    pl = prod.lower()
    return any(k.lower() in pl for k in PRODUCTION_PATTERNS)


def _likely_strong_genrec(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    neg = _text_has_any(t, GEN_REC_NEGATIVE)
    pos_markers = ("生成", "推荐", "generative", "recommendation", "recsys")
    has_domain = any(m in text.lower() for m in pos_markers)
    pos = _text_has_any(t, GEN_REC_POSITIVE) or (
        "相关" in text and "推荐" in text and "生成" in text
    )
    if neg and not pos:
        return False
    return pos and (has_domain or "end-to-end" in t or "端到端" in text)


def _heuristic_highlights(
    papers: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    gen_rec: List[Dict[str, Any]] = []
    bigtech_prod: List[Dict[str, Any]] = []
    for p in papers:
        g = _genrec_blob(p)
        if _likely_strong_genrec(g):
            gen_rec.append(p)
        aff = _affiliation_blob(p)
        prod = _production_blob(p)
        if _likely_bigtech(aff) and _likely_production(prod):
            bigtech_prod.append(p)
    return gen_rec, bigtech_prod


def _build_user_prompt(papers: List[Dict[str, Any]], language: str) -> str:
    lines: List[str] = []
    lines.append(
        f"Target language for ALL output fields: {language}.\n"
        "You will produce an overall_summary and per_paper blurbs.\n"
        "Rules:\n"
        "- overall_summary: synthesize themes across every paper below (2–4 short paragraphs).\n"
        "- per_paper: MUST have exactly len(papers) entries, same order as below.\n"
        "  Each entry: 1–3 sentences, standalone, no numbering prefix.\n"
        "- Do not invent affiliations or deployment claims; stay consistent with the given snippets.\n"
    )
    for i, p in enumerate(papers, start=1):
        pid = p.get("id", "")
        title = p.get("title", "")
        ai = p.get("AI") or {}
        tldr = (ai.get("tldr") or "")[:1200]
        aff = (ai.get("author_affiliations") or "")[:800]
        prod = (ai.get("production_deployment") or "")[:800]
        grec = (ai.get("generative_recommendation") or "")[:800]
        lines.append(
            f"--- Paper {i} ---\n"
            f"id: {pid}\n"
            f"title: {title}\n"
            f"tldr: {tldr}\n"
            f"author_affiliations (from abstract): {aff}\n"
            f"production_deployment (from abstract): {prod}\n"
            f"generative_recommendation (from abstract): {grec}\n"
        )
    return "\n".join(lines)


def _call_llm_digest(
    model_name: str, language: str, papers: List[Dict[str, Any]]
) -> DigestOut:
    llm = ChatOpenAI(model=model_name).with_structured_output(DigestOut, method="function_calling")
    sys = SystemMessage(
        content=(
            "You write accurate, concise academic digests. "
            "Output must match the schema; no extra commentary."
        )
    )
    human = HumanMessage(content=_build_user_prompt(papers, language))
    out: DigestOut = llm.invoke([sys, human])
    if len(out.per_paper) != len(papers):
        raise ValueError(
            f"per_paper length {len(out.per_paper)} != papers {len(papers)}"
        )
    return out


def _fallback_digest(papers: List[Dict[str, Any]]) -> DigestOut:
    blurbs: List[str] = []
    for p in papers:
        ai = p.get("AI") or {}
        t = (ai.get("tldr") or p.get("summary") or "").strip()
        if len(t) > 400:
            t = t[:397] + "..."
        blurbs.append(t or "（无摘要）")
    overall = f"当日共 {len(papers)} 篇论文；以下为各篇 TL;DR 摘录。"
    return DigestOut(overall_summary=overall, per_paper=blurbs)


def _format_feishu_text(
    date: str,
    site_url: str,
    overall: str,
    per_lines: List[str],
    papers: List[Dict[str, Any]],
    gen_rec: List[Dict[str, Any]],
    bigtech_prod: List[Dict[str, Any]],
) -> str:
    blocks: List[str] = []
    blocks.append(f"【arXiv 日报】{date}")
    blocks.append(f"页面：{site_url.strip()}")
    blocks.append("")
    blocks.append("【整体总结】")
    blocks.append(overall.strip())
    blocks.append("")
    blocks.append("【重点 · 端到端生成式推荐】")
    if gen_rec:
        blocks.append("以下论文与端到端生成式推荐关联较强（结合摘要与结构化字段，请点开页面核对原文）：")
        for p in gen_rec:
            blocks.append(f"  · {p.get('id','')} — {p.get('title','')}")
    else:
        blocks.append("未发现摘要/字段中明确强相关的端到端生成式推荐工作（或当日批次无此类信号）。")
    blocks.append("")
    blocks.append("【重点 · 大厂 + 生产环境】")
    if bigtech_prod:
        blocks.append("以下论文在机构与生产部署描述上较突出（请点开页面核对原文）：")
        for p in bigtech_prod:
            blocks.append(f"  · {p.get('id','')} — {p.get('title','')}")
    else:
        blocks.append("未发现同时满足「大厂/知名工业界主体」且「生产/线上部署」表述的论文（或摘要未写明）。")
    blocks.append("")
    blocks.append("【各篇概要】")
    for i, (p, blurb) in enumerate(zip(papers, per_lines), start=1):
        pid = p.get("id", "")
        title = p.get("title", "")
        blocks.append(f"{i}. [{pid}] {title}")
        blocks.append(blurb.strip())
        blocks.append("")
    return "\n".join(blocks).strip()


def _post_feishu_text(webhook: str, text: str) -> None:
    # 飞书自定义机器人 text 消息；过长时截断并提示
    max_len = 12000
    if len(text) > max_len:
        text = text[: max_len - 80] + "\n\n…（内容过长已截断，请打开页面查看全文）"

    body = {"msg_type": "text", "content": {"text": text}}
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        webhook,
        data=raw,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        print(f"Feishu webhook HTTP {e.code}: {e.read().decode()}", file=sys.stderr)
        raise
    try:
        j = json.loads(payload)
    except json.JSONDecodeError:
        print(f"Feishu webhook non-JSON response: {payload[:500]}", file=sys.stderr)
        return
    code_ok = j.get("code") == 0
    status_ok = j.get("StatusCode") == 0
    if not code_ok and not status_ok:
        print(f"Feishu webhook error: {j}", file=sys.stderr)
        raise RuntimeError(f"feishu error: {j}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to *_AI_enhanced_*.jsonl (full day, before aggregate)")
    parser.add_argument("--date", required=True, help="Dataset date YYYY-MM-DD (for title line)")
    parser.add_argument("--site-url", required=True, help="GitHub Pages base URL")
    args = parser.parse_args()

    webhook = (os.environ.get("FEISHU_WEBHOOK_URL") or "").strip()
    if not webhook:
        print("FEISHU_WEBHOOK_URL not set, skip Feishu notify", file=sys.stderr)
        return

    if not os.path.isfile(args.data):
        print(f"Data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    papers = _load_jsonl(args.data)
    site_url = args.site_url.strip()
    if not site_url.endswith("/"):
        site_url += "/"

    if not papers:
        _post_feishu_text(
            webhook,
            f"【arXiv 日报】{args.date}\n页面：{site_url}\n\n当日批次无论文记录。",
        )
        print("Feishu: sent empty-day notice", file=sys.stderr)
        return

    language = (os.environ.get("LANGUAGE") or "Chinese").strip()
    model_name = os.environ.get("MODEL_NAME", "deepseek-chat")

    gen_rec_h, bigtech_h = _heuristic_highlights(papers)

    try:
        digest = _call_llm_digest(model_name, language, papers)
    except Exception as e:
        print(f"LLM digest failed, fallback to TL;DR: {e}", file=sys.stderr)
        digest = _fallback_digest(papers)

    text = _format_feishu_text(
        args.date,
        site_url,
        digest.overall_summary,
        digest.per_paper,
        papers,
        gen_rec_h,
        bigtech_h,
    )
    _post_feishu_text(webhook, text)
    print("Feishu: digest sent", file=sys.stderr)


if __name__ == "__main__":
    main()
