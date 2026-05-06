import os
import re
from urllib.parse import urlencode

import scrapy

from daily_arxiv.arxiv_search_query import (
    build_search_query_for_category,
    parse_keywords,
)


def _published_utc_yyyy_mm_dd(entry) -> str | None:
    """Atom <published> 取 UTC 日历日 YYYY-MM-DD。"""
    pub = entry.xpath('.//*[local-name()="published"]/text()').get()
    if not pub:
        return None
    pub = pub.strip()
    if len(pub) < 10:
        return None
    return pub[:10]


def _normalize_arxiv_id_from_abs_url(abs_url: str) -> str:
    """http://arxiv.org/abs/2401.12345v2 -> 2401.12345"""
    if not abs_url:
        return ""
    tail = abs_url.strip().split("/abs/")[-1]
    return re.sub(r"v\d+$", "", tail, flags=re.IGNORECASE)


class ArxivSpider(scrapy.Spider):
    name = "arxiv"
    allowed_domains = ["arxiv.org", "export.arxiv.org"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        categories = os.environ.get("CATEGORIES", "cs.CV")
        categories = categories.split(",")
        self.target_categories = [c.strip() for c in categories if c.strip()]
        self.target_categories_set = set(self.target_categories)

        self.keywords = parse_keywords(os.environ.get("KEYWORDS"))
        self.keyword_match_mode = (
            os.environ.get("KEYWORD_MATCH_MODE") or "all_words"
        ).strip().lower()
        if self.keyword_match_mode not in ("phrase", "all_words", "any_word"):
            self.keyword_match_mode = "all_words"

        self.crawl_date = (os.environ.get("ARXIV_CRAWL_DATE") or "").strip()
        self.use_api_keyword_search = len(self.keywords) > 0

        self.seen_ids: set[str] = set()
        self.max_results = int(os.environ.get("ARXIV_API_MAX_RESULTS", "200"))
        self.max_start = int(os.environ.get("ARXIV_API_MAX_START", "2000"))

        if self.use_api_keyword_search:
            self.logger.info(
                "Using export.arxiv.org API search (KEYWORDS set); "
                "keywords=%s mode=%s crawl_date=%s (按 published UTC 过滤)",
                self.keywords,
                self.keyword_match_mode,
                self.crawl_date or "(no day filter)",
            )
            self.start_urls = []
        else:
            self.start_urls = [
                f"https://arxiv.org/list/{cat}/new" for cat in self.target_categories
            ]

    def start_requests(self):
        if self.use_api_keyword_search:
            if not self.crawl_date:
                self.logger.warning(
                    "KEYWORDS set but ARXIV_CRAWL_DATE empty — "
                    "no per-day filter on Atom published date (broader results)."
                )
            # 串行按分区请求 API，避免并发打满 export 导致 500
            if self.target_categories:
                cat = self.target_categories[0]
                sq = build_search_query_for_category(
                    cat,
                    self.keywords,
                    self.keyword_match_mode,
                )
                self.logger.info("API search_query[%s]: %s", cat, sq)
                yield self._api_request(cat, sq, start=0)
            return

        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse_list_html)

    def _next_category_request(self, finished_cat: str):
        try:
            idx = self.target_categories.index(finished_cat)
        except ValueError:
            return None
        if idx + 1 >= len(self.target_categories):
            return None
        next_cat = self.target_categories[idx + 1]
        sq = build_search_query_for_category(
            next_cat,
            self.keywords,
            self.keyword_match_mode,
        )
        self.logger.info("API search_query[%s]: %s", next_cat, sq)
        return self._api_request(next_cat, sq, start=0)

    def _api_request(self, category: str, search_query: str, start: int):
        # search_query 中不使用 submittedDate:（export 常 500）；按日过滤见 parse_api_atom + published
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": str(start),
            "max_results": str(self.max_results),
        }
        url = "https://export.arxiv.org/api/query?" + urlencode(params)
        # export.arxiv.org 的 robots.txt 禁止 /api/，但官方 API 允许程序化访问（需限速，见 pipelines）
        return scrapy.Request(
            url,
            callback=self.parse_api_atom,
            errback=self._api_errback,
            meta={
                "category": category,
                "search_query": search_query,
                "start": start,
                "dont_obey_robotstxt": True,
                # 默认 HttpError 会丢弃 429，导致既不回调也无法链式下一分区
                "handle_httpstatus_list": [429, 500, 502, 503, 504],
            },
            dont_filter=True,
        )

    def _api_errback(self, failure):
        """网络层失败时仍继续下一分区，避免整次爬取卡死在第一个分区。"""
        req = failure.request
        cat = req.meta.get("category", "?")
        start = req.meta.get("start", -1)
        self.logger.error(
            "API 请求失败 category=%s start=%s: %s",
            cat,
            start,
            failure.getErrorMessage(),
        )
        nxt = self._next_category_request(cat)
        if nxt:
            yield nxt

    def parse_api_atom(self, response):
        cat = response.meta["category"]
        sq = response.meta["search_query"]
        start = response.meta["start"]

        if response.status != 200:
            self.logger.warning(
                "API HTTP %s category=%s start=%s，放弃本请求并继续下一分区（若仍有）",
                response.status,
                cat,
                start,
            )
            nxt = self._next_category_request(cat)
            if nxt:
                yield nxt
            return

        total = response.xpath(
            '//*[local-name()="totalResults"]/text()'
        ).get()
        try:
            total_int = int(total.strip()) if total else 0
        except ValueError:
            total_int = 0

        entries = response.xpath('//*[local-name()="entry"]')
        self.logger.info(
            "API atom[%s] start=%s entries=%s totalResults=%s",
            cat,
            start,
            len(entries),
            total_int,
        )

        # HTTP 200 但 body 为 API 错误说明（常见于查询语法或服务端异常）
        for entry in entries:
            eid = (entry.xpath('.//*[local-name()="id"]/text()').get() or "").strip()
            if "api/errors" in eid or "arxiv.org/api/errors" in eid:
                err_txt = "".join(
                    entry.xpath('.//*[local-name()="summary"]//text()').getall()
                ).strip()
                self.logger.error(
                    "arXiv API 返回错误条目 category=%s start=%s: %s",
                    cat,
                    start,
                    err_txt[:500],
                )
                nxt = self._next_category_request(cat)
                if nxt:
                    yield nxt
                return

        for entry in entries:
            abs_url = entry.xpath(
                './/*[local-name()="id"]/text()'
            ).get()
            if not abs_url:
                continue
            aid = _normalize_arxiv_id_from_abs_url(abs_url.strip())
            if not aid or aid in self.seen_ids:
                continue
            if "api/errors" in abs_url:
                continue

            if self.crawl_date:
                pday = _published_utc_yyyy_mm_dd(entry)
                if pday != self.crawl_date:
                    continue

            self.seen_ids.add(aid)

            title = entry.xpath(
                './/*[local-name()="title"]/text()'
            ).get()
            title = title.strip() if title else ""
            summary = "".join(
                entry.xpath('.//*[local-name()="summary"]//text()').getall()
            ).strip()

            authors = entry.xpath(
                './/*[local-name()="author"]//*[local-name()="name"]/text()'
            ).getall()
            authors = [a.strip() for a in authors if a.strip()]

            terms = entry.xpath('.//*[local-name()="category"]/@term').getall()
            primary = entry.xpath(
                './/*[local-name()="primary_category"]/@term'
            ).get()
            if primary:
                cats = [primary] + [t for t in terms if t != primary]
            else:
                cats = terms or []

            comment_texts = entry.xpath(
                './/*[local-name()="comment"]//text()'
            ).getall()
            comment = " ".join(t.strip() for t in comment_texts if t.strip()) or None

            yield {
                "id": aid,
                "categories": cats,
                "title": title,
                "summary": summary,
                "authors": authors,
                "comment": comment,
                "_from_api_atom": True,
            }

        # submittedDate 降序：若整页条目的 published 日均早于 crawl_date，后续页更旧，可结束本分区分页
        if self.crawl_date and entries:
            ok_entries = []
            for entry in entries:
                eid = (entry.xpath('.//*[local-name()="id"]/text()').get() or "").strip()
                if "api/errors" in eid:
                    continue
                ok_entries.append(entry)
            if ok_entries:
                pdays = [_published_utc_yyyy_mm_dd(e) for e in ok_entries]
                if all(
                    p is not None and p < self.crawl_date for p in pdays
                ):
                    nxt = self._next_category_request(cat)
                    if nxt:
                        yield nxt
                    return

        next_start = start + len(entries)
        if (
            len(entries) == self.max_results
            and next_start < self.max_start
            and (total_int == 0 or next_start < total_int)
        ):
            yield self._api_request(cat, sq, next_start)
        else:
            nxt = self._next_category_request(cat)
            if nxt:
                yield nxt

    def parse_list_html(self, response):
        anchors = []
        for li in response.css("div[id=dlpage] ul li"):
            href = li.css("a::attr(href)").get()
            if href and "item" in href:
                anchors.append(int(href.split("item")[-1]))

        for paper in response.css("dl dt"):
            paper_anchor = paper.css("a[name^='item']::attr(name)").get()
            if not paper_anchor:
                continue

            paper_id = int(paper_anchor.split("item")[-1])
            if anchors and paper_id >= anchors[-1]:
                continue

            abstract_link = paper.css("a[title='Abstract']::attr(href)").get()
            if not abstract_link:
                continue

            arxiv_id = abstract_link.split("/")[-1]

            paper_dd = paper.xpath("following-sibling::dd[1]")
            if not paper_dd:
                continue

            subjects_text = paper_dd.css(".list-subjects .primary-subject::text").get()
            if not subjects_text:
                subjects_text = paper_dd.css(".list-subjects::text").get()

            if subjects_text:
                categories_in_paper = re.findall(r"\(([^)]+)\)", subjects_text)
                paper_categories = set(categories_in_paper)
                if paper_categories.intersection(self.target_categories_set):
                    yield {
                        "id": arxiv_id,
                        "categories": list(paper_categories),
                    }
                    self.logger.info(
                        "Found paper %s with categories %s",
                        arxiv_id,
                        paper_categories,
                    )
                else:
                    self.logger.debug(
                        "Skipped paper %s with categories %s",
                        arxiv_id,
                        paper_categories,
                    )
            else:
                self.logger.warning(
                    "Could not extract categories for paper %s, including anyway",
                    arxiv_id,
                )
                yield {
                    "id": arxiv_id,
                    "categories": [],
                }
