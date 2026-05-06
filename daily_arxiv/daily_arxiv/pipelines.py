# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import arxiv
import os
import random
import time


def _is_arxiv_rate_limit(err: BaseException) -> bool:
    """arxiv 在请求过快时返回 HTTP 429。"""
    s = str(err).lower()
    return "429" in s or "rate" in s or "too many" in s


class DailyArxivPipeline:
    def __init__(self):
        self.page_size = 100
        # 两笔 API 查询之间的最小间隔；过小易触发 429（尤其 GitHub Actions 出口 IP 共享）
        delay = float(os.environ.get("ARXIV_API_DELAY_SECONDS", "4.0"))
        self.client = arxiv.Client(
            page_size=self.page_size,
            delay_seconds=delay,
            num_retries=4,
        )
        self._api_max_rounds = int(os.environ.get("ARXIV_API_429_RETRIES", "8"))

    def process_item(self, item: dict, spider):
        item["pdf"] = f"https://arxiv.org/pdf/{item['id']}"
        item["abs"] = f"https://arxiv.org/abs/{item['id']}"
        search = arxiv.Search(
            id_list=[item["id"]],
        )
        paper = None
        last_err: BaseException | None = None
        for attempt in range(self._api_max_rounds):
            try:
                paper = next(self.client.results(search))
                last_err = None
                break
            except BaseException as e:
                last_err = e
                if not _is_arxiv_rate_limit(e):
                    raise
                if attempt >= self._api_max_rounds - 1:
                    spider.logger.error(
                        "arxiv API 429 重试耗尽 id=%s: %s",
                        item.get("id"),
                        e,
                    )
                    raise
                # 指数退避 + 抖动，遵守 https://info.arxiv.org/help/api/tou.html 合理使用
                wait = min(
                    120.0,
                    4.0 * (2**attempt) + random.uniform(0, 2.5),
                )
                spider.logger.warning(
                    "arxiv API 限流 (429)，%.1f 秒后重试 (%d/%d) id=%s",
                    wait,
                    attempt + 1,
                    self._api_max_rounds,
                    item.get("id"),
                )
                time.sleep(wait)

        if paper is None and last_err is not None:
            raise last_err

        item["authors"] = [a.name for a in paper.authors]
        item["title"] = paper.title
        item["categories"] = paper.categories
        item["comment"] = paper.comment
        item["summary"] = paper.summary
        return item