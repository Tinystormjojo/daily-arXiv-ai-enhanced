# daily-arXiv-ai-enhanced

一个基于 GitHub Actions + GitHub Pages 的 arXiv 每日追踪工具。  
它会按你配置的分类与关键词自动抓取论文，生成 AI 摘要与前端展示数据，并发布为可直接访问的网站。

---

## 当前版本特性

- 自动化流水线：定时运行，无需自建服务器
- 支持 `CATEGORIES`、`KEYWORDS` 精准筛选
- 关键词模式可选：`phrase` / `all_words` / `any_word`
- 按发布日聚合（UTC 日期），支持历史数据持续累积
- 前端支持分类浏览、日期筛选、统计页、设置页
- 数据与页面解耦：通过 `data` 分支供站点读取
- 飞书日报（可选）：配置 `FEISHU_WEBHOOK_URL` 后自动推送当日论文摘要与重点标注

---

## 工作流程概览

1. GitHub Actions 按计划触发（或手动触发）
2. 爬虫抓取并过滤符合条件的 arXiv 论文
3. 调用大模型生成摘要、关键词等结构化内容
4. 聚合生成前端所需 JSONL/索引文件
5. 页面从 `data` 分支读取数据并渲染展示

---

## 快速开始

1. Fork 本仓库到你自己的账号
2. 在仓库 `Settings -> Secrets and variables -> Actions` 配置参数
3. 启用 GitHub Pages（`main` 分支根目录）
4. 手动运行一次 workflow 验证产出
5. 后续由定时任务自动更新

---

## 必要配置

### Secrets

- `OPENAI_API_KEY`：模型服务密钥
- `OPENAI_BASE_URL`：模型服务地址
- `ACCESS_PASSWORD`（可选）：站点访问密码
- `FEISHU_WEBHOOK_URL`（可选）：飞书自定义机器人 Webhook 完整地址；配置后，每次成功产出当日增强数据会推送日报（整体总结、各篇概要、生成式推荐/大厂生产重点、站点链接）。未配置则跳过

### Variables

- `CATEGORIES`：arXiv 分类，逗号分隔（如 `cs.CL,cs.CV`）
- `KEYWORDS`：关键词短语，逗号分隔
- `KEYWORD_MATCH_MODE`（可选）：`phrase` / `all_words` / `any_word`
- `LANGUAGE`：摘要语言（如 `Chinese` / `English`）
- `MODEL_NAME`：模型名（如 `deepseek-chat`）
- `EMAIL`：用于自动提交的邮箱
- `NAME`：用于自动提交的用户名
- `SITE_URL`（可选）：飞书消息中的展示页链接；不填则默认为 `https://<仓库 owner>.github.io/<仓库名>/`（自定义域名请在此填写完整 URL，建议带末尾 `/`）

---

## KEYWORDS 行为说明

- 当 `KEYWORDS` 非空时，流程会走 arXiv API 检索（`cat + 关键词`）
- 日期过滤以工作流传入的 `ARXIV_CRAWL_DATE` 为准，并按条目的 `<published>` UTC 日期匹配
- 当 `KEYWORDS` 为空时，保留按分类抓取当日新论文的逻辑

---

## 部署说明

- Pages 建议配置：`Source = Deploy from a branch`，`Branch = main / (root)`
- 数据通常由 workflow 写入 `data` 分支
- 首次部署后等待几分钟，再访问：
  - `https://<your-username>.github.io/daily-arXiv-ai-enhanced/`

---

## 常见自定义

- 改抓取范围：调整 `CATEGORIES`、`KEYWORDS`
- 改摘要语言/模型：调整 `LANGUAGE`、`MODEL_NAME`
- 改执行时间：修改 `.github/workflows/run.yml` 的 `schedule`
- 改站点仓库指向：确认 `js/data-config.js` 的 `repoOwner` / `repoName`

---

## 注意事项

- 本项目生成内容包含 AI 输出，请自行甄别
- 不同模型与调用量会产生 API 费用
- Fork 后请替换个人信息与赞助信息（如 `buy-me-a-coffee` 目录）

---

## 赞助

如果这个项目对你有帮助，欢迎通过仓库中的打赏说明支持维护。  
详情见：`buy-me-a-coffee/README.md`
