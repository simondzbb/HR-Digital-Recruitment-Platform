# Hiring Agent

<p align="center"><strong>Resume-to-Score pipeline</strong> that extracts structured data from resumes (PDF, DOCX, DOC, and images), enriches with GitHub signals, and outputs a fair, explainable evaluation.</p>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3110/">
    <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
  </a>
  <a href="https://github.com/interviewstreet/hiring-agent/blob/master/LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-yellow.svg">
  </a>
  <a href="https://github.com/psf/black">
    <img alt="Code style: Black" src="https://img.shields.io/badge/code%20style-Black-000000.svg">
  </a>
</p>

---

## Contents

- [Context and intent](#context-and-intent)
- [Coverage](#coverage)
- [Overview](#overview)
- [Architecture](#architecture)
- [Installation and Setup](#installation-and-setup)
  - [Prerequisites](#prerequisites)
  - [Quick setup with pip](#quick-setup-with-pip)
  - [Ollama models](#ollama-models)
- [Configuration](#configuration)
- [How it works](#how-it-works)
- [CLI usage](#cli-usage)
- [Directory layout](#directory-layout)
- [Provider details](#provider-details)
- [Contributing](#contributing)
- [License](#license)

---

## Context and intent

This project got a lot of attention recently, and some of the discussion surfaced misconceptions worth addressing directly.

**What this is not:**
- Not an ATS (Applicant Tracking System)
- Not used to screen HackerRank's open roles
- Not a product available to HackerRank customers

**What it actually is:**

Every year HackerRank receives 50,000–60,000 intern applications. No human can read that many resumes well. This tool was built to *rank* them — helping decide which resumes to read first. Resumes scoring below the cutoff are filtered out, but the cutoff is intentionally set very low so only candidates at the very bottom of the distribution are removed. The vast majority pass through to human review, where the real decisions are made.

Since this was built, HackerRank has also shipped [AI Interviewer (Chakra)](https://www.hackerrank.com/products/ai-interviewer/) to automate the first round of interviews — so candidates are no longer assessed on their resume alone.

**On the default model:**

The repo ships with `gemma4:latest` as the default because it runs locally on most laptops without any cloud API key. Actual intern resumes at HackerRank are evaluated using a top-tier Gemini model. The repo ships with a demo config, not the production one.

---

## Coverage

Articles and discussions that have shaped how we think about improving this project:

| Article | Key takeaway |
|---|---|
| [HackerRank open sourced its ATS. My resume scored 90/100. Oh wait 74/100. No — 88/100. Actually 83/100.](https://danunparsed.com/p/hackerrank-open-source-ats) — *Dan Kinsky* | Deep statistical analysis of score variance across 100 runs of the same resume. Isolates which categories are stable (technical skills) vs. noisy (project quality judgments). Points to LLM non-determinism as the root cause. |
| [The Score Depends on the Roll of the Dice](https://pinggy.io/blog/hackerrank_open_source_ats_inconsistent_scoring/) — *Pinggy Blog* | Reproduces the variance findings and surfaces a security issue: invisible text embedded in PDFs can inflate scores significantly. |
| [The Hiring Rubric Inside](https://byteiota.com/hackerrank-ats-open-source-the-hiring-rubric-inside/) — *ByteIota* | Breaks down the scoring weights and argues that a GitHub-centric rubric disadvantages engineers whose work is in private enterprise repos. Also notes the signal degradation risk as candidates optimize for the now-public rubric. |
| [Analyzing resume scoring consistency](https://dev.to/mgobea/hackerrank-open-sourced-its-ats-analyzing-resume-scoring-consistency-1j5d) — *Mariano Gobea Alcoba, DEV Community* | Proposes concrete fixes: standardized data formats, versioned evaluation models, ensemble scoring, and explainability layers to reduce variance and make the system more robust. |
| [AI-Powered Pipeline for Explainable Resume Scoring](https://aitoolly.com/ai-news/article/2026-06-26-interviewstreet-unveils-hiring-agent-an-ai-powered-pipeline-for-explainable-resume-scoring-and-githu) — *AIToolly* | Covers the launch and highlights the transparency argument — making scoring logic public allows scrutiny that proprietary ATS systems never face. |
| [Hacker News discussion](https://news.ycombinator.com/item?id=48713832) | 200+ comment thread covering LLM determinism, GDPR Article 22 implications, and the broader ethics of automated resume filtering. |

**Video coverage**

- [HackerRank Open-Sourced Their ATS?](https://www.youtube.com/shorts/0OP2bhYZQfc) — YouTube Short
- [HackerRank Open-Sourced ATS Tool for selecting Resume](https://www.youtube.com/shorts/UnHGC1Ywhys) — YouTube Short
- [HackerRank Custom ATS Released! Get Your Resume Score & Beat ATS Filters](https://www.youtube.com/watch?v=tQSve-xx4_8) — full walkthrough video

**Community tools built on this repo**

- [Resume Reality Check](https://resume-reality-check-seven.vercel.app/) — hosted tool that lets candidates score their own resume against the same rubric

---

## Overview

Hiring Agent parses a resume file (PDF, DOCX, DOC, or image) to Markdown, extracts sectioned JSON using a local or hosted LLM, augments the data with GitHub profile and repository signals, then produces an objective evaluation with category scores, evidence, bonus points, and deductions. You can run fully local with Ollama or use Google Gemini.

---

## Architecture

<table>
<tr>
<td>

**Flow**

1. `pdf_dispatcher.py` picks the right extractor by file extension (PDF → PyMuPDF, DOCX → python-docx, DOC → LibreOffice→PyMuPDF, image → Tesseract) and produces Markdown-like text.
2. `pdf.py` calls the LLM per section using Jinja templates under `prompts/templates`.
3. `github.py` fetches profile and repos, classifies projects, and asks the LLM to select the top 7.
4. `evaluator.py` runs a strict-scored evaluation with fairness constraints.
5. `score.py` orchestrates everything end to end and writes CSV when development mode is on.

</td>
<td>

**Key modules**

- `models.py`
  Pydantic schemas and LLM provider interfaces.

- `llm_utils.py`
  Provider initialization and response cleanup.

- `transform.py`
  Normalization from loose LLM JSON to JSON Resume style.

- `prompts/`
  All Jinja templates for extraction and scoring.

</td>
</tr>
</table>

---

## Installation and Setup

### Prerequisites

- **Python 3.11+**

  The repository pins `.python-version` to 3.11.13.

- **One LLM backend** (either of them)

  - **Ollama** for local models
    Install from the [official site](https://ollama.com/), then run `ollama serve`.
  - **Google Gemini** if you have an API key, get it from [here](https://aistudio.google.com/api-keys).

- **Optional system dependencies** for non-PDF resume formats (see the table below).
  PDF, DOCX, and Python-only pip packages are enough if you only need PDFs.

### Supported resume formats

| Format | Backend | Extra system dependency |
|---|---|---|
| `.pdf` | PyMuPDF | none (wheel ships native lib) |
| `.docx` | python-docx | none |
| `.doc` (legacy Word) | LibreOffice headless → PyMuPDF | LibreOffice 7+ |
| `.png`, `.jpg`, `.jpeg` | Tesseract OCR (pytesseract + Pillow) | Tesseract 5+ with `eng` language pack; for **Chinese** resumes, also install `chi_sim` (Simplified Chinese). |
| `.tif`, `.tiff` | Tesseract OCR (first frame only) | same as above |

Install the extras you need:

**LibreOffice** (only required for `.doc`)
- Debian/Ubuntu: `sudo apt install libreoffice`
- macOS: `brew install --cask libreoffice`
- Windows: download from <https://www.libreoffice.org/download/> and ensure `soffice.exe` is on `PATH`

**Tesseract** (only required for image OCR)
- Debian/Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim` (the `chi-sim` pack enables Simplified Chinese OCR; without it, image resumes containing Chinese characters will not OCR correctly)
- macOS: `brew install tesseract tesseract-lang` (the `tesseract-lang` formula already includes `chi_sim`)
- Windows: download the UB Mannheim build from <https://github.com/UB-Mannheim/tesseract/wiki>, during installation select **both** `chi_sim` (Simplified Chinese) and `eng` language data, and add `tesseract.exe` to `PATH`

If a required system dependency is missing, the CLI prints a clear error such as:

```
LibreOffice (soffice) not found on PATH. Install LibreOffice to enable .doc support.
Tesseract binary not found on PATH. Install Tesseract OCR ...
```

### Quick setup with pip

```bash
$ git clone https://github.com/interviewstreet/hiring-agent
$ cd hiring-agent

$ python -m venv .venv
# Linux or macOS
$ source .venv/bin/activate
# Windows
# .venv\Scripts\activate

$ pip install -r requirements.txt
```

### Ollama Models

Pull the model you want to use. For example:

```bash
$ ollama pull gemma4:latest
```

If you want different results, you can pull other models such as:

```bash
# For higher system configuration
$ ollama pull gemma3:12b

# For lower system configuration
$ ollama pull gemma3:1b
```

---

## Configuration

Copy the template and set your environment variables.

```bash
$ cp .env.example .env
```

**Environment variables**

| Variable         | Values                                      | Description                                                            |
| ---------------- | ------------------------------------------- | ---------------------------------------------------------------------- |
| `DEFAULT_MODEL`  | for example `gemma4:latest` or `gemini-2.5-pro` | Model to use; must exist in `providers.json` — the provider is inferred from which provider lists it. Defaults to `default_model` in `providers.json`. |
| `GEMINI_API_KEY` | string                                      | Required when using a Gemini model.                                   |
| `GITHUB_TOKEN`   | optional                                    | Inherits from your shell environment, improves GitHub API rate limits (60/hr → 5000/hr). |
| `GITEE_TOKEN`    | optional                                    | Personal Access Token from <https://gitee.com/profile/personal_access_tokens>. Raises Gitee rate limit to 5000/hr. Get one if you evaluate Chinese candidates' Gitee profiles frequently. |

Provider mapping lives in `providers.json` — each provider declares its `base_url`, an optional API-key env var, and per-model parameters; `config.py` loads it and resolves the provider for a model. `config.py` also has a flag:

```python
# config.py
DEVELOPMENT_MODE = True  # enables caching and CSV export
```

You can leave it on during iteration. See the next section for details.

---

## How it works

<details>
<summary><b>1) Resume text extraction (format-aware)</b></summary>

- The format-appropriate extractor reads the file and produces Markdown-like text:
  - **PDF** — `pdf.py` + `pymupdf_rag.py` use PyMuPDF and the `to_markdown` routine (handles headings, links, tables, and basic formatting).
  - **DOCX** — `docx.py` uses `python-docx` to walk paragraphs and tables.
  - **DOC** (legacy Word) — `doc.py` invokes headless LibreOffice (`soffice`) to convert to PDF, then reuses the PDF pipeline.
  - **PNG / JPEG / TIFF** — `image.py` uses Tesseract OCR via `pytesseract` + Pillow.
- `pdf_dispatcher.py` selects the extractor based on the file extension.

</details>

<details>
<summary><b>2) Section parsing with templates</b></summary>

- `prompts/templates/*.jinja` define strict instructions for each section
  Basics, Work, Education, Skills, Projects, Awards.
- `pdf.PDFHandler` calls the LLM per section and assembles a `JSONResume` object (see `models.py`).

</details>

<details>
<summary><b>3) GitHub enrichment</b></summary>

- `github.py` extracts a username from the resume profiles, fetches profile and repos, and classifies each project.
- It asks the LLM to select exactly 7 unique projects with a minimum author commit threshold, favoring meaningful contributions.

</details>

<details>
<summary><b>4) Evaluation</b></summary>

- `evaluator.py` scores the resume against the **role** selected on the command line.
- Each role lives in `roles/<role_name>/` and defines its own scoring categories and weights in `role.json`, plus its own `criteria.jinja` and `system_message.jinja` prompts (encoding fairness and scoring rules).
- The shipped `software_engineering_intern` role scores `open_source`, `self_projects`, `production`, and `technical_skills`, plus bonus and deductions, with evidence for each. Other roles can define entirely different categories.

</details>

<details>
<summary><b>5) Output and CSV export</b></summary>

- `score.py` prints a readable summary to stdout.
- When `DEVELOPMENT_MODE=True` it creates or appends a per-role `resume_evaluations_<role>.csv` with key fields (columns follow the role's categories), and caches intermediate JSON under `cache/`.

</details>

---

## Chinese Market Usage / 中文市场使用

The shipped `software_engineering_intern_zh` role scores Chinese-market resumes end to end:

```bash
$ python score.py ./resume/张三简历.pdf --role software_engineering_intern_zh
```

What the Chinese role does differently from the English one:

- **Date parsing** — accepts `2020.01 - 至今`, `2020年1月 - 2021年6月`, `2020/09 - 2024/06`, etc.; output is normalized to `YYYY-MM` in both `cache/` JSON and CSV.
- **Education parsing** — recognizes `本科 / 硕士 / 博士 / 大专 / MBA`; pulls `均分 / 成绩 / GPA / 排名` into the `score` field.
- **Project tech-list parsing** — splits on Chinese enumeration comma `、` and other Chinese separators, in addition to `|` / `,`.
- **Profile platforms** — recognizes `gitee.com`, `gitcode.com`, `zhihu.com`, `juejin.cn`, `csdn.net`, `bilibili.com`, `nowcoder.cn`, `oschina.net`, `cnblogs.com`, `segmentfault.com` in addition to the international ones.
- **Resume field fallbacks** — if the LLM emits Chinese-keyed JSON (`工作经历`, `教育经历`, `荣誉奖项`, etc.), the transform layer still recovers the data.
- **OCR** — `image.py` uses `lang="chi_sim+eng"` so image resumes containing Chinese characters OCR correctly (requires the `chi_sim` Tesseract language pack).
- **Scoring rubric** — Chinese equivalents are weighted: `开源之夏 / Google编程之夏 (GSoC)`, `openEuler 暑期项目`, `中国开源年会`, Chinese-ecosystem OSS projects (Apache SkyWalking, Nacos, Seata, Ant Design, Taro, uni-app, HarmonyOS, TiDB, TDengine, openEuler, etc.). The bonus for having a portfolio/LinkedIn is replaced with a presence of any Chinese tech community account (Gitee / CSDN / 掘金 / 知乎 / Bilibili / GitCode).
- **Output report** — `print_evaluation_results` prints bilingual headers (English label / 中文标签), and category labels come from `role.json` (e.g. `🌐 开源贡献 / Open Source`).

The English role (`software_engineering_intern`) is unchanged and works exactly as before.

**Required setup for image OCR of Chinese resumes:**

Tesseract must have the `chi_sim` language pack installed:

- Debian/Ubuntu: `sudo apt install tesseract-ocr-chi-sim`
- macOS: `brew install tesseract-lang` (already includes `chi_sim`)
- Windows: UB Mannheim build, tick `chi_sim` during install

Without `chi_sim`, image resumes containing Chinese characters will OCR as empty/garbled. The CLI prints a clear error if Tesseract is missing entirely.

---

## Code-platform enrichment (GitHub + Gitee) / 代码平台增强

When a candidate's resume contains GitHub or Gitee profile URLs in `basics.profiles`, the pipeline automatically fetches each platform's public profile, repos, and contributors, then merges the data into a unified `code_platforms` cache file (`cache/code_platforms_cache_<basename>.json`). The LLM evaluator uses this data when scoring `open_source` and `competitions` (especially the GitHub-only contribution analysis that affects `open_source` weighting).

| Platform | Auth | Rate limit (unauth) | Rate limit (auth) | Token env var |
|---|---|---|---|---|
| GitHub | Bearer token | 60 req/hr | 5000 req/hr | `GITHUB_TOKEN` |
| Gitee  | Bearer token | 60 req/min | 5000 req/hr | `GITEE_TOKEN` |

**Setup for Gitee** (optional, only if you evaluate Chinese candidates' Gitee profiles):

1. Go to <https://gitee.com/profile/personal_access_tokens> and generate a Personal Access Token.
2. Set `GITEE_TOKEN=<your-token>` in your environment (or in `.env`).
3. Re-run `score.py`. The CLI prints a one-time tip if the token is missing.

**Behavior notes:**
- The orchestrator (`code_platforms.py`) is opt-in by presence: a candidate without a Gitee URL produces identical output to before this change.
- Projects from both platforms are deduplicated by repo URL (GitHub preferred as the primary profile when both are present).
- The cache filename changed from `cache/githubcache_<basename>.json` to `cache/code_platforms_cache_<basename>.json`. Old caches are not migrated; delete them if you want to force a re-fetch.
- Gitee contributors endpoint may 404 for some repos; in that case the project is conservatively treated as a `self_project`.

---

## CLI usage

### End to end scoring

Provide a path to a resume file (PDF, DOCX, DOC, or image) and the role to score against. `--role` is the name of a directory under `roles/` and is **required**.

```bash
$ python score.py ./resume/sample.pdf --role software_engineering_intern
$ python score.py ./resume/sample.docx --role software_engineering_intern
$ python score.py ./resume/sample.png --role software_engineering_intern
```

What happens:

1. If development mode is on, the resume extraction result is cached to `cache/resumecache_<basename-without-ext>.json` (the cache key strips any extension, so `sample.pdf`, `sample.docx`, `sample.png` all share `cache/resumecache_sample.json`).
2. If a GitHub profile is found in the resume, repositories are fetched and cached to `cache/githubcache_<basename-without-ext>.json`.
3. The evaluator scores the resume against the selected role, prints a report and, in development mode, appends a CSV row to `resume_evaluations_<role>.csv`.

### Roles

A role bundles its rubric in `roles/<role_name>/`:

```text
roles/software_engineering_intern/
├── role.json           # categories, weights (max), bonus_max, score bounds, position_title
├── criteria.jinja      # evaluation criteria prompt (receives {{ text_content }})
└── system_message.jinja
```

`role.json` drives the scoring schema, the printed report, the CSV columns, and
the score caps — so each role can score against its own categories and weights.

To add a role, scaffold one with basic template files and then edit them:

```bash
$ python score.py --init-role backend_engineer
# edit roles/backend_engineer/{role.json,criteria.jinja,system_message.jinja}
$ python score.py ./resume/sample.pdf --role backend_engineer
```

`--init-role` creates the role directory with placeholder categories and prompts
(it only scaffolds; it does not score a resume). You can also copy an existing
role directory instead.

---

## Directory layout

```text
.
├── .env.example
├── .python-version
├── config.py
├── doc.py          # legacy .doc → LibreOffice → PyMuPDF
├── docx_parser.py  # .docx → python-docx
├── evaluator.py
├── github.py
├── image.py        # PNG/JPEG/TIFF → Tesseract OCR
├── llm_utils.py
├── models.py
├── pdf.py          # PDF → PyMuPDF
├── pdf_dispatcher.py  # format-aware entry point used by score.py
├── prompt.py
├── prompts/
│   ├── template_manager.py
│   └── templates/
│       ├── awards.jinja
│       ├── basics.jinja
│       ├── education.jinja
│       ├── github_project_selection.jinja
│       ├── projects.jinja
│       ├── skills.jinja
│       ├── system_message.jinja
│       └── work.jinja
├── providers.json
├── pymupdf_rag.py
├── requirements.txt
├── roles.py
├── roles/
│   ├── software_engineering_intern/
│   │   ├── role.json
│   │   ├── criteria.jinja
│   │   └── system_message.jinja
│   └── software_engineering_intern_zh/   # Chinese-market variant
│       ├── role.json
│       ├── criteria.jinja
│       └── system_message.jinja
├── score.py
└── transform.py
```

---

## Provider details

### Ollama

- Set `DEFAULT_MODEL` to any pulled model listed in `providers.json`, for example `gemma4:latest`
- Requests go through `models.OpenAICompatibleProvider` against Ollama's OpenAI-compatible endpoint (`http://localhost:11434/v1`)

### Gemini

- Set `DEFAULT_MODEL` to a Gemini model listed in `providers.json`, for example `gemini-2.0-flash`
- Provide `GEMINI_API_KEY`
- The same `models.OpenAICompatibleProvider` wrapper is used, pointed at Gemini's OpenAI-compatible endpoint

---

## Contributing

Please read the [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidelines on filing issues, proposing changes, and submitting pull requests. Key principles include:

- Keep prompts declarative and provider-agnostic.
- Validate changes with a couple of real resumes under different providers.
- Add or adjust unit-free smoke tests that call each stage with minimal inputs.

---


## License

[MIT](https://github.com/interviewstreet/hiring-agent/blob/master/LICENSE) © HackerRank
