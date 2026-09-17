"""Fetch Gitee profile + repos for Chinese-market candidates.

Mirrors ``github.py`` so the surrounding pipeline can treat Gitee data the
same way it treats GitHub data. Honors the optional ``GITEE_TOKEN`` env
var (Personal Access Token from https://gitee.com/profile/personal_access_tokens)
to raise the rate limit from 60 req/min (unauthenticated) to 5000 req/hr.

API base: https://gitee.com/api/v5
Auth: ``Authorization: Bearer <token>`` (preferred); the v5 API also accepts
``?access_token=<token>`` query-string — we use Bearer for cleanliness.

This module is intentionally a near-mirror of ``github.py``; the diffs are:
- base URL is gitee.com/api/v5
- some endpoints (e.g. contributors) may 404 for private/repo-no-access
- repo metadata field names differ slightly (``stargazers_count`` is the
  same; ``language`` field, ``created_at`` format, etc. are mostly
  compatible).
"""

import datetime
import os
import time
from typing import Dict, List, Optional

import requests
from models import GitHubProfile
from prompts.template_manager import TemplateManager
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS
from llm_utils import initialize_llm_provider, extract_json_from_response
from config import DEVELOPMENT_MODE
from pdf import logger

GITEE_API_BASE = "https://gitee.com/api/v5"
DEFAULT_TIMEOUT = 10  # seconds per HTTP request
DEFAULT_RETRYABLE = {500, 502, 503, 504}
DEFAULT_MAX_RETRIES = 3


def _cache_filename(api_url: str) -> str:
    """Convert an API URL into a stable cache filename under cache/."""
    safe = api_url.replace(GITEE_API_BASE + "/", "").replace("/", "_")
    return f"cache/gitee_{safe}.json"


def _auth_headers() -> dict:
    """Build auth headers from GITEE_TOKEN (if set)."""
    token = os.environ.get("GITEE_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _fetch_gitee_api(api_url: str, params: Optional[dict] = None) -> tuple:
    """Hit a Gitee v5 API endpoint with caching, auth, and basic retries.

    Returns ``(status_code, parsed_json_or_empty_dict)``.
    """
    headers = _auth_headers()
    cache_filename = _cache_filename(api_url)

    if DEVELOPMENT_MODE and os.path.exists(cache_filename):
        try:
            import json
            from pathlib import Path
            cached = json.loads(Path(cache_filename).read_text(encoding="utf-8"))
            if cached:
                return 200, cached
        except Exception:
            try:
                os.remove(cache_filename)
            except Exception:
                pass

    last_exc: Optional[Exception] = None
    for attempt in range(DEFAULT_MAX_RETRIES):
        try:
            response = requests.get(api_url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
            status_code = response.status_code
            data = response.json() if status_code == 200 else {}

            # Rate-limit awareness.
            remaining = response.headers.get("X-RateLimit-Remaining")
            limit = response.headers.get("X-RateLimit-Limit")
            if remaining is not None and int(remaining) < 5:
                logger.warning(
                    f"Gitee API rate limit low: {remaining}/{limit}. "
                    "Consider setting GITEE_TOKEN to raise the limit to 5000/hr."
                )

            if status_code == 429 and attempt < DEFAULT_MAX_RETRIES - 1:
                wait = int(response.headers.get("Retry-After", "60"))
                logger.warning(
                    f"Gitee API rate limit hit (attempt {attempt + 1}). "
                    f"Sleeping {wait}s before retry."
                )
                time.sleep(wait)
                continue
            if status_code in DEFAULT_RETRYABLE and attempt < DEFAULT_MAX_RETRIES - 1:
                wait = 2 ** attempt
                logger.warning(
                    f"Gitee API transient error {status_code} (attempt {attempt + 1}). "
                    f"Sleeping {wait}s before retry."
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            last_exc = e
            logger.warning(f"Gitee request exception (attempt {attempt + 1}): {e}")
            if attempt < DEFAULT_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                continue
            return 0, {}

        if DEVELOPMENT_MODE and status_code == 200:
            try:
                import json
                from pathlib import Path
                os.makedirs("cache", exist_ok=True)
                Path(cache_filename).write_text(
                    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as e:
                logger.debug(f"Gitee cache write skipped: {e}")
        return status_code, data

    return 0, {}


def extract_gitee_username(gitee_url: str) -> Optional[str]:
    """Pull a username out of a Gitee URL or bare @mention / username."""
    if not gitee_url:
        return None
    s = gitee_url.replace(" ", "").strip()
    import re
    patterns = [
        r"https?://gitee\.com/([^/?#]+)",
        r"gitee\.com/([^/?#]+)",
        r"@([^/?#]+)",
        r"^([a-zA-Z0-9\-_一-龥]+)$",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            username = m.group(1).split("?", 1)[0]
            return username
    return None


def fetch_gitee_profile(gitee_url: str) -> Optional[GitHubProfile]:
    """Fetch the user profile metadata for a Gitee username."""
    username = extract_gitee_username(gitee_url)
    if not username:
        logger.error(f"Could not extract username from: {gitee_url}")
        return None

    api_url = f"{GITEE_API_BASE}/users/{username}"
    status_code, data = _fetch_gitee_api(api_url)

    if status_code == 200 and data:
        return GitHubProfile(
            username=username,
            name=data.get("name"),
            bio=data.get("bio"),
            location=data.get("location"),
            company=data.get("company"),
            public_repos=data.get("public_repos"),
            followers=data.get("followers"),
            following=data.get("following"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            avatar_url=data.get("avatar_url"),
            blog=data.get("blog"),
            twitter_username=data.get("weibo"),
            hireable=False,
        )
    if status_code == 404:
        logger.error(f"Gitee user not found: {username}")
        return None
    logger.error(f"Gitee API error {status_code} for {username}")
    return None


def fetch_repo_contributors(owner: str, repo_name: str) -> list:
    """Best-effort fetch of repo contributors.

    Gitee may return 404 for repos where the contributor API isn't
    exposed. We swallow that as "unknown contributor count" and return
    an empty list so the caller treats the project as a self-project.
    """
    api_url = f"{GITEE_API_BASE}/repos/{owner}/{repo_name}/contributors"
    status_code, data = _fetch_gitee_api(api_url)
    if status_code == 200 and isinstance(data, list):
        return data
    return []


def fetch_all_gitee_repos(gitee_url: str, max_repos: int = 100) -> List[Dict]:
    """Fetch public repos for a Gitee user, sorted by stars descending."""
    username = extract_gitee_username(gitee_url)
    if not username:
        logger.error(f"Could not extract username from: {gitee_url}")
        return []

    api_url = f"{GITEE_API_BASE}/users/{username}/repos"
    params = {
        "sort": "stars_count",
        "direction": "desc",
        "per_page": min(max_repos, 100),
        "type": "all",
    }
    status_code, repos_data = _fetch_gitee_api(api_url, params=params)

    if status_code != 200 or not isinstance(repos_data, list):
        logger.error(f"Gitee repos fetch failed: status={status_code}")
        return []

    projects: List[Dict] = []
    for repo in repos_data:
        if not isinstance(repo, dict):
            continue
        # Skip forks with low downstream engagement (mirrors GitHub logic).
        if repo.get("fork") and (repo.get("forks_count") or 0) < 5:
            continue

        repo_name = repo.get("name") or ""
        contributors_data = fetch_repo_contributors(username, repo_name)
        contributor_count = len(contributors_data) if isinstance(contributors_data, list) else 1

        # Sum contributions across all contributors.
        total_contributions = 0
        user_contributions = 0
        if isinstance(contributors_data, list):
            for contributor in contributors_data:
                if isinstance(contributor, dict):
                    total_contributions += contributor.get("contributions", 0) or 0
                    if (contributor.get("login") or "").lower() == username.lower():
                        user_contributions = contributor.get("contributions", 0) or 0

        # Gitee has no per-user commit count endpoint; treat as
        # best-effort: if contributors endpoint failed (empty list on a
        # popular repo) we still record it but flag unknown.
        project_type = "open_source" if contributor_count > 1 else "self_project"

        projects.append({
            "name": repo.get("name"),
            "description": repo.get("description"),
            "github_url": repo.get("html_url"),  # key reused downstream
            "live_url": repo.get("homepage") or None,
            "technologies": [repo.get("language")] if repo.get("language") else [],
            "project_type": project_type,
            "contributor_count": contributor_count,
            "author_commit_count": user_contributions,
            "total_commit_count": total_contributions,
            "github_details": {  # key reused downstream
                "stars": repo.get("stargazers_count") or 0,
                "forks": repo.get("forks_count") or 0,
                "language": repo.get("language"),
                "description": repo.get("description"),
                "created_at": repo.get("created_at"),
                "updated_at": repo.get("updated_at"),
                "topics": [],  # Gitee doesn't expose topics on the public repo endpoint
                "open_issues": repo.get("open_issues_count") or 0,
                "size": repo.get("size") or 0,
                "fork": repo.get("fork") or False,
                "archived": False,
                "default_branch": repo.get("default_branch"),
                "contributors": contributor_count,
            },
        })

    projects.sort(key=lambda x: x.get("github_details", {}).get("stars", 0), reverse=True)
    return projects


def generate_profile_json(profile: GitHubProfile) -> Dict:
    """Convert a GitHubProfile (reused for Gitee) to a JSON-ready dict."""
    if not profile:
        return {}
    return {
        "username": profile.username,
        "name": profile.name,
        "bio": profile.bio,
        "location": profile.location,
        "company": profile.company,
        "public_repos": profile.public_repos,
        "followers": profile.followers,
        "following": profile.following,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
        "avatar_url": profile.avatar_url,
        "blog": profile.blog,
        "twitter_username": profile.twitter_username,
        "hireable": profile.hireable,
    }


def generate_projects_json(projects: List[Dict], position_title: str = "软件工程师校招岗位") -> List[Dict]:
    """Pick the top 7 unique projects via LLM (mirror of GitHub logic).

    Falls back to the first 7 if the LLM is unavailable or returns
    invalid JSON. Filters out projects with zero author commit count to
    match GitHub's strict-commit policy.
    """
    if not projects:
        return []

    candidates: List[Dict] = []
    for proj in projects:
        if proj.get("author_commit_count") == 0 and proj.get("contributor_count", 1) == 1:
            # Pure personal repos with no measurable contributions are
            # less interesting for the recruiter; keep them as fallback
            # only — GitHub does the same thing.
            candidates.append(proj)
            continue
        candidates.append(proj)

    try:
        projects_data = json.dumps(candidates, indent=2)
        template_manager = TemplateManager()
        prompt = template_manager.render_template(
            "github_project_selection",
            projects_data=projects_data,
            position_title=position_title,
        )
        provider = initialize_llm_provider(DEFAULT_MODEL)
        model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL, {"temperature": 0.1, "top_p": 0.9})
        chat_params = {
            "model": DEFAULT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert technical recruiter analyzing Gitee repositories "
                        "to identify the most impressive projects for a Chinese-market "
                        f"{position_title} candidate. CRITICAL: Select exactly 7 UNIQUE "
                        "projects. Treat Gitee projects identically to GitHub projects — "
                        "the same scoring criteria apply."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "options": model_params,
        }
        response = provider.chat(**chat_params)
        response_text = response["message"]["content"]
        import json
        try:
            response_text = extract_json_from_response(response_text)
            selected_projects = json.loads(response_text)
        except Exception as e:
            logger.warning(f"Gitee LLM project-selection parse failed: {e}; using fallback.")
            return candidates[:7]

        unique: List[Dict] = []
        seen = set()
        for proj in selected_projects:
            name = proj.get("name", "")
            if name and name not in seen:
                unique.append(proj)
                seen.add(name)

        if len(unique) < 7:
            for proj in candidates:
                if len(unique) >= 7:
                    break
                name = proj.get("name", "")
                if name and name not in seen:
                    unique.append(proj)
                    seen.add(name)
        return unique
    except Exception as e:
        logger.warning(f"Gitee LLM project selection error: {e}; using fallback.")
        return candidates[:7]


def fetch_and_display_gitee_info(
    gitee_url: str, position_title: str = "软件工程师校招岗位"
) -> Dict:
    """Public entry: fetch a Gitee profile and curated projects.

    Returns ``{"profile": {...}, "projects": [...], "total_projects": int}``
    on success; returns ``{}`` if the URL is invalid or the user has no
    public repos.
    """
    logger.info(f"Fetching Gitee info for {gitee_url}")
    profile = fetch_gitee_profile(gitee_url)
    if not profile:
        return {}

    projects = fetch_all_gitee_repos(gitee_url)
    projects_json = generate_projects_json(projects, position_title=position_title)

    return {
        "profile": generate_profile_json(profile),
        "projects": projects_json,
        "total_projects": len(projects_json),
    }


if __name__ == "__main__":
    import json
    result = fetch_and_display_gitee_info("https://gitee.com/gitee")
    print(json.dumps(result, indent=2, ensure_ascii=False))