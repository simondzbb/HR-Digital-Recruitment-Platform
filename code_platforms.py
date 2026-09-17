"""Orchestrate GitHub + Gitee enrichment as a single ``fetch_code_platform_data``
call.

The downstream pipeline (``score.py``, ``transform.convert_github_data_to_text``)
expects a dict shaped like GitHub's output: ``{"profile": {...}, "projects": [...],
"total_projects": int}``. This module merges per-platform fetches into that
shape, with a ``per_platform`` sub-dict preserving each platform's full
result.

The output is also compatible with the existing ``convert_github_data_to_text``
formatter: it reads ``profile`` and ``projects`` as top-level keys and the
merger produces them.
"""

from typing import Dict, Optional

from github import fetch_and_display_github_info
from gitee import fetch_and_display_gitee_info


def _find_profile_url(resume_basics, host: str) -> Optional[str]:
    """Return the first profile URL whose domain contains ``host``.

    Handles Gitee URLs (``gitee.com``), GitHub URLs (``github.com``),
    and slightly-mangled strings (with or without scheme, with or
    without ``www.``).
    """
    if not resume_basics or not getattr(resume_basics, "profiles", None):
        return None
    for p in resume_basics.profiles:
        if not isinstance(p, dict):
            continue
        url = (p.get("url") or "").lower()
        if host in url:
            return p.get("url")
    return None


def _dedup_key(project: Dict) -> str:
    """Stable key for de-duplicating projects across platforms.

    Prefer the repo URL (highest signal); fall back to the project
    name. We accept that two different repos could share the same
    name on different platforms - the URL is the safer primary key
    whenever present.
    """
    return (
        (project.get("github_url") or project.get("html_url") or "").rstrip("/").lower()
        or (project.get("name") or "").strip().lower()
    )


def fetch_code_platform_data(resume_basics, position_title: str = "software engineering position") -> Dict:
    """Fetch GitHub + Gitee data for a candidate and merge the results.

    Returns a dict with top-level keys: profile, projects, total_projects,
    per_platform. If no code-platform profiles are present in the resume,
    returns an empty dict.

    If only one platform is present, that platform's data flows through
    unchanged (with per_platform containing a single key).

    Projects are deduped by repo URL. Ties are broken by name.

    The primary profile is GitHub if available, otherwise Gitee. This
    keeps the existing single-platform behavior identical when only GitHub
    is present.
    """
    github_url = _find_profile_url(resume_basics, "github.com")
    gitee_url = _find_profile_url(resume_basics, "gitee.com")

    per_platform: Dict[str, Dict] = {}
    if github_url:
        per_platform["github"] = fetch_and_display_github_info(
            github_url, position_title=position_title
        )
    if gitee_url:
        per_platform["gitee"] = fetch_and_display_gitee_info(
            gitee_url, position_title=position_title
        )

    if not per_platform:
        return {}

    # Pick primary profile (GitHub preferred).
    primary = per_platform.get("github") or next(iter(per_platform.values()))
    primary_profile = primary.get("profile") or {}

    # Merge projects across platforms.
    seen = set()
    merged_projects = []
    for platform_data in per_platform.values():
        for proj in platform_data.get("projects", []) or []:
            key = _dedup_key(proj)
            if not key or key in seen:
                continue
            seen.add(key)
            merged_projects.append(proj)

    return {
        "profile": primary_profile,
        "projects": merged_projects,
        "total_projects": len(merged_projects),
        "per_platform": per_platform,
    }