from typing import Dict, List, Optional, Tuple
import re
import pdb
from models import JSONResume


# ---------------------------------------------------------------------------
# Chinese-market localization helpers
# ---------------------------------------------------------------------------

# Bilingual tokens that mean "this position is current" — used in date
# parsing so a Chinese resume saying "2020.01 - 至今" produces the same
# "Present" sentinel that an English resume saying "Jan 2020 - Present" does.
CURRENT_POSITION_TOKENS = {
    "present", "current", "now", "ongoing",
    "至今", "目前", "现在", "当下", "在读", "在岗",
}

# Chinese-language educational degree keywords, ordered by descending priority
# (most specific first). Used by _classify_study_type to extract the degree
# type out of a free-form Chinese education string.
_CN_STUDY_TYPE_KEYWORDS = [
    "博士", "硕士", "研究生",
    "本科", "学士",
    "大专", "专科",
    "高中", "MBA",
]

# English-language equivalents (kept so the existing English-only path
# still works).
_EN_STUDY_TYPE_KEYWORDS = [
    "Ph.D", "M.S", "M.A", "M.B.A", "B.S", "B.E",
    "Bachelor", "Master", "Doctor", "Associate", "High School",
]


# Separators accepted by _split_technologies. Order matters: the more
# specific separators (Chinese enumeration comma) come first so that a
# string like "Python、Java" splits as expected rather than failing to
# find ",". The legacy English "," is the last entry so plain-English
# resumes keep working as before.
TECH_SEPARATORS = ["|", "、", "，", ";", "；", "和", "与", "&", "+", "/", ","]


def _get(parsed_data: Dict, *keys: str):
    """Look up the first non-empty value across a list of candidate keys.

    Used to make every section lookup resilient to the LLM emitting a
    Chinese-keyed payload (e.g. ``"工作经历"``) instead of the canonical
    English key (``"work"``). Returns an empty list (or None) if none of
    the keys are present or all values are falsy.
    """
    for k in keys:
        if k in parsed_data:
            v = parsed_data[k]
            if v:
                return v
    return []


def _classify_study_type(text: str) -> Tuple[str, str]:
    """Split a free-form degree string into ``(studyType, area)``.

    Works for both Chinese (``本科 计算机科学`` or
    ``计算机科学（本科）``) and English (``B.S., Computer Science``).
    Returns ``("其他", original_text)`` if nothing matches.
    """
    if not text:
        return "其他", ""

    for kw in _CN_STUDY_TYPE_KEYWORDS:
        if kw in text:
            area = text.replace(kw, "").strip(" ,，()（）:：")
            return kw, area or ""

    for kw in _EN_STUDY_TYPE_KEYWORDS:
        if kw in text:
            area = text.replace(kw, "").strip(" ,().")
            return kw, area or ""

    return "其他", text.strip()


def _split_technologies(s: str) -> List[str]:
    """Split a technology string using any supported separator.

    Tries separators in ``TECH_SEPARATORS`` order — the first one present
    in the string wins. Strips whitespace and discards empty pieces.
    """
    if not s:
        return []
    for sep in TECH_SEPARATORS:
        if sep in s:
            return [t.strip() for t in s.split(sep) if t.strip()]
    return [s.strip()]


# Year-month pair regex. Matches:
#   2020年1月, 2020年12月, 2020.01, 2020.12, 2020/1, 2020-1, 202001, 2020
_YM_PATTERN = re.compile(
    r"(\d{4})"               # year (4 digits)
    r"(?:"
    r"[\.\-/年](\d{1,2})?"   # optional separator + 1-2 digit month
    r"(?:月)?"               # optional Chinese 月
    r")?"
)
# Separator between start and end dates inside a range string.
_RANGE_SEP = re.compile(r"[\s]*[-—–~到至][\s]*")


def _parse_single_date(token: str) -> Optional[str]:
    """Parse a single date token to ``YYYY-MM`` (or ``None``).

    Accepts: ``2020年1月``, ``2020.01``, ``2020/1``, ``2020-12``,
    ``202001``, ``2020``, ``Jan 2020``, ``January 2020``.
    """
    if not token:
        return None
    token = token.strip()

    # English month-name form (e.g. "Jan 2020", "January 2020").
    en_months = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    en_match = re.match(
        r"(?i)\s*([A-Za-z]+)\.?\s+(\d{4})\s*$", token
    )
    if en_match:
        month_name = en_match.group(1).lower()
        if month_name in en_months:
            month = en_months[month_name]
            return f"{en_match.group(2)}-{month:02d}"

    # Numeric form (Chinese / ISO / dot / slash variants).
    ym = _YM_PATTERN.search(token)
    if ym:
        year = ym.group(1)
        month = ym.group(2)
        if month:
            return f"{year}-{int(month):02d}"
        return f"{year}-01"
    return None


def _looks_current(token: str) -> bool:
    """Return True if ``token`` matches any current-position keyword."""
    if not token:
        return False
    t = token.strip().lower()
    return t in CURRENT_POSITION_TOKENS


def transform_parsed_data(parsed_data: Dict) -> Dict:
    try:
        if isinstance(parsed_data, dict):
            if "basics" in parsed_data and len(parsed_data) > 1:
                transformed = {
                    "basics": transform_basics(parsed_data.get("basics", {})),
                    "work": transform_work_experience(
                        _get(
                            parsed_data,
                            "work", "work_experience", "experience",
                            "工作经历", "工作", "实习经历", "项目经历",
                        )
                    ),
                    "volunteer": transform_organizations(
                        _get(parsed_data, "organizations", "志愿经历", "志愿者")
                    ),
                    "education": transform_education(
                        _get(parsed_data, "education", "教育经历", "学历", "教育背景")
                    ),
                    "awards": transform_achievements(
                        _get(
                            parsed_data,
                            "awards", "achievements", "honors_and_awards",
                            "荣誉奖项", "获奖经历", "奖项",
                        )
                    ),
                    "certificates": _get(parsed_data, "certificates", "证书", "资格证书"),
                    "publications": _get(parsed_data, "publications", "出版物", "论文"),
                    "skills": transform_skills_comprehensive(parsed_data),
                    "languages": _get(parsed_data, "languages", "语言能力", "语言"),
                    "interests": _get(parsed_data, "interests", "兴趣爱好", "兴趣"),
                    "references": _get(parsed_data, "references", "推荐人"),
                    "projects": transform_projects_comprehensive(parsed_data),
                    "meta": parsed_data.get("meta", {}),
                }
            else:
                if "basics" in parsed_data:
                    basics_data = parsed_data.get("basics", parsed_data)
                    transformed = {"basics": transform_basics(basics_data)}
                elif any(
                    k in parsed_data
                    for k in (
                        "work", "work_experience", "experience",
                        "工作经历", "工作", "实习经历", "项目经历",
                    )
                ):
                    work_data = _get(
                        parsed_data,
                        "work", "work_experience", "experience",
                        "工作经历", "工作", "实习经历", "项目经历",
                    )
                    transformed = {"work": transform_work_experience(work_data)}
                elif any(
                    k in parsed_data for k in ("education", "教育经历", "学历", "教育背景")
                ):
                    transformed = {"education": transform_education(
                        _get(parsed_data, "education", "教育经历", "学历", "教育背景")
                    )}
                elif any(
                    k in parsed_data
                    for k in (
                        "skills", "librariesFrameworks", "toolsPlatforms", "databases",
                        "技能", "专业技能", "技术栈",
                    )
                ):
                    transformed = {
                        "skills": transform_skills_comprehensive(parsed_data)
                    }
                elif any(
                    k in parsed_data
                    for k in ("projects", "projectsOpenSource", "项目经历", "项目")
                ):
                    transformed = {
                        "projects": transform_projects_comprehensive(parsed_data)
                    }
                elif any(
                    k in parsed_data
                    for k in (
                        "awards", "achievements", "honors_and_awards",
                        "荣誉奖项", "获奖经历", "奖项",
                    )
                ):
                    awards_data = _get(
                        parsed_data,
                        "awards", "achievements", "honors_and_awards",
                        "荣誉奖项", "获奖经历", "奖项",
                    )
                    transformed = {"awards": transform_achievements(awards_data)}
                else:
                    transformed = parsed_data

            return transformed
        else:
            return parsed_data

    except Exception as e:
        print(f"Error transforming parsed data: {e}")
        return parsed_data


def extract_domain_from_url(url: str) -> str:
    try:
        if "://" in url:
            url = url.split("://")[1]
        domain = url.split("/")[0]
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def get_network_name(domain: str) -> str:
    domain_mapping = {
        # international platforms
        "github.com": "GitHub",
        "linkedin.com": "LinkedIn",
        "leetcode.com": "LeetCode",
        "stackoverflow.com": "Stack Overflow",
        "hackerrank.com": "HackerRank",
        "behance.net": "Behance",
        "dev.to": "DEV Community",
        "twitter.com": "X",
        "x.com": "X",
        # Chinese-market platforms
        "gitee.com": "Gitee",
        "gitcode.com": "GitCode",
        "csdn.net": "CSDN",
        "juejin.cn": "Juejin",
        "zhihu.com": "Zhihu",
        "bilibili.com": "Bilibili",
        "nowcoder.cn": "Nowcoder",
        "nowcoder.com": "Nowcoder",
        "oschina.net": "OSChina",
        "cnblogs.com": "Cnblogs",
        "segmentfault.com": "SegmentFault",
    }
    return domain_mapping.get(domain, "")


def transform_basics(basics_data: Dict) -> Dict:
    if not isinstance(basics_data, dict):
        return basics_data

    profiles = basics_data.get("profiles", [])

    transformed_profiles = []
    if isinstance(profiles, list):
        for i, profile in enumerate(profiles):
            if isinstance(profile, dict):
                transformed_profile = profile.copy()
                url = transformed_profile.get("url", "")
                network = transformed_profile.get("network")

                if url and network is None:
                    domain = extract_domain_from_url(url)
                    network_name = get_network_name(domain)

                    if network_name:
                        transformed_profile["network"] = network_name
                        username = extract_username_from_url(url, domain)
                        if username:
                            transformed_profile["username"] = username
                transformed_profiles.append(transformed_profile)

    basics_data["profiles"] = transformed_profiles
    return basics_data


def extract_username_from_url(url: str, domain: str) -> str:
    try:
        path = url.split(domain)[1] if domain in url else ""
        if not path:
            return ""
        path = path.lstrip("/")

        parts = [part for part in path.split("/") if part]

        if parts:
            if domain == "linkedin.com":
                return parts[1]
            elif domain == "stackoverflow.com":
                return parts[2]
            else:
                return parts[0]
        return ""
    except Exception:
        return ""


def transform_work_experience(work_list: List) -> List[Dict]:
    transformed = []
    for item in work_list:
        if isinstance(item, dict):
            description = item.get("description", "") or item.get("工作描述", "")
            if isinstance(description, list):
                description = " ".join(description)

            # Try to parse a date range from startDate/endDate if either
            # contains a range (e.g. "2020.01 - 至今"). Otherwise fall back
            # to the LLM-emitted startDate/endDate values as-is.
            start_date_input = item.get("startDate", "") or item.get("开始时间", "")
            end_date_input = item.get("endDate", "") or item.get("结束时间", "")
            if start_date_input and any(
                sep in str(start_date_input)
                for sep in ["-", "–", "—", "~", "到", "至"]
            ):
                start_date, end_date = parse_date_range(str(start_date_input))
            elif start_date_input and (
                _looks_current(end_date_input)
                or end_date_input in ("", None)
            ):
                start_date = _parse_single_date(str(start_date_input))
                end_date = "Present"
            else:
                start_date = _parse_single_date(str(start_date_input)) if start_date_input else item.get("startDate")
                if _looks_current(str(end_date_input)):
                    end_date = "Present"
                else:
                    end_date = _parse_single_date(str(end_date_input)) if end_date_input else item.get("endDate")

            transformed.append(
                {
                    "name": item.get("name", "") or item.get("公司", "") or item.get("单位", ""),
                    "position": item.get(
                        "position",
                        item.get("type", item.get("title", item.get("职位", ""))),
                    ),
                    "url": item.get("url", None),
                    "startDate": start_date,
                    "endDate": end_date,
                    "summary": item.get("summary", item.get("描述", description)),
                    "highlights": item.get("highlights", []) or item.get("亮点", []),
                }
            )
    return transformed


def transform_organizations(org_list: List) -> List[Dict]:
    transformed = []
    for item in org_list:
        if isinstance(item, dict):
            transformed.append(
                {
                    "organization": item.get("name", ""),
                    "position": item.get("role", ""),
                    "url": item.get("url", None),
                    "startDate": None,
                    "endDate": "Present",
                    "summary": None,
                    "highlights": [],
                }
            )
    return transformed


def transform_education(edu_list: List) -> List[Dict]:
    transformed = []
    for item in edu_list:
        if isinstance(item, dict):
            if "degree" in item or any(
                k in item
                for k in ("studyType", "area", "学历", "专业", "均分", "成绩", "gpa")
            ):
                # Score: English keys first, then Chinese.
                score = (
                    item.get("gpa")
                    or item.get("percentage")
                    or item.get("score")
                    or item.get("均分")
                    or item.get("成绩")
                    or item.get("gpa_score")
                    or item.get("排名")
                )
                if score is not None:
                    score = str(score)

                # studyType / area: prefer explicit fields; otherwise classify
                # the combined "degree" string with the Chinese-aware helper.
                explicit_study = (
                    item.get("studyType")
                    or item.get("学历")
                    or item.get("degree_type")
                )
                explicit_area = item.get("area") or item.get("专业")
                degree_raw = item.get("degree", "")
                if explicit_study and explicit_area:
                    study_type = explicit_study
                    area = explicit_area
                elif degree_raw:
                    study_type, area = _classify_study_type(degree_raw)
                    if explicit_study:
                        study_type = explicit_study
                    if explicit_area:
                        area = explicit_area
                else:
                    study_type = explicit_study or ""
                    area = explicit_area or ""

                start_date, end_date = parse_date_range(item.get("years", ""))
                transformed.append(
                    {
                        "institution": item.get("institution", "") or item.get("学校", ""),
                        "url": item.get("url", None),
                        "area": area or None,
                        "studyType": study_type or None,
                        "startDate": start_date,
                        "endDate": end_date,
                        "score": score,
                        "courses": item.get("courses", []) or item.get("课程", []),
                    }
                )
            else:
                transformed.append(item)
    return transformed


def transform_achievements(achievements_list: List) -> List[Dict]:
    transformed = []
    for item in achievements_list:
        if isinstance(item, dict):
            title = item.get("title", item.get("name", ""))
            awarder = item.get("awarder", item.get("organization", ""))
            summary = item.get("summary", item.get("description", None))

            transformed.append(
                {
                    "title": title,
                    "date": f"{item.get('year', '')}-01" if item.get("year") else None,
                    "awarder": awarder,
                    "summary": summary,
                }
            )
    return transformed


def transform_skills(skills_list: List) -> List[Dict]:
    transformed = []
    for item in skills_list:
        if isinstance(item, dict):
            if "category" in item:
                transformed.append(
                    {
                        "name": item.get("category", ""),
                        "level": None,
                        "keywords": item.get("keywords", []),
                    }
                )
            else:
                transformed.append(item)
    return transformed


def transform_projects(projects_list: List) -> List[Dict]:
    transformed = []
    for item in projects_list:
        if isinstance(item, dict):
            skills = []
            project_name = item.get("name", "") or item.get("项目名称", "")
            if "|" in project_name:
                name_parts = project_name.split("|")
                if len(name_parts) > 1:
                    skills_part = name_parts[1].strip()
                    skills = _split_technologies(skills_part)
                    item["name"] = name_parts[0].strip()

            technologies = item.get("technologies", []) or item.get("技术栈", [])
            if isinstance(technologies, str):
                technologies = _split_technologies(technologies)

            if not skills and technologies:
                skills = technologies

            transformed.append(
                {
                    "name": item.get("name", ""),
                    "startDate": None,
                    "endDate": None,
                    "description": item.get("description", ""),
                    "highlights": [item.get("type", "")] if item.get("type") else [],
                    "url": item.get("url", None),
                    "technologies": technologies,
                    "skills": skills,
                }
            )
    return transformed


def transform_skills_comprehensive(parsed_data: Dict) -> List[Dict]:
    skills = []

    if "skills" in parsed_data and isinstance(parsed_data["skills"], list):
        if parsed_data["skills"] and isinstance(parsed_data["skills"][0], str):
            skills.append(
                {
                    "name": "Programming Languages",
                    "level": None,
                    "keywords": parsed_data["skills"],
                }
            )
        else:
            skills.extend(transform_skills(parsed_data["skills"]))

    skill_categories = {
        "librariesFrameworks": "Libraries/Frameworks",
        "toolsPlatforms": "Tools/Platforms",
        "databases": "Databases",
    }

    for field, category_name in skill_categories.items():
        if field in parsed_data and isinstance(parsed_data[field], list):
            skills.append(
                {"name": category_name, "level": None, "keywords": parsed_data[field]}
            )

    return skills


def transform_projects_comprehensive(parsed_data: Dict) -> List[Dict]:
    projects = []

    if "projects" in parsed_data:
        projects.extend(transform_projects(parsed_data["projects"]))

    if "projectsOpenSource" in parsed_data:
        for item in parsed_data["projectsOpenSource"]:
            if isinstance(item, dict):
                skills = []
                project_name = item.get("name", "")
                if "|" in project_name:
                    name_parts = project_name.split("|")
                    if len(name_parts) > 1:
                        skills_part = name_parts[1].strip()
                        skills = [skill.strip() for skill in skills_part.split(",")]
                        item["name"] = name_parts[0].strip()

                projects.append(
                    {
                        "name": item.get("name", ""),
                        "startDate": None,
                        "endDate": None,
                        "description": item.get("summary", ""),
                        "highlights": [],
                        "url": item.get("url", None),
                        "technologies": item.get("technologies", []),
                        "skills": skills,
                    }
                )

    return projects


def parse_date_range(date_range: str) -> tuple:
    """Parse a date range into ``(start_date, end_date)`` strings.

    Handles Chinese, English-month, and bare-numeric formats. Returns
    ``YYYY-MM`` for both ends; uses the sentinel ``"Present"`` for any
    current-position keyword (中英双语). Returns ``(None, None)`` if the
    input is empty and no start token is found.

    Accepted formats:
        ``2020年1月 - 2021年6月``
        ``2020.01 - 2021.06``
        ``2020/01 - 2021/06``
        ``2020-01 - 2021-06``
        ``2020.01 - 至今``
        ``2020-2021`` (year-only range)
        ``Jan 2020 - Mar 2021``
        ``Jan 2020 - Present``
    """
    if not date_range:
        return None, None

    text = str(date_range).strip()

    # Bare current-position keyword, e.g. "至今" or "Present".
    if _looks_current(text):
        return None, "Present"

    # Detect current-position (bilingual) — "至今", "Present", etc.
    # We do this by splitting on the range separator and checking each
    # half independently so a date range with a present end still works.
    parts = _RANGE_SEP.split(text, maxsplit=1)
    if len(parts) == 2:
        start_token, end_token = parts
        start_date = _parse_single_date(start_token)
        if _looks_current(end_token):
            return start_date, "Present"
        end_date = _parse_single_date(end_token)
        if start_date is None and end_date is None:
            return None, None
        # Year-only range like "2020-2021" (no month on either side):
        # treat the end year as a full December, not January.
        if (
            end_date
            and end_token.strip().isdigit()
            and len(end_token.strip()) == 4
            and start_date
        ):
            end_date = f"{end_token.strip()}-12"
        return start_date, end_date

    # Single-date input.
    return _parse_single_date(text), None


def fetch_profile(profiles, network_names, prefix):
    """Helper function to extract profile information for a given network."""
    for network in network_names:
        profile = next(
            (p for p in profiles if p.network and p.network.lower() == network.lower()),
            None,
        )
        if profile:
            return profile


def transform_evaluation_response(
    file_name=None, resume_data=None, github_data=None, evaluation=None, role=None
):
    """
    Transform the three inputs (resume_data, github_data, evaluation) into the most important columns as a CSV row.

    Args:
        resume_data: JSONResume object containing parsed resume data
        github_data: dict containing GitHub profile data
        evaluation: EvaluationData object containing evaluation results

    Returns:
        dict: Dictionary with the most important columns for CSV output
    """
    csv_row = {}

    csv_row["file_name"] = file_name

    # Extract basic information from resume_data
    if resume_data and hasattr(resume_data, "basics") and resume_data.basics:
        basics = resume_data.basics
        csv_row["name"] = basics.name if basics.name else ""
        csv_row["email"] = basics.email if basics.email else ""
        csv_row["phone"] = basics.phone if basics.phone else ""
        csv_row["location"] = (
            f"{basics.location.city}, {basics.location.region}"
            if basics.location
            else ""
        )
        csv_row["summary"] = basics.summary if basics.summary else ""

        # Extract all profile information
        if basics.profiles:
            # Extract profiles for each platform
            github_profile = fetch_profile(basics.profiles, ["github"], "github")
            linkedin_profile = fetch_profile(basics.profiles, ["linkedin"], "linkedin")
            twitter_profile = fetch_profile(
                basics.profiles, ["twitter", "x"], "twitter"
            )
            dev_profile = fetch_profile(
                basics.profiles, ["dev community", "dev"], "dev"
            )
            behance_profile = fetch_profile(basics.profiles, ["behance"], "behance")

            # Add GitHub profile columns
            if github_profile:
                csv_row["github_url"] = github_profile.url
                csv_row["github_username"] = (
                    github_profile.username if github_profile.username else ""
                )
            else:
                csv_row["github_url"] = ""
                csv_row["github_username"] = ""

            # Add LinkedIn profile columns
            if linkedin_profile:
                csv_row["linkedin_url"] = linkedin_profile.url
                csv_row["linkedin_username"] = (
                    linkedin_profile.username if linkedin_profile.username else ""
                )
            else:
                csv_row["linkedin_url"] = ""
                csv_row["linkedin_username"] = ""

            # Add Twitter/X profile columns
            if twitter_profile:
                csv_row["twitter_url"] = twitter_profile.url
                csv_row["twitter_username"] = (
                    twitter_profile.username if twitter_profile.username else ""
                )
            else:
                csv_row["twitter_url"] = ""
                csv_row["twitter_username"] = ""

            # Add DEV Community profile columns
            if dev_profile:
                csv_row["dev_url"] = dev_profile.url
                csv_row["dev_username"] = (
                    dev_profile.username if dev_profile.username else ""
                )
            else:
                csv_row["dev_url"] = ""
                csv_row["dev_username"] = ""

            # Add Behance profile columns
            if behance_profile:
                csv_row["behance_url"] = behance_profile.url
                csv_row["behance_username"] = (
                    behance_profile.username if behance_profile.username else ""
                )
            else:
                csv_row["behance_url"] = ""
                csv_row["behance_username"] = ""
        else:
            # Initialize empty profile columns
            for prefix in ["github", "linkedin", "twitter", "dev", "behance"]:
                csv_row[f"{prefix}_url"] = ""
                csv_row[f"{prefix}_username"] = ""

    # Extract work experience summary
    if resume_data and hasattr(resume_data, "work") and resume_data.work:
        work_experience = resume_data.work
        csv_row["total_work_experience"] = len(work_experience)

        # Get most recent position
        if work_experience:
            latest_work = work_experience[0]  # Assuming sorted by date
            csv_row["current_position"] = (
                latest_work.position if latest_work.position else ""
            )
            csv_row["current_company"] = latest_work.name if latest_work.name else ""
        else:
            csv_row["current_position"] = ""
            csv_row["current_company"] = ""
    else:
        csv_row["total_work_experience"] = 0
        csv_row["current_position"] = ""
        csv_row["current_company"] = ""

    # Extract education summary
    if resume_data and hasattr(resume_data, "education") and resume_data.education:
        education = resume_data.education
        csv_row["total_education"] = len(education)

        # Get highest education level
        if education:
            highest_edu = education[0]  # Assuming sorted by date
            csv_row["highest_degree"] = (
                highest_edu.studyType if highest_edu.studyType else ""
            )
            csv_row["institution"] = (
                highest_edu.institution if highest_edu.institution else ""
            )
        else:
            csv_row["highest_degree"] = ""
            csv_row["institution"] = ""
    else:
        csv_row["total_education"] = 0
        csv_row["highest_degree"] = ""
        csv_row["institution"] = ""

    # Extract skills summary
    if resume_data and hasattr(resume_data, "skills") and resume_data.skills:
        skills = resume_data.skills
        all_skills = []
        for skill_category in skills:
            if skill_category.keywords:
                all_skills.extend(skill_category.keywords)
        csv_row["total_skills"] = len(all_skills)
        csv_row["skills_list"] = ", ".join(all_skills[:10])  # Top 10 skills
    else:
        csv_row["total_skills"] = 0
        csv_row["skills_list"] = ""

    # Extract projects summary
    if resume_data and hasattr(resume_data, "projects") and resume_data.projects:
        projects = resume_data.projects
        csv_row["total_projects"] = len(projects)
    else:
        csv_row["total_projects"] = 0

    # Extract GitHub data
    if github_data:
        profile = github_data.get("profile", {})
        csv_row["github_repos"] = profile.get("public_repos", 0)
        csv_row["github_followers"] = profile.get("followers", 0)
        csv_row["github_following"] = profile.get("following", 0)
        csv_row["github_created_at"] = profile.get("created_at", "")
        csv_row["github_bio"] = profile.get("bio", "")
    else:
        csv_row["github_repos"] = 0
        csv_row["github_followers"] = 0
        csv_row["github_following"] = 0
        csv_row["github_created_at"] = ""
        csv_row["github_bio"] = ""

    # Extract evaluation scores (one pair of columns per role category)
    category_keys = [c.key for c in role.categories] if role else []
    if evaluation and hasattr(evaluation, "scores"):
        scores = evaluation.scores
        total_score = 0
        total_max = 0
        for key in category_keys:
            cat = getattr(scores, key, None)
            if cat is None:
                csv_row[f"{key}_score"] = "N/A"
                csv_row[f"{key}_max"] = "N/A"
                continue
            csv_row[f"{key}_score"] = cat.score
            csv_row[f"{key}_max"] = cat.max
            total_score += cat.score
            total_max += cat.max

        csv_row["total_score"] = total_score
        csv_row["total_max"] = total_max
    else:
        for key in category_keys:
            csv_row[f"{key}_score"] = "N/A"
            csv_row[f"{key}_max"] = "N/A"
        csv_row["total_score"] = "N/A"
        csv_row["total_max"] = "N/A"

    # Extract bonus points and deductions
    if evaluation and hasattr(evaluation, "bonus_points"):
        csv_row["bonus_points"] = evaluation.bonus_points.total
        csv_row["bonus_breakdown"] = evaluation.bonus_points.breakdown
    else:
        csv_row["bonus_points"] = 0
        csv_row["bonus_breakdown"] = ""

    if evaluation and hasattr(evaluation, "deductions"):
        csv_row["deductions"] = evaluation.deductions.total
        csv_row["deduction_reasons"] = evaluation.deductions.reasons
    else:
        csv_row["deductions"] = 0
        csv_row["deduction_reasons"] = ""

    # Extract key strengths and areas for improvement
    if evaluation and hasattr(evaluation, "key_strengths"):
        csv_row["key_strengths"] = "; ".join(evaluation.key_strengths)
    else:
        csv_row["key_strengths"] = ""

    if evaluation and hasattr(evaluation, "areas_for_improvement"):
        csv_row["areas_for_improvement"] = "; ".join(evaluation.areas_for_improvement)
    else:
        csv_row["areas_for_improvement"] = ""

    return csv_row


def convert_json_resume_to_text(resume_data: JSONResume) -> str:
    text_parts = []

    if resume_data.basics:
        basics = resume_data.basics
        text_parts.append("=== BASIC INFORMATION / 个人信息 ===")
        text_parts.append(f"Name / 姓名: {basics.name or 'Not provided / 未提供'}")
        text_parts.append(f"Email / 邮箱: {basics.email or 'Not provided / 未提供'}")
        text_parts.append(f"Phone / 电话: {basics.phone or 'Not provided / 未提供'}")
        text_parts.append(f"Website / 个人网站: {basics.url or 'Not provided / 未提供'}")

        if basics.summary:
            text_parts.append(f"Summary / 个人简介: {basics.summary}")

        if basics.location:
            loc = basics.location
            location_parts = []
            if loc.address:
                location_parts.append(loc.address)
            if loc.city:
                location_parts.append(loc.city)
            if loc.region:
                location_parts.append(loc.region)
            if loc.postalCode:
                location_parts.append(loc.postalCode)
            if loc.countryCode:
                location_parts.append(loc.countryCode)

            if location_parts:
                text_parts.append(f"Location / 所在地: {', '.join(location_parts)}")

        if basics.profiles:
            text_parts.append("Profiles / 个人主页:")
            for profile in basics.profiles:
                text_parts.append(
                    f"  - {profile.network}: {profile.username} ({profile.url})"
                )

    if resume_data.work:
        text_parts.append("\n=== WORK EXPERIENCE / 工作经历 ===")
        for i, work in enumerate(resume_data.work, 1):
            text_parts.append(f"{i}. {work.position} at {work.name}")
            text_parts.append(f"   Period / 时间: {work.startDate} - {work.endDate}")
            if work.url:
                text_parts.append(f"   Website / 链接: {work.url}")
            if work.summary:
                text_parts.append(f"   Description / 描述: {work.summary}")
            if work.highlights:
                text_parts.append("   Key Achievements / 主要成就:")
                for highlight in work.highlights:
                    text_parts.append(f"     • {highlight}")

    if resume_data.education:
        text_parts.append("\n=== EDUCATION / 教育经历 ===")
        for i, edu in enumerate(resume_data.education, 1):
            text_parts.append(f"{i}. {edu.studyType} in {edu.area}")
            text_parts.append(f"   Institution / 学校: {edu.institution}")
            text_parts.append(f"   Period / 时间: {edu.startDate} - {edu.endDate}")
            if edu.score:
                text_parts.append(f"   Score / 成绩: {edu.score}")
            if edu.url:
                text_parts.append(f"   Website / 链接: {edu.url}")
            if edu.courses:
                text_parts.append(f"   Courses / 主修课程: {', '.join(edu.courses)}")

    if resume_data.skills:
        text_parts.append("\n=== SKILLS / 技能 ===")
        for skill in resume_data.skills:
            text_parts.append(f"• {skill.name}")
            if skill.level:
                text_parts.append(f"  Level / 水平: {skill.level}")
            if skill.keywords:
                text_parts.append(f"  Keywords / 关键词: {', '.join(skill.keywords)}")

    if resume_data.projects:
        text_parts.append("\n=== PROJECTS / 项目经历 ===")
        for i, project in enumerate(resume_data.projects, 1):
            text_parts.append(f"{i}. {project.name}")
            if project.startDate and project.endDate:
                text_parts.append(f"   Period / 时间: {project.startDate} - {project.endDate}")
            if project.description:
                text_parts.append(f"   Description / 描述: {project.description}")
            if project.url:
                text_parts.append(f"   URL / 链接: {project.url}")
            if project.highlights:
                text_parts.append("   Highlights / 亮点:")
                for highlight in project.highlights:
                    text_parts.append(f"     • {highlight}")

    if resume_data.awards:
        text_parts.append("\n=== AWARDS / 荣誉奖项 ===")
        for award in resume_data.awards:
            text_parts.append(f"• {award.title} - {award.awarder} ({award.date})")
            if award.summary:
                text_parts.append(f"  {award.summary}")

    if resume_data.certificates:
        text_parts.append("\n=== CERTIFICATES / 证书 ===")
        for cert in resume_data.certificates:
            text_parts.append(f"• {cert.name} - {cert.issuer} ({cert.date})")
            if cert.url:
                text_parts.append(f"  URL / 链接: {cert.url}")

    if resume_data.publications:
        text_parts.append("\n=== PUBLICATIONS / 出版物 ===")
        for pub in resume_data.publications:
            text_parts.append(f"• {pub.name} - {pub.publisher} ({pub.releaseDate})")
            if pub.url:
                text_parts.append(f"  URL / 链接: {pub.url}")
            if pub.summary:
                text_parts.append(f"  {pub.summary}")

    if resume_data.languages:
        text_parts.append("\n=== LANGUAGES / 语言能力 ===")
        for lang in resume_data.languages:
            text_parts.append(f"• {lang.language} - {lang.fluency}")

    if resume_data.interests:
        text_parts.append("\n=== INTERESTS / 兴趣爱好 ===")
        for interest in resume_data.interests:
            text_parts.append(f"• {interest.name}")
            if interest.keywords:
                text_parts.append(f"  Keywords / 关键词: {', '.join(interest.keywords)}")

    if resume_data.references:
        text_parts.append("\n=== REFERENCES / 推荐人 ===")
        for ref in resume_data.references:
            text_parts.append(f"• {ref.name}")
            if ref.reference:
                text_parts.append(f"  {ref.reference}")

    if resume_data.volunteer:
        text_parts.append("\n=== VOLUNTEER EXPERIENCE / 志愿服务 ===")
        for volunteer in resume_data.volunteer:
            text_parts.append(f"• {volunteer.position} at {volunteer.organization}")
            text_parts.append(f"  Period / 时间: {volunteer.startDate} - {volunteer.endDate}")
            if volunteer.url:
                text_parts.append(f"  Website / 链接: {volunteer.url}")
            if volunteer.summary:
                text_parts.append(f"  Description / 描述: {volunteer.summary}")
            if volunteer.highlights:
                text_parts.append("  Highlights / 亮点:")
                for highlight in volunteer.highlights:
                    text_parts.append(f"    • {highlight}")

    return "\n".join(text_parts)


def convert_github_data_to_text(github_data: dict) -> str:
    github_text = "\n\n=== GITHUB DATA ===\n"

    if "profile" in github_data:
        profile = github_data["profile"]
        github_text += f"GitHub Profile:\n"
        github_text += f"- Username: {profile.get('username', 'N/A')}\n"
        github_text += f"- Name: {profile.get('name', 'N/A')}\n"
        github_text += f"- Bio: {profile.get('bio', 'N/A')}\n"
        github_text += f"- Public Repositories: {profile.get('public_repos', 'N/A')}\n"
        github_text += f"- Followers: {profile.get('followers', 'N/A')}\n"
        github_text += f"- Following: {profile.get('following', 'N/A')}\n"
        github_text += f"- Account Created: {profile.get('created_at', 'N/A')}\n"
        github_text += f"- Last Updated: {profile.get('updated_at', 'N/A')}\n"

    if "projects" in github_data:
        projects = github_data["projects"]
        github_text += f"\nGitHub Projects ({len(projects)} total):\n"
        for i, project in enumerate(projects[:10], 1):
            github_text += f"{i}. {project.get('name', 'N/A')}\n"
            github_text += f"   Description: {project.get('description', 'N/A')}\n"
            github_text += f"   URL: {project.get('github_url', 'N/A')}\n"
            if "github_details" in project:
                details = project["github_details"]
                github_text += f"   Stars: {details.get('stars', 'N/A')}\n"
                github_text += f"   Forks: {details.get('forks', 'N/A')}\n"
                github_text += f"   Language: {details.get('language', 'N/A')}\n"
            github_text += "\n"

    return github_text


def convert_blog_data_to_text(blog_data: dict) -> str:
    blog_text = "\n\n=== BLOG DATA ===\n"
    blog_text += f"Total Blogs Found: {blog_data.get('total_blogs', 'N/A')}\n"
    blog_text += f"Blog Score: {blog_data.get('blog_score', 'N/A')}/10.0\n"
    blog_text += f"Blog Details: {blog_data.get('blog_details', 'N/A')}\n"

    if "blogs" in blog_data:
        blog_text += "\nBlog URLs Found:\n"
        for i, blog in enumerate(blog_data["blogs"][:5], 1):
            blog_text += f"{i}. {blog.get('url', 'N/A')}\n"
            blog_text += f"   Score: {blog.get('score', 'N/A')}/10.0\n"
            blog_text += f"   Details: {blog.get('details', 'N/A')}\n"
            blog_text += "\n"

    return blog_text
