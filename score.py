import os
import sys
import json

# 修复 Windows 控制台 Unicode 错误
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')    # 把标准输出重新配置成 UTF-8 编码。这样打印中文、emoji 时不容易报错
    except AttributeError:
        pass

# 修复 Python 3.14 下 protobuf 可能出现的类型错误
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import logging
import csv

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import argparse

from pdf_dispatcher import extract_resume_data, UnsupportedFormatError
from code_platforms import fetch_code_platform_data
from models import JSONResume, build_evaluation_model
from typing import List, Optional, Dict
from evaluator import ResumeEvaluator
from roles import Role, load_role, list_available_roles, scaffold_role
from pathlib import Path
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS
from transform import (
    transform_evaluation_response,
    convert_json_resume_to_text,
    convert_github_data_to_text,
    convert_blog_data_to_text,
)
from config import DEVELOPMENT_MODE

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)5s - %(lineno)5d - %(funcName)33s - %(levelname)5s - %(message)s",
)


def print_evaluation_results(
    evaluation, role: Role, candidate_name: str = "Candidate"
):
    """Print evaluation results in a readable format.

    Output is bilingual (English label / 中文标签) so both English and
    Chinese role reports look natural. Category labels come from
    ``role.json`` and already include both languages when the role is
    the Chinese variant.
    """
    print("\n" + "=" * 80)
    print(f"📊 RESUME EVALUATION RESULTS / 简历评估结果: {candidate_name}")
    print("=" * 80)

    if not evaluation:
        print("❌ No evaluation data available / 暂无评估数据")
        return

    # Calculate overall score
    total_score = 0
    max_score = 0

    if hasattr(evaluation, "scores") and evaluation.scores:
        for category_name, category_data in evaluation.scores.model_dump().items():
            category_score = min(category_data["score"], category_data["max"])
            total_score += category_score
            max_score += category_data["max"]

            # Log warning if score was capped
            if category_score < category_data["score"]:
                print(
                    f"⚠️  Warning / 警告: {category_name} score capped from {category_data['score']} to {category_score} (max: {category_data['max']})"
                )

    # Add bonus points
    if hasattr(evaluation, "bonus_points") and evaluation.bonus_points:
        total_score += evaluation.bonus_points.total

    # Subtract deductions
    if hasattr(evaluation, "deductions") and evaluation.deductions:
        total_score -= evaluation.deductions.total

    # Ensure total score doesn't exceed maximum possible score
    max_possible_score = max_score + role.bonus_max
    if total_score > max_possible_score:
        total_score = max_possible_score
        print(f"⚠️  Warning / 警告: Total score capped at maximum possible value / 总分已封顶")

    # Overall Score
    print(f"\n🎯 OVERALL SCORE / 综合评分: {total_score:.1f}/{max_score}")

    # Detailed Scores
    print("\n📈 DETAILED SCORES / 详细评分:")
    print("-" * 60)

    if hasattr(evaluation, "scores") and evaluation.scores:
        for category in role.categories:
            cat_score = getattr(evaluation.scores, category.key, None)
            if not cat_score:
                continue
            capped_score = min(cat_score.score, category.max)
            print(f"{category.icon} {category.label}: {capped_score}/{cat_score.max}")
            print(f"   Evidence / 评分依据: {cat_score.evidence}")
            print()

    # Bonus Points
    if hasattr(evaluation, "bonus_points") and evaluation.bonus_points:
        print(f"\n⭐ BONUS POINTS / 加分项: {evaluation.bonus_points.total}")
        print("-" * 30)
        print(f"   {evaluation.bonus_points.breakdown}")

    # Deductions
    if (
        hasattr(evaluation, "deductions")
        and evaluation.deductions
        and evaluation.deductions.total > 0
    ):
        print(f"\n⚠️  DEDUCTIONS / 扣分项: -{evaluation.deductions.total}")
        print("-" * 30)
        if evaluation.deductions.reasons:
            print(f"   {evaluation.deductions.reasons}")

    # Key Strengths
    if hasattr(evaluation, "key_strengths") and evaluation.key_strengths:
        print(f"\n✅ KEY STRENGTHS / 核心优势:")
        print("-" * 30)
        for i, strength in enumerate(evaluation.key_strengths, 1):
            print(f"  {i}. {strength}")

    # Areas for Improvement
    if (
        hasattr(evaluation, "areas_for_improvement")
        and evaluation.areas_for_improvement
    ):
        print(f"\n🔧 AREAS FOR IMPROVEMENT / 待提升方向:")
        print("-" * 30)
        for i, area in enumerate(evaluation.areas_for_improvement, 1):
            print(f"  {i}. {area}")

    print("\n" + "=" * 80)


def _evaluate_resume(
    resume_data: JSONResume,
    role: Role,
    evaluation_model,
    github_data: dict = None,
    blog_data: dict = None,
):
    """Evaluate the resume using AI and display results."""

    model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL)
    evaluator = ResumeEvaluator(
        role=role,
        evaluation_model=evaluation_model,
        model_name=DEFAULT_MODEL,
        model_params=model_params,
    )

    # Convert JSON resume data to text
    resume_text = convert_json_resume_to_text(resume_data)

    # Add GitHub data if available
    if github_data:
        github_text = convert_github_data_to_text(github_data)
        resume_text += github_text

    # Add blog data if available
    if blog_data:
        blog_text = convert_blog_data_to_text(blog_data)
        resume_text += blog_text

    # Evaluate the enhanced resume
    evaluation_result = evaluator.evaluate_resume(resume_text)

    # print(evaluation_result)

    return evaluation_result


def is_valid_resume_data(resume_data: JSONResume) -> bool:
    """Check if the resume data has at least some extracted core content."""
    if not resume_data:
        return False
    core_sections = [
        resume_data.basics,
        resume_data.work,
        resume_data.education,
        resume_data.skills,
        resume_data.projects,
    ]
    return any(section is not None for section in core_sections)


def find_profile(profiles, network):
    if not profiles:
        return None
    return next(
        (p for p in profiles if p.network and p.network.lower() == network.lower()),
        None,
    )


def main(resume_path, role: Role):
    evaluation_model = build_evaluation_model(role)

    # Cache filename uses the basename without ANY extension, so it works
    # uniformly for .pdf, .docx, .doc, .png, .jpg, .jpeg, .tif, .tiff.
    base = os.path.splitext(os.path.basename(resume_path))[0]
    cache_filename = f"cache/resumecache_{base}.json"
    code_platforms_cache_filename = f"cache/code_platforms_cache_{base}.json"

    resume_data = None
    cache_loaded = False

    # Check if cache exists and we're in development mode
    if DEVELOPMENT_MODE and os.path.exists(cache_filename):
        print(f"Loading cached data from {cache_filename}")
        try:
            cached_data = json.loads(Path(cache_filename).read_text(encoding="utf-8"))
            loaded_resume = JSONResume(**cached_data)
            if not is_valid_resume_data(loaded_resume):
                raise ValueError("Cached resume data contains no core content")
            resume_data = loaded_resume
            cache_loaded = True
        except Exception as e:
            print(f"⚠️ Warning: Invalid cache file {cache_filename}: {e}")
            print("Ignoring cache and reprocessing PDF...")
            try:
                os.remove(cache_filename)
            except Exception as delete_err:
                print(
                    f"Failed to delete invalid cache file {cache_filename}: {delete_err}"
                )

    if not cache_loaded:
        logger.debug(
            f"Extracting data from resume {resume_path}"
            + (" and caching to " + cache_filename if DEVELOPMENT_MODE else "")
        )
        try:
            resume_data = extract_resume_data(resume_path)
        except UnsupportedFormatError as e:
            print(f"Error: {e}")
            return None

        if resume_data is None:
            return None

        if DEVELOPMENT_MODE:
            if is_valid_resume_data(resume_data):
                os.makedirs(os.path.dirname(cache_filename), exist_ok=True)
                Path(cache_filename).write_text(
                    json.dumps(resume_data.model_dump(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            else:
                logger.warning(
                    "Newly extracted resume data is empty/invalid. Skipping cache write."
                )

    # Check if cache exists and we're in development mode
    github_data = {}
    code_platforms_cache_loaded = False
    if DEVELOPMENT_MODE and os.path.exists(code_platforms_cache_filename):
        print(f"Loading cached data from {code_platforms_cache_filename}")
        try:
            loaded_code_platforms = json.loads(
                Path(code_platforms_cache_filename).read_text(encoding="utf-8")
            )
            if (
                not isinstance(loaded_code_platforms, dict)
                or not loaded_code_platforms
                or "profile" not in loaded_code_platforms
            ):
                raise ValueError("Cached code-platform data is invalid or empty")
            github_data = loaded_code_platforms
            code_platforms_cache_loaded = True
        except Exception as e:
            print(f"⚠️ Warning: Invalid code-platform cache file {code_platforms_cache_filename}: {e}")
            print("Ignoring code-platform cache and refetching...")
            try:
                os.remove(code_platforms_cache_filename)
            except Exception as delete_err:
                print(
                    f"Failed to delete invalid code-platform cache file {code_platforms_cache_filename}: {delete_err}"
                )

    if not code_platforms_cache_loaded:
        # Fetch GitHub + Gitee in one unified call. The orchestrator
        # only fetches platforms whose URL is present in resume.basics.profiles.
        profiles = []
        if resume_data and hasattr(resume_data, "basics") and resume_data.basics:
            profiles = resume_data.basics.profiles or []

        has_code_platform_profile = any(
            isinstance(p, dict)
            and (
                "github.com" in (p.get("url") or "").lower()
                or "gitee.com" in (p.get("url") or "").lower()
            )
            for p in profiles
        )

        if has_code_platform_profile:
            print(
                f"Fetching code-platform data (GitHub + Gitee)"
                + (
                    " and caching to " + code_platforms_cache_filename
                    if DEVELOPMENT_MODE
                    else ""
                )
            )
            github_data = fetch_code_platform_data(
                resume_data.basics if resume_data else None,
                position_title=role.position_title,
            )

            if (
                DEVELOPMENT_MODE
                and github_data
                and isinstance(github_data, dict)
                and "profile" in github_data
            ):
                os.makedirs(os.path.dirname(code_platforms_cache_filename), exist_ok=True)
                Path(code_platforms_cache_filename).write_text(
                    json.dumps(github_data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

    score = _evaluate_resume(resume_data, role, evaluation_model, github_data)

    # Get candidate name for display
    candidate_name = os.path.splitext(os.path.basename(resume_path))[0]
    if (
        resume_data
        and hasattr(resume_data, "basics")
        and resume_data.basics
        and resume_data.basics.name
    ):
        candidate_name = resume_data.basics.name

    # Print evaluation results in readable format
    print_evaluation_results(score, role, candidate_name)

    if DEVELOPMENT_MODE:
        csv_row = transform_evaluation_response(
            file_name=os.path.basename(resume_path),
            evaluation=score,
            resume_data=resume_data,
            github_data=github_data,
            role=role,
        )

        # Write CSV row to a role-specific file, since each role's columns differ.
        csv_path = f"resume_evaluations_{role.name}.csv"
        file_exists = os.path.exists(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = list(csv_row.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write headers if file doesn't exist
            if not file_exists:
                writer.writeheader()

            # Write the row
            writer.writerow(csv_row)

    return score


if __name__ == "__main__":
    available_roles = list_available_roles()
    parser = argparse.ArgumentParser(
        description="Score a resume against a role's rubric."
    )
    parser.add_argument(
        "resume_path",
        nargs="?",
        help=(
            "Path to the resume file to evaluate. "
            "Supported formats: .pdf, .docx, .doc, .png, .jpg, .jpeg, .tif, .tiff."
        ),
    )
    parser.add_argument(
        "--role",
        help="Role to score against (a directory name under roles/). "
        + (f"Available: {', '.join(available_roles)}" if available_roles else ""),
    )
    parser.add_argument(
        "--init-role",
        metavar="NAME",
        help="Scaffold a new role directory under roles/ with basic template "
        "files, then exit (does not score a resume).",
    )
    args = parser.parse_args()

    # Scaffold mode: create a new role and exit.
    if args.init_role:
        try:
            role_dir = scaffold_role(args.init_role)
        except ValueError as e:
            print(f"Error: {e}")
            exit(1)
        print(f"✅ Created role '{args.init_role}' at {role_dir}")
        print("   Edit role.json, criteria.jinja and system_message.jinja, then run:")
        print(f"   python score.py <resume_path> --role {args.init_role}")
        exit(0)

    # Scoring mode: both resume_path and --role are required.
    if not args.resume_path or not args.role:
        parser.error("resume_path and --role are required (or use --init-role NAME)")

    if not os.path.exists(args.resume_path):
        print(f"Error: File '{args.resume_path}' does not exist.")
        exit(1)

    try:
        role = load_role(args.role)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    main(args.resume_path, role)
