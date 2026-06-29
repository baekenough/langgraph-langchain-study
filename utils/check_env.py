"""
API 키 검증 스크립트 — LangChain/LangGraph 학습 프로젝트

사용법:
    uv run python utils/check_env.py            # 전체 검증 (라이브 API 호출 포함)
    uv run python utils/check_env.py --offline  # 키 존재 여부만 확인 (API 호출 없음)
    uv run python utils/check_env.py --check-only  # --offline 과 동일

종료 코드:
    0  — 필수 키(OpenRouter, OpenAI) 모두 정상
    1  — 필수 키 하나 이상 실패 또는 .env 파일 없음
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ─── 상태 심볼 ────────────────────────────────────────────────────────────────

STATUS_OK = "✓ OK"
STATUS_FAIL = "✗ FAIL"
STATUS_SKIP = "– SKIP"


def mask_key(key: str) -> str:
    """API 키를 마스킹하여 반환합니다 (앞 6자 + '...' + 뒤 4자)."""
    if len(key) <= 10:
        return "***"
    return f"{key[:6]}...{key[-4:]}"


# ─── 키 검증 함수들 ───────────────────────────────────────────────────────────


def check_openrouter(offline: bool) -> tuple[str, str]:
    """OPENROUTER_API_KEY 존재 여부 및 유효성을 검증합니다."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        return STATUS_FAIL, "OPENROUTER_API_KEY 미설정 — .env 파일에 추가하세요"

    if offline:
        return STATUS_OK, f"키 존재 ({mask_key(key)}) — 오프라인 모드, API 호출 생략"

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model="openai/gpt-4o-mini",
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            max_tokens=5,
        )
        resp = llm.invoke("ping")
        if resp.content:
            return STATUS_OK, f"키 유효 ({mask_key(key)}) — 응답 수신 성공"
        return STATUS_FAIL, f"키 ({mask_key(key)}) — 빈 응답 반환됨"
    except Exception as exc:
        short = str(exc)[:120]
        return STATUS_FAIL, f"키 ({mask_key(key)}) — 오류: {short}"


def check_openai(offline: bool) -> tuple[str, str]:
    """OPENAI_API_KEY 존재 여부 및 유효성을 검증합니다."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return STATUS_FAIL, "OPENAI_API_KEY 미설정 — 임베딩(RAG Phase 13)에 필요합니다"

    if offline:
        return STATUS_OK, f"키 존재 ({mask_key(key)}) — 오프라인 모드, API 호출 생략"

    try:
        from langchain_openai import OpenAIEmbeddings

        emb = OpenAIEmbeddings(model="text-embedding-3-small", api_key=key)
        vec = emb.embed_query("ping")
        if len(vec) > 0:
            return STATUS_OK, f"키 유효 ({mask_key(key)}) — {len(vec)}차원 벡터 수신"
        return STATUS_FAIL, f"키 ({mask_key(key)}) — 빈 벡터 반환됨"
    except Exception as exc:
        short = str(exc)[:120]
        return STATUS_FAIL, f"키 ({mask_key(key)}) — 오류: {short}"


def check_langsmith(offline: bool) -> tuple[str, str]:
    """LANGSMITH_API_KEY 존재 여부 및 유효성을 검증합니다 (선택)."""
    key = os.environ.get("LANGSMITH_API_KEY", "")
    if not key:
        return STATUS_SKIP, "LANGSMITH_API_KEY 미설정 — 트레이싱 비활성 (선택 사항)"

    if offline:
        return STATUS_OK, f"키 존재 ({mask_key(key)}) — 오프라인 모드, API 호출 생략"

    try:
        from langsmith import Client

        client = Client(api_key=key)
        list(client.list_projects(limit=1))
        return STATUS_OK, f"키 유효 ({mask_key(key)}) — LangSmith 연결 성공"
    except ImportError:
        return STATUS_SKIP, "langsmith 패키지 미설치 — `uv add langsmith` 실행 필요"
    except Exception as exc:
        short = str(exc)[:120]
        return STATUS_FAIL, f"키 ({mask_key(key)}) — 오류: {short}"


def check_tavily(offline: bool) -> tuple[str, str]:
    """TAVILY_API_KEY 존재 여부 및 유효성을 검증합니다 (선택)."""
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        return STATUS_SKIP, "TAVILY_API_KEY 미설정 — 웹 검색 도구 비활성 (선택 사항)"

    if offline:
        return STATUS_OK, f"키 존재 ({mask_key(key)}) — 오프라인 모드, API 호출 생략"

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=key)
        result = client.search("langchain", max_results=1)
        if result:
            return STATUS_OK, f"키 유효 ({mask_key(key)}) — Tavily 검색 성공"
        return STATUS_FAIL, f"키 ({mask_key(key)}) — 빈 결과 반환됨"
    except ImportError:
        return STATUS_SKIP, "tavily 패키지 미설치 — `uv add tavily-python` 실행 필요"
    except Exception as exc:
        short = str(exc)[:120]
        return STATUS_FAIL, f"키 ({mask_key(key)}) — 오류: {short}"


# ─── 출력 헬퍼 ────────────────────────────────────────────────────────────────


def print_row(label: str, status: str, message: str) -> None:
    """단일 검증 결과 행을 출력합니다."""
    print(f"  {status:<8}  {label:<24}  {message}")


# ─── 메인 ────────────────────────────────────────────────────────────────────


def main() -> int:
    """API 키 검증을 수행하고 종료 코드를 반환합니다."""
    parser = argparse.ArgumentParser(
        description="LangChain 프로젝트 API 키 검증 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "예시:\n"
            "  uv run python utils/check_env.py            # 라이브 API 호출 포함\n"
            "  uv run python utils/check_env.py --offline  # 키 존재 여부만 확인\n"
        ),
    )
    parser.add_argument(
        "--offline",
        "--check-only",
        dest="offline",
        action="store_true",
        help="API 호출 없이 키 존재 여부만 확인합니다",
    )
    args = parser.parse_args()

    # .env 파일 로드
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        print(f"\n[오류] .env 파일을 찾을 수 없습니다: {env_path}")
        print("  → .env.example을 복사하여 .env를 만들고 API 키를 입력하세요:")
        print("       cp .env.example .env\n")
        return 1

    load_dotenv(dotenv_path=env_path)

    mode_label = "오프라인 (존재 여부만)" if args.offline else "라이브 (API 호출 포함)"
    print()
    print(f"=== API 키 검증 — {mode_label} ===")
    print()

    # 필수 키 검증
    print("[ 필수 ]")
    or_status, or_msg = check_openrouter(args.offline)
    oa_status, oa_msg = check_openai(args.offline)
    print_row("OpenRouter (채팅)", or_status, or_msg)
    print_row("OpenAI (임베딩)", oa_status, oa_msg)

    # 선택 키 검증
    print()
    print("[ 선택 ]")
    ls_status, ls_msg = check_langsmith(args.offline)
    tv_status, tv_msg = check_tavily(args.offline)
    print_row("LangSmith (트레이싱)", ls_status, ls_msg)
    print_row("Tavily (웹 검색)", tv_status, tv_msg)

    # 요약
    required_ok = or_status == STATUS_OK and oa_status == STATUS_OK
    optional_ok = sum(1 for s in (ls_status, tv_status) if s == STATUS_OK)
    optional_skipped = sum(1 for s in (ls_status, tv_status) if s == STATUS_SKIP)

    print()
    print("─" * 60)
    if required_ok:
        print("결과: 필수 키 모두 정상 — 학습을 시작할 수 있습니다.")
    else:
        failed = [
            name
            for name, status in [("OpenRouter", or_status), ("OpenAI", oa_status)]
            if status != STATUS_OK
        ]
        print(f"결과: 필수 키 실패 — {', '.join(failed)}")
        print("      위 항목의 키를 .env 파일에 설정하세요.")

    suffix = f", {optional_skipped}개 미설정" if optional_skipped else ""
    print(f"       선택 키: {optional_ok}개 정상{suffix}")
    print()

    return 0 if required_ok else 1


if __name__ == "__main__":
    sys.exit(main())
