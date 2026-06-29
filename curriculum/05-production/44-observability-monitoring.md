# Phase 44: 관측성·모니터링

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 2시간 30분 |
| 난이도 | ★★★☆☆ |
| 선행 학습 | Phase 35 (LangSmith 트레이싱), Phase 42 (FastAPI 배포) |

---

## 🎯 학습 목표

- 구조화 로깅(Structured Logging)으로 프로덕션에서 의미 있는 로그를 수집할 수 있습니다.
- LangSmith를 운영 모니터링 도구로 활용하여 지연·비용·오류율을 추적할 수 있습니다.
- OpenTelemetry의 개념과 LangChain 통합 방식을 설명할 수 있습니다.
- 사용자 피드백을 수집하고 평가 루프에 통합할 수 있습니다.
- 프로덕션 환경에서 디버깅하는 전략을 설명할 수 있습니다.

> **참고**: LangSmith API 및 OpenTelemetry 통합은 업데이트될 수 있습니다. 공식 문서를 함께 확인하세요.
> - LangSmith: https://docs.smith.langchain.com
> - LangChain OpenTelemetry: https://python.langchain.com/docs/concepts/tracing

---

## 📚 핵심 개념

### 1. 관측성의 세 기둥

프로덕션 LLM 시스템에는 세 가지 관측성 데이터가 필요합니다.

| 기둥 | 설명 | LangChain 도구 |
|------|------|--------------|
| **추적(Traces)** | 요청 흐름, 노드별 입출력, 체인 구조 | LangSmith, OpenTelemetry |
| **지표(Metrics)** | 지연, 토큰 비용, 오류율, 처리량 | LangSmith 대시보드, Prometheus |
| **로그(Logs)** | 이벤트 기록, 예외, 구조화 컨텍스트 | Python logging, structlog |

### 2. LangSmith 운영 모니터링

Phase 35에서 LangSmith를 개발 도구로 배웠다면, 프로덕션에서는 다음을 추가로 활용합니다.

- **프로젝트별 모니터링**: 환경(dev/staging/prod)을 별도 프로젝트로 분리
- **Run 태그**: 특정 사용자·기능·버전별로 필터링
- **피드백 API**: 사용자 평점(👍/👎)을 트레이스에 연결
- **사용자 정의 평가**: 자동화된 LLM-as-Judge 파이프라인

### 3. OpenTelemetry 개념

OpenTelemetry(OTel)는 벤더 중립 관측성 표준입니다. LangChain은 OTel SDK와 통합을 지원하여 Jaeger, Datadog, Grafana Tempo 등 다양한 백엔드로 트레이스를 전송할 수 있습니다.

```
LangChain App
    │  (OTel SDK)
    ▼
OTel Collector
    ├──► Jaeger (시각화)
    ├──► Prometheus (지표)
    └──► LangSmith (LLM 특화)
```

---

## 💻 코드 예제

### 예제 1: 구조화 로깅 설정

```python
# app/logging_config.py
import logging
import json
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """
    Structured JSON log formatter for production use.

    JSON logs are easier to parse, filter, and query in log aggregation systems
    (e.g., Loki, CloudWatch, Datadog).
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include structured extra fields if provided
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno",
                "pathname", "filename", "module", "exc_info",
                "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated", "thread",
                "threadName", "processName", "process", "message",
            }:
                log_data[key] = value

        # Attach exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure application-wide structured logging."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.handlers = [handler]

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


# Usage
logger = logging.getLogger(__name__)


def log_llm_request(
    user_id: str,
    thread_id: str,
    input_tokens: int,
    model: str,
) -> None:
    """Log an LLM request with structured fields."""
    logger.info(
        "LLM request initiated",
        extra={
            "user_id": user_id,
            "thread_id": thread_id,
            "input_tokens": input_tokens,
            "model": model,
        },
    )
```

### 예제 2: LangSmith 운영 트레이싱 설정

```python
# app/tracing.py
import os
import time
import functools
from typing import Any, Callable, TypeVar
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

# LangSmith is automatically activated when LANGSMITH_API_KEY is set.
# No additional code is required for basic tracing.
#
# Environment variables:
#   LANGSMITH_API_KEY   — your LangSmith API key
#   LANGSMITH_PROJECT   — project name (use different names per environment)
#   LANGCHAIN_TRACING_V2=true

# Example .env for production:
# LANGSMITH_API_KEY=ls__xxxxx
# LANGSMITH_PROJECT=my-app-production
# LANGCHAIN_TRACING_V2=true


import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable  # type: ignore[import]
from pydantic import SecretStr

# NOTE: langsmith 패키지가 필요합니다: pip install langsmith


@traceable(name="chat_with_context", run_type="chain")
async def chat_with_metadata(
    message: str,
    user_id: str,
    session_id: str,
) -> str:
    """
    Invoke LLM with metadata tags for LangSmith filtering.

    Tags and metadata allow filtering runs by user, feature, or environment
    in the LangSmith dashboard.
    """
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        # LangSmith run metadata (visible in dashboard)
        tags=["production", "v1.2"],
        metadata={
            "user_id": user_id,
            "session_id": session_id,
            "feature": "main_chat",
        },
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer in Korean."),
        ("human", "{message}"),
    ])

    chain = prompt | llm
    response = await chain.ainvoke({"message": message})
    return response.content
```

### 예제 3: 피드백 수집 API

```python
# app/feedback.py
import os
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, SecretStr
from fastapi import APIRouter, HTTPException
from langsmith import Client as LangSmithClient  # type: ignore[import]

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackScore(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class FeedbackRequest(BaseModel):
    """User feedback on a specific LLM run."""
    run_id: str = Field(..., description="LangSmith run ID from trace")
    score: FeedbackScore
    comment: str = Field(default="", max_length=1000)


@router.post("/submit")
async def submit_feedback(feedback: FeedbackRequest) -> dict[str, str]:
    """
    Submit user feedback to LangSmith.

    The run_id links feedback to the specific LangSmith trace,
    enabling dataset curation and regression testing.
    """
    score_map = {
        FeedbackScore.POSITIVE: 1,
        FeedbackScore.NEGATIVE: 0,
        FeedbackScore.NEUTRAL: 0.5,
    }

    try:
        client = LangSmithClient(api_key=SecretStr(os.environ["LANGSMITH_API_KEY"]))
        client.create_feedback(
            run_id=feedback.run_id,
            key="user_rating",
            score=score_map[feedback.score],
            comment=feedback.comment or None,
        )
        return {"status": "ok", "run_id": feedback.run_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {e}")
```

### 예제 4: 지표 추적 — 지연·비용·오류율

```python
# app/metrics.py
import os
import time
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.outputs import LLMResult
from langchain_core.callbacks import AsyncCallbackHandler
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class MetricsCallbackHandler(AsyncCallbackHandler):
    """
    Async callback handler that captures LLM performance metrics.

    Attaches to LangChain chains to collect latency, token usage,
    and error events without modifying business logic.
    """

    def __init__(self, user_id: str) -> None:
        super().__init__()
        self.user_id = user_id
        self._start_time: float = 0.0

    async def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
        self._start_time = time.monotonic()
        logger.info("LLM call started", extra={"user_id": self.user_id})

    async def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        latency_ms = (time.monotonic() - self._start_time) * 1000
        usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}

        logger.info(
            "LLM call completed",
            extra={
                "user_id": self.user_id,
                "latency_ms": round(latency_ms, 1),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        )

    async def on_llm_error(self, error: Exception, **kwargs) -> None:
        latency_ms = (time.monotonic() - self._start_time) * 1000
        logger.error(
            "LLM call failed",
            extra={
                "user_id": self.user_id,
                "latency_ms": round(latency_ms, 1),
                "error_type": type(error).__name__,
                "error_msg": str(error),
            },
        )


async def invoke_with_metrics(message: str, user_id: str) -> str:
    """Invoke LLM chain with automatic metrics capture."""
    callback = MetricsCallbackHandler(user_id=user_id)

    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        callbacks=[callback],
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question concisely in Korean."),
        ("human", "{message}"),
    ])

    chain = prompt | llm
    response = await chain.ainvoke({"message": message})
    return response.content
```

### 예제 5: OpenTelemetry 연동 개념 코드

```python
# app/otel_setup.py
"""
OpenTelemetry integration for LangChain.

NOTE: 실제 OTel 백엔드 설정은 인프라에 따라 다릅니다.
      이 예제는 개념 설명용입니다.

Required packages:
    pip install opentelemetry-sdk opentelemetry-exporter-otlp
    pip install opentelemetry-instrumentation-fastapi

OTel 공식 문서: https://opentelemetry.io/docs/languages/python/
"""
import os
from typing import Optional


def setup_opentelemetry(
    service_name: str = "langchain-app",
    otlp_endpoint: Optional[str] = None,
) -> None:
    """
    Configure OpenTelemetry SDK with OTLP exporter.

    Traces are exported to the configured backend (Jaeger, Grafana Tempo, etc.)
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        print(f"[OTel] Tracing enabled → {endpoint}")

    except ImportError:
        print("[OTel] opentelemetry-sdk not installed. Skipping OTel setup.")


# LangChain과 OTel 통합:
# LangChain은 LANGCHAIN_TRACING_V2=true 설정 시 자동으로 OTel 스팬을 생성합니다.
# 이를 LangSmith 대신 자체 OTel 백엔드로 보내려면:
#
# export LANGCHAIN_TRACING_V2=true
# export LANGCHAIN_ENDPOINT=http://your-otel-collector:4318
#
# 또는 python-langchain-opentelemetry 패키지를 사용하세요.
```

### 예제 6: 프로덕션 디버깅 헬퍼

```python
# app/debug_utils.py
"""
Production debugging utilities.

LangSmith run ID를 응답 헤더에 포함하면,
프로덕션 사용자 리포트를 받았을 때 바로 해당 트레이스를 찾을 수 있습니다.
"""
import os
import logging
from contextvars import ContextVar
from uuid import uuid4

from fastapi import Request, Response
from langsmith import Client as LangSmithClient  # type: ignore[import]

logger = logging.getLogger(__name__)

# Context variable to carry run_id through async request lifecycle
_current_run_id: ContextVar[str] = ContextVar("current_run_id", default="")


def get_current_run_id() -> str:
    return _current_run_id.get()


def set_current_run_id(run_id: str) -> None:
    _current_run_id.set(run_id)


async def add_trace_id_header(request: Request, call_next) -> Response:
    """
    FastAPI middleware that adds X-Trace-ID to every response.

    Clients can include this ID in bug reports to pinpoint the exact LangSmith trace.
    """
    trace_id = str(uuid4())
    set_current_run_id(trace_id)

    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    return response


def get_langsmith_run_url(run_id: str) -> str:
    """Build a direct URL to a LangSmith run for internal debugging."""
    project = os.getenv("LANGSMITH_PROJECT", "default")
    return f"https://smith.langchain.com/projects/{project}/r/{run_id}"


# Example: logging run URL for internal debugging
# logger.info("LLM run completed", extra={"langsmith_url": get_langsmith_run_url(run_id)})
```

---

## ✏️ 실습 과제

**과제 1 — 환경별 LangSmith 프로젝트 분리**

`.env.dev`, `.env.prod` 파일에 각각 `LANGSMITH_PROJECT=my-app-dev`, `LANGSMITH_PROJECT=my-app-prod`를 설정하고, 두 환경에서 같은 코드를 실행했을 때 LangSmith 대시보드에서 트레이스가 분리되는지 확인하세요.

**과제 2 — 비용 추적 대시보드 설계**

`MetricsCallbackHandler`에서 수집한 `total_tokens`와 모델별 가격을 곱하여 요청당 예상 비용을 계산하고, 일일 누적 비용을 집계하는 인메모리 카운터를 구현하세요.

```python
from collections import defaultdict
from datetime import date

# date → total_cost (USD)
daily_cost_tracker: dict[date, float] = defaultdict(float)

# gpt-4o-mini pricing (변경될 수 있음, 항상 공식 가격 확인)
COST_PER_1K_INPUT_TOKENS = 0.00015
COST_PER_1K_OUTPUT_TOKENS = 0.00060

def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate estimated cost in USD."""
    # TODO: 구현
    pass
```

**과제 3 — 슬로우 쿼리 알림**

지연 시간이 5초를 초과하면 `logger.warning()`으로 경고를 기록하는 로직을 `MetricsCallbackHandler.on_llm_end()`에 추가하세요.

---

## ⚠️ 흔한 함정

**1. 개발 환경과 프로덕션 환경을 같은 LangSmith 프로젝트로 사용**

트레이스가 섞이면 의미 있는 지표를 뽑을 수 없습니다. `LANGSMITH_PROJECT` 환경 변수로 반드시 분리하세요.

**2. 동기 콜백을 비동기 체인에 사용**

`BaseCallbackHandler`의 동기 메서드(`on_llm_end`)를 `async` 체인에 붙이면 이벤트 루프가 블로킹됩니다. 반드시 `AsyncCallbackHandler`를 상속하세요.

**3. 민감 정보를 트레이스에 그대로 남기는 실수**

LangSmith 트레이스에는 입력 전체가 저장됩니다. PII가 포함된 입력은 Phase 41에서 배운 마스킹을 먼저 적용하세요.

**4. 로그를 `print()`로 출력하는 습관**

`print()`는 타임스탬프·레벨·구조화 필드가 없어 프로덕션 디버깅에 쓸모가 없습니다. 반드시 `logging` 모듈을 사용하세요.

**5. 피드백 없이 모델 성능을 판단**

지연·비용 지표만으로는 모델 품질을 알 수 없습니다. 사용자 피드백(👍/👎)과 LLM-as-Judge 자동 평가를 함께 수집하세요.

---

## ✅ 셀프 체크

- [ ] `JSONFormatter`로 구조화 JSON 로그가 출력된다.
- [ ] `LANGSMITH_API_KEY`와 `LANGCHAIN_TRACING_V2=true` 설정 후 트레이스가 LangSmith에 나타난다.
- [ ] `MetricsCallbackHandler`가 지연(ms)·토큰 수를 로그에 기록한다.
- [ ] `/feedback/submit` 엔드포인트가 LangSmith에 피드백을 전송한다.
- [ ] `X-Trace-ID` 헤더가 응답에 포함된다.
- [ ] 개발·프로덕션 LangSmith 프로젝트를 분리했다.
- [ ] OpenTelemetry의 역할과 LangSmith와의 차이를 설명할 수 있다.

---

## 🔗 참고 자료

- [LangSmith 공식 문서](https://docs.smith.langchain.com)
- [LangSmith 피드백 API](https://docs.smith.langchain.com/reference/data_formats/feedback)
- [LangChain 콜백 핸들러](https://python.langchain.com/docs/concepts/callbacks)
- [OpenTelemetry Python SDK](https://opentelemetry.io/docs/languages/python/)
- [Python logging HOWTO](https://docs.python.org/3/howto/logging.html)

---

## 🔗 네비게이션

| 이전 | 현재 | 다음 |
|------|------|------|
| [Phase 43: LangGraph Platform](43-deploy-langgraph-platform.md) | **Phase 44: 관측성·모니터링** | [Phase 45: 캡스톤 프로젝트](../99-capstone/45-capstone-project.md) |
