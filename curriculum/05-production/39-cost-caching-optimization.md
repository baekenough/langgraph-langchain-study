# Phase 39: 비용·캐싱 최적화

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 90분 |
| 난이도 | ★★★☆☆ |
| 선행 학습 | Phase 04 (Chat Models), Phase 08 (스트리밍) |

---

## 🎯 학습 목표

- `usage_metadata`와 OpenRouter 응답 메타데이터로 토큰 사용량을 측정합니다.
- 모델 크기·비용 트레이드오프를 이해하고 상황에 맞는 모델을 선택합니다.
- `InMemoryCache`와 `SQLiteCache`로 LLM 응답을 캐싱해 중복 비용을 줄입니다.
- 시맨틱 캐시의 개념과 적합한 사용 사례를 설명합니다.
- provider prefix 캐시(cache_control)의 원리와 멀티턴 대화에서의 활용법을 이해합니다.
- 배치 처리와 스트리밍으로 체감 성능을 개선합니다.
- 시스템 프롬프트 최적화로 토큰을 절약하는 방법을 적용합니다.

---

## 📚 핵심 개념

### 1. 토큰과 비용의 관계

```
총 비용 = (입력 토큰 × 입력 단가) + (출력 토큰 × 출력 단가)

GPT-4o-mini (OpenRouter 기준, 2024년 기준 참고용):
  입력: ~$0.15 / 1M 토큰
  출력: ~$0.60 / 1M 토큰

※ 실제 비용은 변동되므로 반드시 https://openrouter.ai/models 에서 확인하세요.
```

### 2. OpenRouter에서 토큰 측정

OpenRouter를 통해 호출하면 OpenAI 직접 호출의 `get_openai_callback()`이 정확하게 동작하지 않습니다.
대신 **응답 객체의 메타데이터**에서 토큰 정보를 읽어야 합니다:

```python
response = llm.invoke(messages)

# 방법 1: usage_metadata (LangChain v0.2+ 통합 필드)
tokens = response.usage_metadata
# {"input_tokens": 45, "output_tokens": 123, "total_tokens": 168}

# 방법 2: response_metadata (원본 API 응답)
meta = response.response_metadata
# {"token_usage": {"prompt_tokens": 45, "completion_tokens": 123, ...}}
```

### 3. 캐시 패밀리 구분: 두 가지 다른 메커니즘

LLM 비용 최적화에서 "캐시"는 두 가지 전혀 다른 메커니즘을 가리킵니다:

| 구분 | provider prefix 캐시 (cache_control) | LangChain 응답 캐시 (InMemory/SQLite) |
|------|--------------------------------------|---------------------------------------|
| 매칭 단위 | 입력 앞부분(prefix)만 동일하면 부분 재사용 | 입력 전체가 byte 동일해야 함 |
| 저장 위치 | provider 서버 | 클라이언트 (프로세스/파일) |
| 효과 | prefix 토큰만 저렴 (API는 여전히 호출) | 응답 통째 재사용 (API 호출 0) |
| 멀티턴 효과 | 있음 (이력 prefix 재사용) | 없음 (매 턴 이력이 달라 전체 일치 불가) |

두 캐시를 함께 사용할 수 있습니다: prefix 캐시로 입력 토큰 비용을 줄이면서, 완전히 동일한 단건 요청에는 응답 캐시로 API 호출 자체를 생략합니다.

### 4. provider prefix 캐시 (cache_control)

#### 작동 원리

```
일반 멀티턴 (매 턴 전체 재전송):
  턴1: [시스템(500토큰)] + [질문1]              → 전부 과금
  턴2: [시스템(500토큰)] + [질문1+답1] + [질문2] → 전부 과금
  턴3: [시스템(500토큰)] + [이전이력] + [질문3]  → 전부 과금

prefix 캐시 (고정 블록 재사용):
  턴1: [시스템★캐시저장] + [질문1]              → 시스템 처음 과금
  턴2: [시스템★캐시히트] + [질문1+답1] + [질문2] → 시스템 할인
  턴3: [시스템★캐시히트] + [이전이력] + [질문3]  → 시스템 할인
```

#### 핵심 조건

- **최소 토큰**: 캐시 블록은 약 1,024 토큰 이상이어야 히트합니다 (provider마다 임계값 상이).
- **TTL**: 캐시는 약 5분간 유지됩니다. 연속 대화가 아니면 히트율이 낮습니다.
- **순서 고정**: 캐시 히트는 prefix가 변하지 않아야 합니다. 동적 값(현재 시각, 요청 ID 등)을 시스템 프롬프트 **앞**에 삽입하면 매 턴 prefix가 달라져 캐시 미스가 발생합니다.

#### provider별 동작

| provider (OpenRouter 경유) | cache_control 필요 여부 | 비고 |
|---------------------------|------------------------|------|
| `openai/...` 모델 | **불필요** — 자동 적용 | prefix 충분히 크고 고정되면 자동 할인 |
| `anthropic/...` 모델 | **수동 필수** — `cache_control` 블록 명시 | 누락 시 캐시 0 |

> **API 변동 주의**: cache_control 동작은 LangChain·OpenRouter·provider API 버전에 따라 달라질 수 있습니다. 실제 프로덕션 적용 전에 [OpenRouter Prompt Caching](https://openrouter.ai/docs/prompt-caching) 및 [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) 공식 문서를 반드시 확인하세요.

### 5. LangChain 응답 캐시 원리 (완전 일치)

```
캐싱 없이:               캐싱 사용 시:
요청 1 → API 호출        요청 1 → API 호출 → 캐시 저장
요청 2 → API 호출        요청 2 (동일) → 캐시 히트! (0원, 즉시 반환)
요청 3 → API 호출        요청 3 (동일) → 캐시 히트!
```

**언제 캐싱이 유효한가:**
- FAQ 챗봇: 동일한 질문이 반복됨
- 문서 분류: 같은 카테고리 설명 반복 사용
- 템플릿 기반 요청: 입력 변수만 다르고 프롬프트 구조 동일

**언제 캐싱이 부적합한가 (LangChain 응답 캐시 기준):**
- temperature > 0 (매번 다른 응답 원할 때) — provider prefix 캐시는 temperature와 무관
- 날짜/시간 의존적 질문
- 사용자별 맞춤 응답
- 멀티턴 대화 이력 (매 턴 입력이 달라 전체 일치 불가)

### 6. 캐시 유형 비교

| 캐시 유형 | 저장 위치 | 매칭 방식 | 멀티턴 효과 | 적합한 용도 |
|----------|---------|----------|-----------|------------|
| provider prefix 캐시 | provider 서버 | prefix 부분 일치 | 있음 | 긴 시스템 프롬프트·문서 고정 |
| `InMemoryCache` | 프로세스 내 | 완전 일치 | 없음 | 개발/테스트 |
| `SQLiteCache` | 파일 (영구) | 완전 일치 | 없음 | 단일 서버 프로덕션 |
| 시맨틱 캐시 | 벡터 DB | 의미 유사도 | 없음 | "비슷한" 질문 캐싱 |

---

## 💻 코드 예제

### 예제 1: 토큰 사용량 측정

```python
# measure_tokens.py
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


def measure_call(prompt: str, model_name: str = "openai/gpt-4o-mini") -> dict:
    """LLM 호출의 토큰 사용량과 지연시간을 측정합니다."""
    start = time.perf_counter()

    response = llm.invoke([
        SystemMessage(content="당신은 도움이 되는 어시스턴트입니다."),
        HumanMessage(content=prompt),
    ])

    elapsed = time.perf_counter() - start

    # usage_metadata로 토큰 확인 (LangChain 통합 필드)
    usage = response.usage_metadata or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    # response_metadata에서 원본 API 응답 확인
    raw_meta = response.response_metadata or {}
    token_usage_raw = raw_meta.get("token_usage", {})

    return {
        "input_tokens": input_tokens or token_usage_raw.get("prompt_tokens", 0),
        "output_tokens": output_tokens or token_usage_raw.get("completion_tokens", 0),
        "total_tokens": (input_tokens + output_tokens) or token_usage_raw.get("total_tokens", 0),
        "latency_sec": round(elapsed, 3),
        "response_preview": response.content[:100],
    }


# 측정 비교: 짧은 프롬프트 vs 긴 프롬프트
short_prompt = "파이썬이란?"
long_prompt = """
파이썬 언어의 역사, 설계 철학, 주요 특징, 다른 언어와의 비교,
주요 사용 사례, 유명한 라이브러리, 미래 전망에 대해
각 항목당 3문장 이상으로 자세히 설명해 주세요.
"""

print("=== 토큰 사용량 비교 ===\n")

short_result = measure_call(short_prompt)
print(f"짧은 프롬프트:")
print(f"  입력 토큰: {short_result['input_tokens']}")
print(f"  출력 토큰: {short_result['output_tokens']}")
print(f"  지연시간: {short_result['latency_sec']}초\n")

long_result = measure_call(long_prompt)
print(f"긴 프롬프트:")
print(f"  입력 토큰: {long_result['input_tokens']}")
print(f"  출력 토큰: {long_result['output_tokens']}")
print(f"  지연시간: {long_result['latency_sec']}초")

ratio = long_result['total_tokens'] / max(short_result['total_tokens'], 1)
print(f"\n토큰 배율: {ratio:.1f}x")
```

### 예제 2: provider prefix 캐시 (cache_control)

```python
# prefix_cache.py
"""
provider prefix 캐시(cache_control) 예제

- openai/* 모델: 자동 prefix 캐시 — 별도 코드 불필요
- anthropic/* 모델 (OpenRouter 경유): cache_control 블록 수동 명시 필요

주의: cache_control 동작은 LangChain·OpenRouter·provider 버전에 따라 달라질 수 있습니다.
      공식 문서를 반드시 확인하세요:
      - https://openrouter.ai/docs/prompt-caching
      - https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
"""
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# 긴 고정 시스템 프롬프트 (캐시 효과를 위해 약 1,024 토큰 이상 필요)
# 실제 운영에서는 페르소나 설명, 지식 문서, few-shot 예시 등을 배치합니다.
LONG_SYSTEM_TEXT = (
    "당신은 파이썬 전문 튜터입니다.\n\n"
    "== 파이썬 핵심 개념 참조 문서 ==\n"
    "1. 리스트(list): 순서 있는 가변 시퀀스.\n"
    "2. 튜플(tuple): 순서 있는 불변 시퀀스. 해시 가능.\n"
    "3. 딕셔너리(dict): 키-값 쌍. O(1) 조회.\n"
    "4. 컴프리헨션: [x for x in seq if cond] 형태로 간결하게 생성.\n"
    "5. 제너레이터: yield로 지연 평가. 메모리 효율적.\n"
    "6. 데코레이터: @func 문법으로 함수를 감싸 기능 추가.\n"
    "7. GIL: CPython 인터프리터 수준 잠금. CPU 바운드 멀티스레딩 제한.\n"
    "8. asyncio: async/await로 I/O 바운드 작업 논블로킹 처리.\n"
) * 8  # 반복으로 토큰 확보 (실제 운영에서는 실질적인 내용으로 채우세요)


# ─── 시나리오 A: openai/* 모델 — 자동 prefix 캐시 ─────────────────────────
# 별도 코드 추가 없이 시스템 프롬프트를 고정하고 충분히 크게 유지하면 됩니다.

openai_llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

print("=== 시나리오 A: OpenAI 모델 (자동 prefix 캐시) ===\n")

# 멀티턴: 시스템 프롬프트 고정, 이력만 순차적으로 추가
messages_a: list = [
    SystemMessage(content=LONG_SYSTEM_TEXT),  # 고정 블록 — 절대 변경하지 말 것
    HumanMessage(content="리스트 컴프리헨션이란 무엇인가요?"),
]

response_a1 = openai_llm.invoke(messages_a)
print(f"턴1 usage_metadata: {response_a1.usage_metadata}")

# 다음 턴: 이력 추가, 시스템 메시지는 그대로 유지
messages_a.append(response_a1)
messages_a.append(HumanMessage(content="딕셔너리 컴프리헨션도 같은 방식인가요?"))

response_a2 = openai_llm.invoke(messages_a)
print(f"턴2 usage_metadata: {response_a2.usage_metadata}")
# cache_read_input_tokens > 0 이면 prefix 캐시 적중 (필드명은 버전·provider마다 다를 수 있음)


# ─── 시나리오 B: anthropic/* 모델 (OpenRouter 경유) — 수동 cache_control 필수 ───
# cache_control 블록 없이는 prefix 캐시가 전혀 동작하지 않습니다.

anthropic_llm = ChatOpenAI(
    model="anthropic/claude-3.5-haiku",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# SystemMessage content를 블록 리스트로 구성: 캐시할 블록 마지막에 cache_control 삽입
cached_system = SystemMessage(content=[
    {
        "type": "text",
        "text": LONG_SYSTEM_TEXT,
        "cache_control": {"type": "ephemeral"},  # 이 블록까지를 prefix로 캐싱
    }
])

print("\n=== 시나리오 B: Anthropic 모델 (OpenRouter 경유, 수동 cache_control) ===\n")

messages_b: list = [
    cached_system,  # 매 턴 동일한 객체 재사용 — 변경하면 캐시 미스 발생
    HumanMessage(content="리스트 컴프리헨션이란 무엇인가요?"),
]

response_b1 = anthropic_llm.invoke(messages_b)
print(f"턴1 usage_metadata: {response_b1.usage_metadata}")
# cache_creation_input_tokens > 0: 이번 턴에 캐시가 새로 생성됨

messages_b.append(response_b1)
messages_b.append(HumanMessage(content="딕셔너리 컴프리헨션도 같은 방식인가요?"))

response_b2 = anthropic_llm.invoke(messages_b)
print(f"턴2 usage_metadata: {response_b2.usage_metadata}")
# cache_read_input_tokens > 0: 이전에 저장된 캐시에서 읽음 (할인 또는 무료)

# 참고: 대화 이력 전체를 캐싱하려면 cache_control breakpoint를 마지막 메시지로 이동합니다.
# 이 패턴은 이력이 길어질수록 효과가 커집니다.
print("\n※ 필드명은 LangChain·OpenRouter·provider 버전에 따라 다를 수 있습니다.")
print("  response.usage_metadata와 response.response_metadata를 직접 출력해 확인하세요.")
```

### 예제 3: InMemoryCache로 중복 호출 방지

```python
# inmemory_cache.py
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.caches import InMemoryCache
from langchain.globals import set_llm_cache
from pydantic import SecretStr

load_dotenv()

# 전역 캐시 설정 (애플리케이션 시작 시 한 번만)
set_llm_cache(InMemoryCache())

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,  # 캐싱은 temperature=0일 때만 의미 있음
)


def timed_invoke(prompt: str) -> tuple[str, float]:
    """호출 시간을 측정하며 LLM을 호출합니다."""
    start = time.perf_counter()
    result = llm.invoke(prompt)
    elapsed = time.perf_counter() - start
    return result.content, round(elapsed, 3)


print("=== InMemoryCache 효과 측정 ===\n")

question = "파이썬에서 리스트 컴프리헨션이란 무엇인가요?"

# 첫 번째 호출: API 요청 발생
print("첫 번째 호출 (캐시 미스):")
response1, time1 = timed_invoke(question)
print(f"  응답: {response1[:80]}...")
print(f"  소요시간: {time1}초\n")

# 두 번째 호출: 동일한 입력 → 캐시 히트
print("두 번째 호출 (캐시 히트):")
response2, time2 = timed_invoke(question)
print(f"  응답: {response2[:80]}...")
print(f"  소요시간: {time2}초\n")

# 다른 질문: 캐시 미스
print("다른 질문 (캐시 미스):")
other_q = "파이썬에서 제너레이터란?"
response3, time3 = timed_invoke(other_q)
print(f"  소요시간: {time3}초\n")

print(f"캐시 효과: {time1/max(time2, 0.001):.0f}x 빠름 (예상: 수십~수백배)")
print("주의: InMemoryCache는 프로세스 재시작 시 초기화됩니다.")
```

### 예제 4: SQLiteCache로 영구 캐싱

```python
# sqlite_cache.py
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
from pydantic import SecretStr

load_dotenv()

# SQLite 파일로 영구 캐싱 설정
# 파일은 프로세스 재시작 후에도 유지됩니다
cache_path = ".langchain_cache.db"
set_llm_cache(SQLiteCache(database_path=cache_path))

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


def timed_invoke(prompt: str) -> tuple[str, float]:
    start = time.perf_counter()
    result = llm.invoke(prompt)
    elapsed = time.perf_counter() - start
    return result.content, round(elapsed, 3)


question = "파이썬의 GIL이란 무엇인가요?"

print("=== SQLiteCache 효과 ===\n")
print(f"캐시 파일: {cache_path}")
print("(첫 실행 후 스크립트를 재시작해도 캐시가 유지됩니다)\n")

response, elapsed = timed_invoke(question)
print(f"소요시간: {elapsed}초")
print(f"응답 미리보기: {response[:100]}...")

# 캐시 파일 크기 확인
if os.path.exists(cache_path):
    size_kb = os.path.getsize(cache_path) / 1024
    print(f"\n캐시 파일 크기: {size_kb:.1f} KB")
```

### 예제 5: 모델 선택 전략 (비용 최적화)

```python
# model_selection.py
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

load_dotenv()


def make_llm(model: str) -> ChatOpenAI:
    """모델을 지정해 ChatOpenAI 인스턴스를 생성합니다."""
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )


# 작업 유형에 따른 모델 선택 전략
class AdaptiveLLMSelector:
    """작업 유형에 따라 적절한 모델을 선택합니다."""

    def __init__(self):
        # 저렴한 모델: 단순 분류, 포맷팅, 추출
        self.cheap_llm = make_llm("openai/gpt-4o-mini")
        # 고성능 모델: 복잡한 추론, 창작, 코드 생성
        self.powerful_llm = make_llm("openai/gpt-4o")

    def classify_task(self, task: str) -> str:
        """작업의 복잡도를 분류합니다 (simple/complex)."""
        # 실제 구현에서는 규칙 기반 또는 cheap_llm으로 분류
        simple_keywords = ["분류", "추출", "요약", "번역", "포맷"]
        complex_keywords = ["분석", "설계", "작성", "최적화", "추론"]

        if any(kw in task for kw in complex_keywords):
            return "complex"
        return "simple"

    def execute(self, task: str, content: str) -> str:
        """작업 복잡도에 따라 적절한 모델로 실행합니다."""
        complexity = self.classify_task(task)
        llm = self.cheap_llm if complexity == "simple" else self.powerful_llm

        print(f"작업: '{task}' → 복잡도: {complexity} → 모델: {llm.model_name}")

        chain = (
            ChatPromptTemplate.from_messages([
                ("system", f"다음 작업을 수행하세요: {task}"),
                ("human", "{content}"),
            ])
            | llm
            | StrOutputParser()
        )

        return chain.invoke({"content": content})


selector = AdaptiveLLMSelector()

# 단순 작업: 저렴한 모델로 처리
result1 = selector.execute(
    task="텍스트에서 날짜를 추출하세요",
    content="회의는 2024년 3월 15일 오후 2시에 예정되어 있습니다.",
)
print(f"결과: {result1}\n")

# 복잡한 작업: 고성능 모델로 처리
result2 = selector.execute(
    task="코드를 분석하고 잠재적 버그를 찾으세요",
    content="def divide(a, b): return a / b",
)
print(f"결과: {result2}\n")
```

### 예제 6: 배치와 스트리밍으로 체감 성능 개선

```python
# batch_streaming.py
import os
import time
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "간결하게 답변하세요."),
        ("human", "{question}"),
    ])
    | llm
    | StrOutputParser()
)


# 방법 1: 순차 처리 (느림)
def sequential_process(questions: list[str]) -> list[str]:
    """여러 질문을 순차적으로 처리합니다."""
    return [chain.invoke({"question": q}) for q in questions]


# 방법 2: 배치 처리 (빠름)
def batch_process(questions: list[str]) -> list[str]:
    """여러 질문을 병렬 배치로 처리합니다."""
    inputs = [{"question": q} for q in questions]
    return chain.batch(inputs, config={"max_concurrency": 3})


# 방법 3: 스트리밍 (체감 빠름)
def streaming_process(question: str) -> None:
    """스트리밍으로 토큰을 실시간 출력합니다."""
    print("스트리밍 응답: ", end="", flush=True)
    for chunk in chain.stream({"question": question}):
        print(chunk, end="", flush=True)
    print()  # 줄바꿈


questions = [
    "파이썬의 장점은?",
    "딕셔너리 컴프리헨션이란?",
    "with 문의 역할은?",
]

print("=== 처리 방식 비교 ===\n")

# 순차 처리 시간 측정
start = time.perf_counter()
sequential_results = sequential_process(questions)
sequential_time = time.perf_counter() - start
print(f"순차 처리: {sequential_time:.2f}초")

# 배치 처리 시간 측정
start = time.perf_counter()
batch_results = batch_process(questions)
batch_time = time.perf_counter() - start
print(f"배치 처리: {batch_time:.2f}초")

print(f"\n속도 개선: {sequential_time/max(batch_time, 0.001):.1f}x\n")

# 스트리밍 예시
print("=== 스트리밍 예시 ===")
streaming_process("파이썬 제너레이터의 장점은?")
```

### 예제 7: 프롬프트 최적화로 토큰 절약

```python
# prompt_optimization.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


def count_tokens(messages: list) -> int:
    """메시지의 토큰 수를 측정합니다 (tiktoken 없이 근사값)."""
    total_chars = sum(len(m.content) for m in messages)
    return total_chars // 4  # 영어 기준 4자 ≈ 1토큰 (한국어는 더 많음)


def compare_prompts(
    prompt_a: str,
    prompt_b: str,
    question: str,
    label_a: str = "기존",
    label_b: str = "최적화",
) -> None:
    """두 가지 프롬프트 전략을 비교합니다."""
    messages_a = [
        SystemMessage(content=prompt_a),
        HumanMessage(content=question),
    ]
    messages_b = [
        SystemMessage(content=prompt_b),
        HumanMessage(content=question),
    ]

    # 실제 토큰은 usage_metadata에서 확인
    response_a = llm.invoke(messages_a)
    response_b = llm.invoke(messages_b)

    tokens_a = response_a.usage_metadata or {}
    tokens_b = response_b.usage_metadata or {}

    print(f"=== {label_a} vs {label_b} ===")
    print(f"[{label_a}] 입력 토큰: {tokens_a.get('input_tokens', '?')}")
    print(f"[{label_b}] 입력 토큰: {tokens_b.get('input_tokens', '?')}")
    print()


# 비교 1: 장황한 시스템 프롬프트 vs 간결한 시스템 프롬프트
verbose_system = """
당신은 파이썬 프로그래밍 언어 전문가입니다. 사용자가 파이썬에 관련된 질문을 할 때,
친절하고 전문적인 방식으로 답변해 주세요. 답변은 명확하고 이해하기 쉽도록 작성하며,
필요한 경우 코드 예시를 포함할 수 있습니다. 사용자의 수준에 맞게 설명을 조절하고,
전문 용어를 사용할 때는 간단한 설명을 덧붙이세요. 언제나 정확하고 최신 정보를
기반으로 답변하며, 확실하지 않은 내용은 그렇다고 명시하세요.
"""

concise_system = "파이썬 전문가. 명확하고 실용적으로 답변하라."

compare_prompts(
    verbose_system,
    concise_system,
    "리스트 컴프리헨션 예시를 보여주세요.",
    "장황한 프롬프트",
    "간결한 프롬프트",
)

# 실용적인 프롬프트 최적화 팁 출력
print("=== 프롬프트 토큰 절약 팁 ===")
tips = [
    "1. 시스템 프롬프트를 간결하게 유지하세요 (50토큰 이하 목표)",
    "2. 반복되는 예시는 few-shot 대신 retrieval로 대체하세요",
    "3. 불필요한 정중한 표현('항상', '언제나', '반드시')을 제거하세요",
    "4. 대화 이력은 최근 N개만 유지하세요 (슬라이딩 윈도우)",
    "5. 긴 문서는 RAG로 청크 단위로 제공하세요",
]
for tip in tips:
    print(f"  {tip}")
```

---

## ✏️ 실습 과제

### 과제 1: 토큰 측정 대시보드 (필수)

5개의 서로 다른 질문에 대해 `usage_metadata`로 토큰을 측정하고, 입력/출력 토큰 비율과 예상 비용을 계산하는 스크립트를 작성합니다. (OpenRouter 요금 페이지 참조)

### 과제 2: 캐시 효과 측정 (중급)

같은 질문 10개를 캐시 없이, `InMemoryCache`로, `SQLiteCache`로 각각 실행해 총 소요 시간을 비교합니다. 캐시 히트율이 비용에 미치는 영향을 정량화하세요.

### 과제 3: 적응형 모델 라우터 (심화)

입력 텍스트의 복잡도를 자동으로 판단해 gpt-4o-mini(단순)와 gpt-4o(복잡) 중 하나를 선택하는 라우터를 구현합니다. Phase 20(노드/엣지/라우팅)에서 배운 패턴을 활용하세요.

---

## ⚠️ 흔한 함정

### 1. temperature > 0에서 LangChain 응답 캐싱 오류

> **적용 범위**: 이 함정은 `InMemoryCache`/`SQLiteCache` 등 **LangChain 응답 캐시**에만 해당합니다.
> provider prefix 캐시(cache_control)는 temperature와 무관하게 동작합니다.

```python
from pydantic import SecretStr

# temperature > 0이면 LangChain 응답 캐시가 동작하지 않습니다
# LangChain은 temperature=0인 경우에만 응답 캐싱을 적용합니다
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,  # 응답 캐싱을 위해 0으로 설정
)
```

### 2. OpenRouter에서 get_openai_callback 오작동

```python
# 이 방법은 OpenRouter에서 정확하지 않습니다
from langchain_community.callbacks import get_openai_callback
with get_openai_callback() as cb:
    llm.invoke("질문")
print(cb.total_tokens)  # 0 또는 부정확한 값

# OpenRouter에서는 이 방법을 사용하세요:
response = llm.invoke("질문")
print(response.usage_metadata)  # 정확한 값
```

### 3. SQLiteCache 동시 접근 문제

SQLiteCache는 단일 프로세스 환경에 적합합니다.
멀티프로세스나 분산 환경에서는 Redis 캐시를 고려하세요:

```python
# 고급: Redis 기반 캐시 (langchain_community 필요)
# from langchain_community.cache import RedisCache
# import redis
# set_llm_cache(RedisCache(redis_=redis.from_url("redis://localhost:6379")))
```

### 4. 배치 크기와 레이트 리밋

```python
# max_concurrency를 너무 높게 설정하면 레이트 리밋 오류 발생
# API 제공자의 RPM(분당 요청 수)을 확인하세요
chain.batch(inputs, config={"max_concurrency": 3})  # 보수적으로 시작
```

### 5. 멀티턴에서 prefix를 흔드는 안티패턴

prefix 캐시는 메시지 목록의 **앞부분이 동일**해야 히트합니다.
동적 값을 시스템 프롬프트 앞에 삽입하면 매 턴 prefix가 달라져 캐시가 전혀 적용되지 않습니다.

```python
import datetime

# 나쁜 예: 현재 시각을 시스템 프롬프트 앞에 삽입 → 매 턴 prefix가 달라짐
bad_system = SystemMessage(content=f"현재 시각: {datetime.datetime.now()}\n\n" + LONG_SYSTEM_TEXT)
# → 캐시 미스 100%

# 좋은 예: 고정 내용만 시스템 프롬프트에, 동적 값은 HumanMessage에 포함
good_system = SystemMessage(content=[
    {
        "type": "text",
        "text": LONG_SYSTEM_TEXT,
        "cache_control": {"type": "ephemeral"},
    }
])
messages = [
    good_system,  # 변경하지 않음
    HumanMessage(content=f"현재 시각 {datetime.datetime.now()} 기준으로 질문합니다: ..."),
]
```

### 6. OpenRouter 경유 anthropic 모델에서 cache_control 누락

`anthropic/*` 모델은 `openai/*` 모델과 달리 cache_control을 자동 적용하지 않습니다.
명시적으로 블록을 구성하지 않으면 캐시 히트율 0%입니다.

```python
# 나쁜 예: 일반 문자열 content — anthropic 모델에서 prefix 캐시 없음
bad_msg = SystemMessage(content=LONG_SYSTEM_TEXT)

# 좋은 예: 블록 리스트 + cache_control 명시
good_msg = SystemMessage(content=[
    {
        "type": "text",
        "text": LONG_SYSTEM_TEXT,
        "cache_control": {"type": "ephemeral"},
    }
])
# openai/* 모델에서는 good_msg 방식도 동작하지만, bad_msg 방식도 자동 캐시됩니다.
# anthropic/* 모델에서는 반드시 good_msg 방식을 사용해야 합니다.
```

> **API 변동 주의**: cache_control 지원 여부와 content 블록 구조는 LangChain·OpenRouter·provider 버전에 따라 다를 수 있습니다. 실제 동작은 [OpenRouter Prompt Caching 문서](https://openrouter.ai/docs/prompt-caching)를 확인하세요.

---

## ✅ 셀프 체크

- [ ] `usage_metadata`로 입력/출력 토큰을 측정했습니다.
- [ ] `response_metadata`에서 원본 API 응답의 토큰 정보를 확인했습니다.
- [ ] OpenRouter에서 `get_openai_callback()`이 정확하지 않은 이유를 설명합니다.
- [ ] `InMemoryCache`로 캐시 히트와 미스를 실험으로 확인했습니다.
- [ ] `SQLiteCache`로 프로세스 재시작 후에도 캐시가 유지됨을 확인했습니다.
- [ ] 작업 복잡도에 따른 모델 선택 전략을 구현했습니다.
- [ ] `chain.batch()`가 순차 처리보다 빠른 이유를 설명합니다.
- [ ] 시스템 프롬프트를 최적화해 입력 토큰을 줄이는 방법을 적용했습니다.
- [ ] provider prefix 캐시와 LangChain 응답 캐시의 차이(매칭 단위·저장 위치·멀티턴 효과)를 설명할 수 있습니다.
- [ ] anthropic/* 모델(OpenRouter 경유)에서 cache_control 블록을 SystemMessage에 삽입하는 방법을 구현했습니다.

---

## 🔗 참고 자료

- [LangChain 캐싱 가이드](https://python.langchain.com/docs/concepts/chat_models/#caching)
- [InMemoryCache API](https://python.langchain.com/api_reference/core/caches/langchain_core.caches.InMemoryCache.html)
- [SQLiteCache API](https://python.langchain.com/api_reference/community/cache/langchain_community.cache.SQLiteCache.html)
- [OpenRouter 모델/요금](https://openrouter.ai/models)
- [LangChain usage_metadata](https://python.langchain.com/docs/concepts/messages/#aimessage)
- [OpenRouter Prompt Caching](https://openrouter.ai/docs/prompt-caching)
- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)

> **API 변동 안내**: `usage_metadata` 필드는 LangChain v0.2에서 표준화되었습니다. 이전 버전에서는 `response_metadata`의 `token_usage`를 직접 사용해야 합니다. 또한 캐시 클래스의 임포트 경로(`langchain_core` vs `langchain_community`)가 버전에 따라 다를 수 있으니 [공식 문서](https://python.langchain.com/docs/concepts/chat_models/)를 확인하세요.

---

← [Phase 38: 테스트 전략](38-testing-strategies.md) | [Phase 40: 에러 처리·회복력](40-error-handling-resilience.md) →
