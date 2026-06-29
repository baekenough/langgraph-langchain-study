# Phase 40: 에러 처리·회복력

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 90분 |
| 난이도 | ★★★★☆ |
| 선행 학습 | Phase 07 (LCEL), Phase 18-22 (LangGraph 기초) |

---

## 🎯 학습 목표

- LLM 애플리케이션에서 발생하는 주요 예외 유형을 구분합니다.
- `.with_retry()`로 일시적 오류에 대한 지수 백오프 재시도를 구현합니다.
- `.with_fallbacks()`로 모델 실패 시 대체 체인으로 전환합니다.
- 타임아웃을 설정해 무한 대기를 방지합니다.
- `InMemoryRateLimiter`로 API 레이트 리밋을 사전에 제어합니다.
- LangGraph 그래프에서 노드 오류를 처리하는 패턴을 구현합니다.

---

## 📚 핵심 개념

### 1. LLM 앱의 주요 예외 유형

```
예외 유형                 원인                          대응 전략
─────────────────────────────────────────────────────────────────
RateLimitError           API 호출 한도 초과            재시도 + 레이트 리미터
APIConnectionError        네트워크 일시적 오류          재시도 (지수 백오프)
APITimeoutError           응답 지연 초과               타임아웃 설정 + 재시도
AuthenticationError       잘못된 API 키                즉시 실패 (재시도 불필요)
BadRequestError           잘못된 요청 파라미터          입력 검증, 즉시 실패
ContentFilterError        콘텐츠 정책 위반              폴백 또는 재작성
OutputParserException     LLM 출력 파싱 실패            폴백 파서 또는 재시도
```

### 2. with_retry() 동작 원리

```
with_retry(stop_after_attempt=3, wait_exponential_jitter=True)

시도 1 실패 → 대기 1초 + 랜덤 지터
시도 2 실패 → 대기 2초 + 랜덤 지터
시도 3 실패 → 예외 발생 (상위로 전파)
```

**지수 백오프 + 지터(Jitter)**: 여러 클라이언트가 동시에 재시도할 때 충돌을 피하기 위해 대기 시간에 무작위 노이즈를 추가합니다.

### 3. with_fallbacks() 동작 원리

```
기본 체인 실패
    ↓
폴백 체인 1 시도 → 성공 시 반환
    ↓ (실패 시)
폴백 체인 2 시도 → 성공 시 반환
    ↓ (모두 실패 시)
예외 발생
```

**언제 폴백이 유용한가:**
- 고성능 모델(GPT-4o) → 저렴한 모델(GPT-4o-mini) 폴백
- 외부 API → 로컬 캐시 폴백
- 복잡한 체인 → 단순 응답 폴백

### 4. Rate Limiter vs Retry의 차이

| 전략 | 방식 | 장점 | 단점 |
|------|------|------|------|
| Retry | 실패 후 재시도 | 간단한 구현 | 레이트 리밋 시 효과 없음 |
| Rate Limiter | 사전 속도 제한 | 레이트 리밋 원천 차단 | 처리량 감소 |
| 조합 | Rate Limiter + Retry | 최선의 결과 | 설정 복잡 |

---

## 💻 코드 예제

### 예제 1: with_retry()로 일시적 오류 처리

```python
# retry_basic.py
import os
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

# 기본 재시도: 네트워크 오류, 레이트 리밋에 자동 대응
llm_with_retry = llm.with_retry(
    stop_after_attempt=3,          # 최대 3회 시도
    wait_exponential_jitter=True,  # 지수 백오프 + 랜덤 지터
)

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "간결하게 답변하세요."),
        ("human", "{question}"),
    ])
    | llm_with_retry
    | StrOutputParser()
)

# 일반 호출과 동일하게 사용합니다
result = chain.invoke({"question": "파이썬 예외 처리 방법은?"})
print(result)


# 특정 예외 유형에만 재시도 적용
from openai import RateLimitError, APIConnectionError, APITimeoutError

llm_selective_retry = llm.with_retry(
    retry_if_exception_type=(
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
    ),
    stop_after_attempt=5,
    wait_exponential_jitter=True,
)

print("\n선택적 재시도 LLM 생성 완료")
```

### 예제 2: with_fallbacks()로 폴백 체인 구성

```python
# fallbacks_basic.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

load_dotenv()


def make_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )


prompt = ChatPromptTemplate.from_messages([
    ("system", "도움이 되는 어시스턴트입니다."),
    ("human", "{question}"),
])
parser = StrOutputParser()

# 기본 체인: 고성능 모델
primary_chain = prompt | make_llm("openai/gpt-4o") | parser

# 폴백 체인 1: 더 저렴한 모델
fallback_chain_1 = prompt | make_llm("openai/gpt-4o-mini") | parser

# 폴백 체인 2: 최후 수단 (정적 응답)
def emergency_response(inputs: dict) -> str:
    """API를 사용할 수 없을 때 기본 응답을 반환합니다."""
    question = inputs.get("question", "")
    return (
        f"죄송합니다. 현재 AI 서비스가 일시적으로 불가합니다. "
        f"질문 '{question[:50]}'에 대해 잠시 후 다시 시도해 주세요."
    )

fallback_chain_2 = RunnableLambda(emergency_response)

# 폴백 체인 연결: primary → fallback_1 → fallback_2
resilient_chain = primary_chain.with_fallbacks(
    [fallback_chain_1, fallback_chain_2],
    exceptions_to_handle=(Exception,),  # 모든 예외에 폴백 적용
)

result = resilient_chain.invoke({"question": "파이썬의 특징은 무엇인가요?"})
print("응답:", result)
```

### 예제 3: 타임아웃 설정

```python
# timeout_handling.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

load_dotenv()


def make_llm_with_timeout(timeout_seconds: float = 30.0) -> ChatOpenAI:
    """타임아웃이 설정된 LLM을 생성합니다."""
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        timeout=timeout_seconds,      # 요청 타임아웃 (초)
        max_retries=0,                # 타임아웃 시 즉시 실패
    )


# 짧은 타임아웃 (빠른 응답 필요한 경우)
fast_llm = make_llm_with_timeout(timeout_seconds=5.0)

# 긴 타임아웃 (복잡한 작업)
slow_llm = make_llm_with_timeout(timeout_seconds=60.0)


def timeout_fallback(inputs: dict) -> str:
    """타임아웃 발생 시 캐시에서 유사한 응답을 반환합니다."""
    # 실제 환경에서는 Redis 등에서 이전 응답을 조회합니다
    return "응답 시간이 초과되었습니다. 더 간단한 질문으로 다시 시도해 주세요."


prompt = ChatPromptTemplate.from_messages([
    ("system", "간결하게 답변하세요."),
    ("human", "{question}"),
])
parser = StrOutputParser()

# 타임아웃 체인 + 폴백
chain_with_timeout = (
    (prompt | fast_llm | parser)
    .with_fallbacks(
        [RunnableLambda(timeout_fallback)],
        exceptions_to_handle=(Exception,),
    )
)

result = chain_with_timeout.invoke({"question": "파이썬 타입 힌트란?"})
print("결과:", result)
```

### 예제 4: InMemoryRateLimiter로 속도 제한

```python
# rate_limiter.py
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter
from pydantic import SecretStr

load_dotenv()

# 레이트 리미터 설정
# requests_per_second: 초당 최대 요청 수
# check_every_n_seconds: 체크 주기
# max_bucket_size: 버스트 허용량 (토큰 버킷 알고리즘)
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,   # 초당 0.5요청 = 2초에 1요청
    check_every_n_seconds=0.1,  # 100ms마다 체크
    max_bucket_size=5,          # 최대 5개 버스트 허용
)

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    rate_limiter=rate_limiter,  # LLM에 직접 연결
)

chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | llm
    | StrOutputParser()
)

questions = [
    "파이썬이란?",
    "리스트 컴프리헨션이란?",
    "제너레이터란?",
    "데코레이터란?",
    "컨텍스트 매니저란?",
]

print(f"레이트 리미터: {rate_limiter.requests_per_second}req/s")
print(f"예상 총 소요시간: ~{len(questions) / rate_limiter.requests_per_second:.0f}초\n")

start = time.perf_counter()

for i, question in enumerate(questions, 1):
    response = chain.invoke({"question": question})
    elapsed = time.perf_counter() - start
    print(f"[{elapsed:.1f}s] Q{i}: {question} → {response[:50]}...")

total = time.perf_counter() - start
print(f"\n총 소요시간: {total:.1f}초 ({len(questions)}개 요청)")
```

### 예제 5: 재시도 + 폴백 + 레이트 리미터 조합

```python
# resilient_chain.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

load_dotenv()


def build_resilient_llm_chain(
    primary_model: str = "openai/gpt-4o",
    fallback_model: str = "openai/gpt-4o-mini",
    requests_per_second: float = 1.0,
    max_attempts: int = 3,
    timeout: float = 30.0,
):
    """회복력 있는 LLM 체인을 생성합니다."""

    # 레이트 리미터 (공유)
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=0.1,
        max_bucket_size=3,
    )

    def make_llm(model: str) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            timeout=timeout,
            rate_limiter=rate_limiter,
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "도움이 되는 어시스턴트입니다. 간결하게 답변하세요."),
        ("human", "{question}"),
    ])
    parser = StrOutputParser()

    # 기본 체인 (재시도 포함)
    primary_chain = (
        prompt | make_llm(primary_model).with_retry(stop_after_attempt=max_attempts) | parser
    )

    # 폴백 체인 (재시도 포함)
    fallback_chain = (
        prompt | make_llm(fallback_model).with_retry(stop_after_attempt=2) | parser
    )

    # 최후 폴백 (정적 응답)
    def last_resort(inputs: dict) -> str:
        return (
            "현재 AI 서비스를 이용할 수 없습니다. "
            "잠시 후 다시 시도해 주세요."
        )

    emergency_chain = RunnableLambda(last_resort)

    # 전체 체인 조합
    return primary_chain.with_fallbacks(
        [fallback_chain, emergency_chain],
        exceptions_to_handle=(Exception,),
    )


# 회복력 있는 체인 생성
resilient_chain = build_resilient_llm_chain(
    primary_model="openai/gpt-4o-mini",  # 테스트용 (실제로는 gpt-4o)
    fallback_model="openai/gpt-4o-mini",
    requests_per_second=2.0,
    max_attempts=3,
    timeout=30.0,
)

# 여러 질문에 회복력 있게 응답
questions = [
    "파이썬의 주요 특징은?",
    "LCEL이란 무엇인가요?",
    "LangGraph의 장점은?",
]

for question in questions:
    try:
        result = resilient_chain.invoke({"question": question})
        print(f"Q: {question}")
        print(f"A: {result[:100]}...\n")
    except Exception as e:
        print(f"Q: {question}")
        print(f"모든 폴백 실패: {e}\n")
```

### 예제 6: LangGraph 그래프에서 에러 처리

```python
# langgraph_error_handling.py
import os
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from pydantic import SecretStr

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


class AgentState(TypedDict):
    question: str
    answer: str
    error: Optional[str]
    retry_count: int


def safe_llm_call(state: AgentState) -> AgentState:
    """LLM 호출을 안전하게 수행하고 오류를 상태에 기록합니다."""
    if state["retry_count"] >= 3:
        return {
            "error": "최대 재시도 횟수 초과",
            "answer": "죄송합니다. 현재 응답을 생성할 수 없습니다.",
        }

    try:
        response = llm.invoke([
            HumanMessage(content=state["question"])
        ])
        return {
            "answer": response.content,
            "error": None,
        }

    except Exception as e:
        return {
            "error": str(e),
            "retry_count": state["retry_count"] + 1,
        }


def validate_answer(state: AgentState) -> AgentState:
    """답변의 기본 품질을 검증합니다."""
    answer = state.get("answer", "")

    if len(answer) < 10:
        return {
            "error": "답변이 너무 짧습니다.",
            "retry_count": state["retry_count"] + 1,
        }

    return {"error": None}


def route_after_llm(state: AgentState) -> str:
    """LLM 호출 결과에 따라 다음 노드를 결정합니다."""
    if state.get("error") and state["retry_count"] < 3:
        return "retry"        # 재시도
    elif state.get("error"):
        return "handle_error"  # 에러 처리
    else:
        return "validate"      # 검증으로 이동


def route_after_validate(state: AgentState) -> str:
    """검증 결과에 따라 다음 노드를 결정합니다."""
    if state.get("error"):
        return "retry"
    return "done"


def handle_error_node(state: AgentState) -> AgentState:
    """최종 에러 처리 노드입니다."""
    error_msg = state.get("error", "알 수 없는 오류")
    print(f"에러 발생: {error_msg}")
    return {
        "answer": f"오류로 인해 응답을 제공할 수 없습니다. ({error_msg})",
    }


# 그래프 구성
builder = StateGraph(AgentState)
builder.add_node("llm_call", safe_llm_call)
builder.add_node("validate", validate_answer)
builder.add_node("handle_error", handle_error_node)
# retry는 llm_call 노드를 재사용하는 엣지로 처리

builder.set_entry_point("llm_call")

builder.add_conditional_edges(
    "llm_call",
    route_after_llm,
    {
        "retry": "llm_call",       # 자신으로 다시 라우팅 (재시도)
        "validate": "validate",
        "handle_error": "handle_error",
    },
)

builder.add_conditional_edges(
    "validate",
    route_after_validate,
    {
        "retry": "llm_call",
        "done": END,
    },
)

builder.add_edge("handle_error", END)

graph = builder.compile()

# 실행
result = graph.invoke({
    "question": "파이썬에서 예외 처리를 어떻게 하나요?",
    "answer": "",
    "error": None,
    "retry_count": 0,
})

print(f"최종 답변: {result['answer'][:200]}")
if result.get("error"):
    print(f"에러: {result['error']}")
```

---

## ✏️ 실습 과제

### 과제 1: 재시도 정책 비교 (필수)

`with_retry()`의 세 가지 설정을 비교합니다:
- `stop_after_attempt=1` (재시도 없음)
- `stop_after_attempt=3` + `wait_exponential_jitter=True`
- `stop_after_attempt=5` + `wait_exponential_jitter=True`

FakeListChatModel에서 일부러 오류를 발생시켜 재시도 동작을 테스트합니다.

### 과제 2: 3단계 폴백 시스템 (중급)

다음 3단계 폴백 체인을 구현합니다:
1. **1단계**: GPT-4o (고성능)
2. **2단계**: GPT-4o-mini (저렴)
3. **3단계**: 로컬 캐시에서 이전 응답 반환

각 단계가 언제 활성화되는지 로그로 기록합니다.

### 과제 3: 회복력 있는 LangGraph 에이전트 (심화)

Phase 27-34에서 만든 에이전트 중 하나를 선택해 다음을 추가합니다:
- 각 노드에 try/except 에러 처리
- 에러 횟수를 상태에 기록
- 3회 이상 실패 시 에러 처리 노드로 라우팅
- 성공/실패 통계를 출력하는 후처리 노드

---

## ⚠️ 흔한 함정

### 1. 인증 오류는 재시도하지 마세요

```python
from openai import AuthenticationError

# AuthenticationError는 재시도해도 의미가 없습니다
# API 키가 틀렸으면 계속 실패합니다
llm_with_smart_retry = llm.with_retry(
    retry_if_exception_type=(
        # AuthenticationError 제외!
        RateLimitError,
        APIConnectionError,
    ),
    stop_after_attempt=3,
)
```

### 2. 폴백이 무한 루프를 만들지 않도록 주의

```python
# 위험: 폴백 체인도 같은 오류가 발생하면 무한 루프 가능성
chain.with_fallbacks([chain])  # 자기 자신을 폴백으로 설정 금지

# 안전: 반드시 정적 응답 폴백을 마지막에 추가
chain.with_fallbacks([
    fallback_chain,
    RunnableLambda(lambda x: "기본 응답"),  # 항상 성공
])
```

### 3. 레이트 리미터 공유 주의

```python
# 여러 LLM 인스턴스가 동일한 rate_limiter를 공유하면
# 의도치 않게 전체 처리량이 줄어들 수 있습니다
rate_limiter = InMemoryRateLimiter(requests_per_second=1.0)

llm1 = ChatOpenAI(..., rate_limiter=rate_limiter)  # 0.5 req/s 효과
llm2 = ChatOpenAI(..., rate_limiter=rate_limiter)  # 0.5 req/s 효과

# 독립적으로 제한하려면 별도 rate_limiter를 생성하세요
```

### 4. LangGraph에서 상태 누적 방지

```python
# 재시도 시 상태가 예상치 못하게 누적될 수 있습니다
# retry_count를 명시적으로 관리하세요

class AgentState(TypedDict):
    retry_count: int  # 재시도 횟수 명시적 추적
    errors: list[str]  # 오류 이력 추적 (선택)
```

---

## ✅ 셀프 체크

- [ ] LLM 앱에서 발생하는 4가지 이상의 예외 유형을 설명합니다.
- [ ] `.with_retry(stop_after_attempt=3)`을 체인에 적용했습니다.
- [ ] 지수 백오프와 지터의 의미를 설명합니다.
- [ ] `.with_fallbacks()`로 2단계 이상의 폴백 체인을 구성했습니다.
- [ ] 마지막 폴백으로 항상 성공하는 정적 응답을 추가했습니다.
- [ ] `timeout` 파라미터로 무한 대기를 방지했습니다.
- [ ] `InMemoryRateLimiter`로 API 레이트 리밋을 사전에 제어했습니다.
- [ ] LangGraph 노드에서 try/except로 오류를 캡처하고 상태에 기록했습니다.
- [ ] 재시도 횟수를 상태에서 추적하는 LangGraph 패턴을 구현했습니다.

---

## 🔗 참고 자료

- [LangChain with_retry() 가이드](https://python.langchain.com/docs/concepts/runnables/#with-retry)
- [LangChain with_fallbacks() 가이드](https://python.langchain.com/docs/how_to/fallbacks/)
- [InMemoryRateLimiter API](https://python.langchain.com/api_reference/core/rate_limiters/langchain_core.rate_limiters.InMemoryRateLimiter.html)
- [LangGraph 에러 처리](https://langchain-ai.github.io/langgraph/how-tos/error-handling/)
- [OpenAI 예외 타입](https://platform.openai.com/docs/guides/error-codes)

> **API 변동 안내**: `with_retry()`와 `with_fallbacks()`의 파라미터 이름은 LangChain 버전에 따라 변경될 수 있습니다. 특히 `stop_after_attempt` vs `retry_stop` 등의 파라미터 이름을 [공식 문서](https://python.langchain.com/docs/concepts/runnables/)에서 확인하세요. `InMemoryRateLimiter`는 `langchain-core`에 포함되어 있으며, 외부 패키지가 불필요합니다.

---

← [Phase 39: 비용·캐싱 최적화](39-cost-caching-optimization.md) | [Phase 41: 보안·가드레일](41-security-guardrails.md) →
