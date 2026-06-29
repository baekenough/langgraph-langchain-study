# Phase 35: LangSmith 트레이싱

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 90분 |
| 난이도 | ★★★☆☆ |
| 선행 학습 | Phase 34 (Agentic RAG) |

---

## 🎯 학습 목표

- LangSmith 가입 및 API 키 발급 절차를 완료할 수 있습니다.
- 환경변수 설정만으로 LangChain/LangGraph 호출이 자동 트레이싱되는 원리를 이해합니다.
- LangSmith UI에서 실행(run) 트리를 탐색하고 각 단계의 입출력을 확인합니다.
- `@traceable` 데코레이터로 일반 Python 함수를 트레이스에 포함시킵니다.
- 트레이싱 데이터를 활용한 디버깅 워크플로를 경험합니다.
- 프로젝트(project)를 분리해 환경별 트레이스를 관리합니다.

---

## 📚 핵심 개념

### 1. LangSmith란?

**LangSmith**는 LangChain 팀이 만든 LLM 애플리케이션 관측(observability) 플랫폼입니다.
LangChain/LangGraph로 만든 체인·에이전트의 모든 실행 단계를 자동으로 기록하고, 디버깅·평가·모니터링을 한 곳에서 수행할 수 있습니다.

```
사용자 요청
    │
    ▼
LangChain 체인 실행  ──→  LangSmith (백그라운드 비동기 전송)
    │                         │
    ▼                         ▼
최종 응답                 run 트리 저장
                          (각 단계 입출력, 지연시간, 토큰 수)
```

### 2. 자동 트레이싱 활성화

LangChain은 환경변수 `LANGSMITH_TRACING=true`가 설정되면 모든 Runnable 호출을 자동으로 LangSmith에 전송합니다.
**코드 변경 없이** 트레이싱이 활성화됩니다.

```bash
# .env 파일
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=my-project   # 선택사항, 기본값: "default"
```

### 3. run 트리 구조

LangSmith는 체인 실행을 **트리 형태**로 저장합니다:

```
RunnableSequence  (Root)
├── ChatPromptTemplate
│   └── format_messages
├── ChatOpenAI
│   ├── input: [SystemMessage, HumanMessage]
│   ├── output: AIMessage
│   └── tokens: {input: 45, output: 120}
└── StrOutputParser
    └── output: "..."
```

각 노드에는 입력값·출력값·소요 시간·토큰 수·오류 정보가 포함됩니다.

### 4. @traceable 데코레이터

LangChain Runnable이 아닌 일반 Python 함수도 `@traceable`로 트레이스에 포함할 수 있습니다.
전처리 함수, 후처리 함수, 비즈니스 로직 등을 포함시키면 전체 파이프라인을 한눈에 파악할 수 있습니다.

### 5. 프로젝트 분리

`LANGSMITH_PROJECT` 환경변수로 트레이스를 프로젝트별로 분리합니다:

| 프로젝트 이름 예시 | 용도 |
|------------------|------|
| `dev` | 로컬 개발 중 실험 |
| `staging` | 스테이징 환경 검증 |
| `production` | 프로덕션 모니터링 |
| `eval-experiment-1` | 특정 평가 실험 |

---

## 💻 코드 예제

### 예제 1: 환경 설정 및 자동 트레이싱 확인

```python
# trace_basic.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

load_dotenv()

# LangSmith 설정 확인
assert os.environ.get("LANGSMITH_TRACING") == "true", "LANGSMITH_TRACING 미설정"
assert os.environ.get("LANGSMITH_API_KEY"), "LANGSMITH_API_KEY 미설정"

# 현재 프로젝트 확인
project = os.environ.get("LANGSMITH_PROJECT", "default")
print(f"LangSmith 프로젝트: {project}")

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 요약 전문가입니다."),
    ("human", "다음 텍스트를 한 문장으로 요약하세요: {text}"),
])

chain = prompt | llm | StrOutputParser()

# 이 호출은 LangSmith에 자동으로 트레이스됩니다
result = chain.invoke({
    "text": "LangSmith는 LLM 애플리케이션의 디버깅과 모니터링을 위한 플랫폼입니다. "
            "체인 실행의 모든 단계를 기록하여 개발자가 문제를 빠르게 파악하도록 돕습니다."
})

print("결과:", result)
print(f"\nLangSmith UI에서 확인: https://smith.langchain.com/")
```

### 예제 2: @traceable로 커스텀 함수 추적

```python
# trace_custom.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from pydantic import SecretStr

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 감성 분석 전문가입니다."),
    ("human", "다음 리뷰의 감성을 분석하세요: {review}"),
])

chain = prompt | llm | StrOutputParser()


@traceable(name="전처리: 텍스트 정제")
def preprocess_review(raw_text: str) -> str:
    """리뷰 텍스트에서 특수문자를 제거하고 공백을 정리합니다."""
    import re
    cleaned = re.sub(r"[^\w\s가-힣]", " ", raw_text)
    cleaned = " ".join(cleaned.split())
    return cleaned


@traceable(name="후처리: 결과 파싱")
def parse_sentiment_result(raw_output: str) -> dict:
    """LLM 출력을 구조화된 딕셔너리로 변환합니다."""
    result = {"raw": raw_output, "label": "unknown"}

    if "긍정" in raw_output or "positive" in raw_output.lower():
        result["label"] = "positive"
    elif "부정" in raw_output or "negative" in raw_output.lower():
        result["label"] = "negative"
    else:
        result["label"] = "neutral"

    return result


@traceable(name="감성 분석 파이프라인")
def analyze_sentiment(raw_review: str) -> dict:
    """전체 감성 분석 파이프라인을 실행합니다."""
    # 전처리
    clean_review = preprocess_review(raw_review)

    # LLM 호출
    llm_output = chain.invoke({"review": clean_review})

    # 후처리
    result = parse_sentiment_result(llm_output)
    result["original_input"] = raw_review

    return result


# 실행 - LangSmith에서 중첩된 트레이스 트리를 확인할 수 있습니다
test_reviews = [
    "이 제품 정말 최고예요!!! 배송도 빠르고 품질도 완벽합니다 👍",
    "기대했던 것보다 별로네요... 환불 요청하겠습니다.",
]

for review in test_reviews:
    result = analyze_sentiment(review)
    print(f"입력: {review[:30]}...")
    print(f"결과: {result}\n")
```

### 예제 3: 런타임 메타데이터 추가

```python
# trace_metadata.py
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

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "당신은 도움이 되는 어시스턴트입니다."),
        ("human", "{question}"),
    ])
    | llm
    | StrOutputParser()
)

# RunnableConfig를 통해 트레이스에 메타데이터를 추가합니다
result = chain.invoke(
    {"question": "파이썬에서 리스트와 튜플의 차이는 무엇인가요?"},
    config={
        "run_name": "파이썬-개념-질문",  # UI에 표시될 이름
        "tags": ["tutorial", "python-basics"],  # 필터링용 태그
        "metadata": {                           # 커스텀 메타데이터
            "user_id": "user_123",
            "session_id": "sess_abc",
            "environment": "development",
        },
    },
)

print("응답:", result)
print("\n LangSmith UI에서 태그와 메타데이터로 필터링할 수 있습니다.")
```

### 예제 4: LangGraph 트레이싱

```python
# trace_langgraph.py
import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
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
    review: str
    final: str


def answer_node(state: AgentState) -> AgentState:
    """첫 번째 답변을 생성합니다."""
    response = llm.invoke([HumanMessage(content=state["question"])])
    return {"answer": response.content}


def review_node(state: AgentState) -> AgentState:
    """답변을 검토하고 개선점을 제안합니다."""
    prompt = f"다음 답변을 검토하고 개선점을 제시하세요:\n\n{state['answer']}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"review": response.content}


def finalize_node(state: AgentState) -> AgentState:
    """검토 의견을 반영해 최종 답변을 생성합니다."""
    prompt = (
        f"원본 답변: {state['answer']}\n\n"
        f"검토 의견: {state['review']}\n\n"
        "검토 의견을 반영해 최종 답변을 작성하세요."
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"final": response.content}


# 그래프 구성
builder = StateGraph(AgentState)
builder.add_node("answer", answer_node)
builder.add_node("review", review_node)
builder.add_node("finalize", finalize_node)
builder.set_entry_point("answer")
builder.add_edge("answer", "review")
builder.add_edge("review", "finalize")
builder.add_edge("finalize", END)

graph = builder.compile()

# LangSmith는 각 노드 실행을 별도의 run으로 기록합니다
result = graph.invoke(
    {"question": "머신러닝과 딥러닝의 차이를 설명해주세요."},
    config={
        "run_name": "답변-검토-개선 그래프",
        "tags": ["langgraph", "review-loop"],
    },
)

print("최종 답변:", result["final"][:200], "...")
```

### 예제 5: 프로젝트 동적 전환

```python
# trace_project_switch.py
import os
import contextlib
from dotenv import load_dotenv
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

load_dotenv()


@contextlib.contextmanager
def langsmith_project(project_name: str):
    """LangSmith 프로젝트를 임시로 변경하는 컨텍스트 매니저입니다."""
    original = os.environ.get("LANGSMITH_PROJECT", "default")
    os.environ["LANGSMITH_PROJECT"] = project_name
    try:
        yield project_name
    finally:
        os.environ["LANGSMITH_PROJECT"] = original


llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | llm
    | StrOutputParser()
)

# 실험 A: 기본 프롬프트
with langsmith_project("experiment-baseline"):
    result_a = chain.invoke({"question": "파이썬의 GIL이란 무엇인가요?"})
    print(f"[baseline] {result_a[:80]}...")

# 실험 B: 개선된 프롬프트 (별도 프로젝트에 기록)
with langsmith_project("experiment-v2"):
    chain_v2 = (
        ChatPromptTemplate.from_messages([
            ("system", "당신은 파이썬 전문가입니다. 초보자도 이해할 수 있게 설명하세요."),
            ("human", "{question}"),
        ])
        | llm
        | StrOutputParser()
    )
    result_b = chain_v2.invoke({"question": "파이썬의 GIL이란 무엇인가요?"})
    print(f"[v2] {result_b[:80]}...")
```

---

## ✏️ 실습 과제

### 과제 1: 기본 트레이싱 설정 (필수)

1. [LangSmith](https://smith.langchain.com/)에 가입하고 API 키를 발급합니다.
2. `.env` 파일에 `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`를 설정합니다.
3. 예제 1의 코드를 실행하고 LangSmith UI에서 run 트리를 확인합니다.
4. 각 단계의 입력·출력·소요 시간을 기록합니다.

### 과제 2: 커스텀 트레이싱 (중급)

`@traceable`을 사용하여 다음 파이프라인을 트레이스합니다:
1. 사용자 입력 검증 함수 (길이 제한, 금지어 필터)
2. LLM 호출 (기존 체인 사용)
3. 출력 후처리 함수 (마크다운 제거, 길이 제한)

LangSmith UI에서 각 함수의 실행 시간을 비교하세요.

### 과제 3: 멀티 프로젝트 실험 (심화)

동일한 질문에 대해 두 가지 프롬프트 전략을 비교합니다:
- `experiment-zero-shot`: 시스템 프롬프트 없이 직접 질문
- `experiment-few-shot`: 2~3개의 예시를 포함한 few-shot 프롬프트

각 프로젝트에 5개 이상의 질문을 실행하고 UI에서 응답 품질을 비교합니다.

---

## ⚠️ 흔한 함정

### 1. 환경변수 적용 순서 문제

```python
# 잘못된 예: LangChain import 후 환경변수 설정
from langchain_openai import ChatOpenAI  # 이 시점에 설정 확인
import os
os.environ["LANGSMITH_TRACING"] = "true"  # 너무 늦을 수 있음

# 올바른 예: .env 파일을 먼저 로드
import os
from dotenv import load_dotenv
load_dotenv()  # 환경변수 먼저 설정
from langchain_openai import ChatOpenAI  # 그 후 import
```

### 2. 트레이싱 비활성화를 잊는 경우

```python
# 테스트나 벤치마크 시 트레이싱이 성능에 영향을 줄 수 있습니다
# 비활성화 방법:
os.environ["LANGSMITH_TRACING"] = "false"

# 또는 코드에서 직접 비활성화
from langchain_core.tracers.context import tracing_v2_enabled
with tracing_v2_enabled(False):
    result = chain.invoke({"question": "..."})
```

### 3. API 키 노출 주의

```python
# 절대 하드코딩하지 마세요
api_key = "lsv2_pt_abc123..."  # 위험!

# .env 파일 + .gitignore 패턴을 사용하세요
# .gitignore에 반드시 .env 추가
```

### 4. 비동기 컨텍스트에서의 @traceable

```python
from langsmith import traceable

# 비동기 함수에도 @traceable을 사용할 수 있습니다
@traceable
async def async_process(data: str) -> str:
    # 비동기 작업
    return data.upper()
```

---

## ✅ 셀프 체크

- [ ] LangSmith 계정을 생성하고 API 키를 발급했습니다.
- [ ] `.env` 파일에 세 가지 LangSmith 환경변수를 설정했습니다.
- [ ] 코드 변경 없이 체인 호출이 자동으로 트레이스됨을 확인했습니다.
- [ ] LangSmith UI에서 run 트리를 탐색하고 각 단계의 입출력을 확인했습니다.
- [ ] `@traceable` 데코레이터로 일반 Python 함수를 트레이스에 포함시켰습니다.
- [ ] `config`의 `run_name`, `tags`, `metadata`를 활용해 트레이스에 정보를 추가했습니다.
- [ ] 두 개 이상의 프로젝트를 만들어 트레이스를 분리했습니다.
- [ ] LangGraph 노드별 트레이스가 별도로 기록되는 것을 확인했습니다.

---

## 🔗 참고 자료

- [LangSmith 공식 문서](https://docs.smith.langchain.com/)
- [LangSmith Python SDK](https://docs.smith.langchain.com/sdk/python)
- [traceable 데코레이터](https://docs.smith.langchain.com/tracing/annotate_code)
- [LangChain 트레이싱 가이드](https://python.langchain.com/docs/langsmith/walkthrough)

> **API 변동 안내**: LangSmith는 활발히 개발 중인 플랫폼입니다. 환경변수명이나 SDK 메서드가 변경될 수 있으므로 [공식 문서](https://docs.smith.langchain.com/)를 항상 확인하세요.

---

← [Phase 34: Agentic RAG](../04-agents/34-agentic-rag.md) | [Phase 36: LangSmith 평가](36-langsmith-evaluation.md) →
