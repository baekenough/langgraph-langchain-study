# Phase 38: 테스트 전략

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 90분 |
| 난이도 | ★★★★☆ |
| 선행 학습 | Phase 07 (LCEL), Phase 18-22 (LangGraph 기초) |

---

## 🎯 학습 목표

- LLM 애플리케이션 테스트가 기존 소프트웨어 테스트와 어떻게 다른지 설명합니다.
- `GenericFakeChatModel`과 `FakeListChatModel`로 LLM 의존성 없이 단위 테스트를 작성합니다.
- LCEL 체인과 LangGraph 그래프를 테스트하는 패턴을 이해합니다.
- 비결정성을 다루는 테스트 전략을 적용합니다.
- pytest와 pytest-asyncio로 동기·비동기 테스트를 구성합니다.
- CI 파이프라인에 LLM 앱 테스트를 통합하는 개념을 이해합니다.

---

## 📚 핵심 개념

### 1. LLM 테스트가 어려운 이유

```
일반 소프트웨어 테스트              LLM 앱 테스트
────────────────────────          ─────────────────────────
• 입력 → 정확한 출력              • 입력 → "좋은" 출력 (주관적)
• 단위 테스트 명확                • 비결정성 (temperature > 0)
• 빠르고 저렴                    • 느리고 비쌈 (API 비용)
• 100% 재현 가능                 • 재현이 어려움
• assert a == b                  • "의미론적으로 유사한가?"
```

### 2. 테스트 피라미드 (LLM 버전)

```
              △ E2E 테스트 (실제 LLM)
             ╱ ╲    • 핵심 시나리오만
            ╱   ╲   • LangSmith 평가와 연계
           ╱─────╲
          ╱ 통합   ╲  • 컴포넌트 간 연결 검증
         ╱  테스트  ╲  • Mock LLM 또는 저렴한 모델
        ╱────────────╲
       ╱   단위 테스트  ╲ • Fake LLM으로 빠른 검증
      ╱     (대부분)    ╲  • 입출력 형식, 비즈니스 로직
     ╱──────────────────╲
```

### 3. Fake 모델 선택 기준

| 모델 | 임포트 경로 | 사용 시기 |
|------|-----------|---------|
| `GenericFakeChatModel` | `langchain_core.language_models.fake_chat_models` | 커스텀 응답 시퀀스 필요 시 |
| `FakeListChatModel` | `langchain_core.language_models.fake_chat_models` | 미리 정의한 문자열 목록으로 응답 |
| `FakeStreamingListLLM` | `langchain_core.language_models.fake` | 스트리밍 테스트 |

### 4. 비결정성 처리 전략

| 전략 | 설명 | 적합한 경우 |
|------|------|------------|
| Fake LLM 고정 | 항상 동일한 응답 반환 | 형식·로직 테스트 |
| temperature=0 | 결정론적 응답 (완전하지 않음) | 통합 테스트 |
| 의미론적 검사 | "키워드 포함 여부" 등 | 내용 품질 테스트 |
| LangSmith 평가 | LLM-as-judge로 분포 평가 | 회귀 방지 |

---

## 💻 코드 예제

### 예제 1: 기본 Fake LLM 단위 테스트

```python
# tests/test_chain_basic.py
import pytest
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
    FakeListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def make_fake_llm(responses: list[str]) -> FakeListChatModel:
    """테스트용 Fake LLM을 생성합니다."""
    return FakeListChatModel(responses=responses)


class TestQAChain:
    """QA 체인의 단위 테스트 클래스입니다."""

    def setup_method(self):
        """각 테스트 전에 공통 설정을 초기화합니다."""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 도움이 되는 어시스턴트입니다."),
            ("human", "{question}"),
        ])
        self.parser = StrOutputParser()

    def test_chain_returns_string(self):
        """체인이 문자열을 반환하는지 확인합니다."""
        fake_llm = make_fake_llm(["파이썬은 인터프리터 언어입니다."])
        chain = self.prompt | fake_llm | self.parser

        result = chain.invoke({"question": "파이썬이란?"})

        assert isinstance(result, str)
        assert len(result) > 0

    def test_chain_passes_question_to_llm(self):
        """체인이 사용자 질문을 LLM에 올바르게 전달하는지 확인합니다."""
        expected_response = "이것은 테스트 응답입니다."
        fake_llm = make_fake_llm([expected_response])
        chain = self.prompt | fake_llm | self.parser

        result = chain.invoke({"question": "테스트 질문"})

        assert result == expected_response

    def test_chain_handles_multiple_calls(self):
        """체인이 여러 번 호출될 때 순서대로 응답하는지 확인합니다."""
        responses = ["첫 번째 응답", "두 번째 응답", "세 번째 응답"]
        fake_llm = make_fake_llm(responses)
        chain = self.prompt | fake_llm | self.parser

        results = [
            chain.invoke({"question": f"질문 {i}"})
            for i in range(3)
        ]

        assert results == responses

    def test_chain_with_empty_question(self):
        """빈 질문에 대한 체인의 동작을 확인합니다."""
        fake_llm = make_fake_llm(["빈 질문에 대한 응답"])
        chain = self.prompt | fake_llm | self.parser

        # 빈 문자열도 처리 가능해야 합니다
        result = chain.invoke({"question": ""})
        assert isinstance(result, str)
```

### 예제 2: GenericFakeChatModel을 활용한 구조적 테스트

```python
# tests/test_structured_output.py
import pytest
import json
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel


class SentimentResult(BaseModel):
    label: str
    confidence: float
    reason: str


def test_structured_output_parsing():
    """구조화된 JSON 출력 파싱을 테스트합니다."""
    # GenericFakeChatModel은 AIMessage를 직접 반환합니다
    fake_response = json.dumps({
        "label": "positive",
        "confidence": 0.95,
        "reason": "긍정적인 어조와 표현이 사용되었습니다.",
    })

    fake_llm = GenericFakeChatModel(
        messages=iter([AIMessage(content=fake_response)])
    )

    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "감성을 JSON으로 분석하세요."),
            ("human", "{text}"),
        ])
        | fake_llm
        | JsonOutputParser()
    )

    result = chain.invoke({"text": "오늘 정말 행복한 날이에요!"})

    assert result["label"] == "positive"
    assert result["confidence"] == 0.95
    assert "reason" in result


def test_json_parse_error_handling():
    """잘못된 JSON 응답에 대한 파싱 오류를 테스트합니다."""
    fake_llm = GenericFakeChatModel(
        messages=iter([AIMessage(content="유효하지 않은 JSON")])
    )

    chain = (
        ChatPromptTemplate.from_messages([("human", "{text}")])
        | fake_llm
        | JsonOutputParser()
    )

    with pytest.raises(Exception):
        chain.invoke({"text": "테스트"})
```

### 예제 3: LangGraph 그래프 단위 테스트

```python
# tests/test_langgraph.py
import pytest
from typing import TypedDict
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    query: str
    analysis: str
    final_answer: str


def build_test_graph(llm):
    """테스트용 LangGraph 그래프를 생성합니다."""

    def analyze_node(state: AgentState) -> AgentState:
        response = llm.invoke([HumanMessage(content=f"분석: {state['query']}")])
        return {"analysis": response.content}

    def answer_node(state: AgentState) -> AgentState:
        prompt = f"분석 결과: {state['analysis']}\n최종 답변:"
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"final_answer": response.content}

    builder = StateGraph(AgentState)
    builder.add_node("analyze", analyze_node)
    builder.add_node("answer", answer_node)
    builder.set_entry_point("analyze")
    builder.add_edge("analyze", "answer")
    builder.add_edge("answer", END)

    return builder.compile()


class TestLangGraph:
    """LangGraph 그래프 테스트 클래스입니다."""

    def test_graph_completes_successfully(self):
        """그래프가 정상적으로 완료되는지 확인합니다."""
        fake_llm = FakeListChatModel(
            responses=["분석 완료: 기본 파이썬 개념 질문입니다.", "파이썬은 고수준 언어입니다."]
        )
        graph = build_test_graph(fake_llm)

        result = graph.invoke({"query": "파이썬이란?", "analysis": "", "final_answer": ""})

        assert "final_answer" in result
        assert len(result["final_answer"]) > 0

    def test_graph_state_transitions(self):
        """그래프의 상태 전환이 올바른지 확인합니다."""
        analysis_response = "이것은 분석 결과입니다."
        final_response = "이것은 최종 답변입니다."

        fake_llm = FakeListChatModel(
            responses=[analysis_response, final_response]
        )
        graph = build_test_graph(fake_llm)

        result = graph.invoke({"query": "테스트", "analysis": "", "final_answer": ""})

        # 각 노드가 올바른 상태를 생성했는지 확인
        assert result["analysis"] == analysis_response
        assert result["final_answer"] == final_response

    def test_graph_handles_empty_query(self):
        """빈 쿼리에도 그래프가 오류 없이 실행되는지 확인합니다."""
        fake_llm = FakeListChatModel(
            responses=["빈 쿼리 분석", "기본 응답"]
        )
        graph = build_test_graph(fake_llm)

        result = graph.invoke({"query": "", "analysis": "", "final_answer": ""})
        assert "final_answer" in result
```

### 예제 4: 비동기 체인 테스트

```python
# tests/test_async_chain.py
import pytest
import asyncio
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


@pytest.fixture
def async_chain():
    """비동기 테스트용 체인을 생성하는 픽스처입니다."""
    fake_llm = FakeListChatModel(
        responses=["비동기 응답 1", "비동기 응답 2", "비동기 응답 3"]
    )
    prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
    return prompt | fake_llm | StrOutputParser()


@pytest.mark.asyncio
async def test_async_invoke(async_chain):
    """ainvoke가 올바르게 동작하는지 확인합니다."""
    result = await async_chain.ainvoke({"question": "테스트 질문"})

    assert isinstance(result, str)
    assert result == "비동기 응답 1"


@pytest.mark.asyncio
async def test_async_batch(async_chain):
    """abatch가 여러 입력을 병렬 처리하는지 확인합니다."""
    questions = [
        {"question": "질문 1"},
        {"question": "질문 2"},
        {"question": "질문 3"},
    ]

    results = await async_chain.abatch(questions)

    assert len(results) == 3
    assert all(isinstance(r, str) for r in results)


@pytest.mark.asyncio
async def test_async_stream(async_chain):
    """astream이 청크를 순서대로 반환하는지 확인합니다."""
    chunks = []
    async for chunk in async_chain.astream({"question": "스트리밍 테스트"}):
        chunks.append(chunk)

    # 모든 청크를 합치면 완성된 응답이 되어야 합니다
    full_response = "".join(chunks)
    assert len(full_response) > 0
```

### 예제 5: pytest 설정 및 픽스처 구성

```python
# tests/conftest.py
import os
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


@pytest.fixture(autouse=True)
def disable_langsmith_tracing():
    """테스트 중 LangSmith 트레이싱을 비활성화합니다."""
    original = os.environ.get("LANGSMITH_TRACING")
    os.environ["LANGSMITH_TRACING"] = "false"
    yield
    if original is not None:
        os.environ["LANGSMITH_TRACING"] = original
    else:
        os.environ.pop("LANGSMITH_TRACING", None)


@pytest.fixture
def fake_llm_factory():
    """다양한 응답 시퀀스로 Fake LLM을 생성하는 팩토리입니다."""
    def _factory(responses: list[str]) -> FakeListChatModel:
        return FakeListChatModel(responses=responses)
    return _factory


@pytest.fixture
def simple_chain(fake_llm_factory):
    """단순 QA 체인 픽스처입니다."""
    fake_llm = fake_llm_factory(["테스트 응답"] * 10)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "도움이 되는 어시스턴트입니다."),
        ("human", "{question}"),
    ])
    return prompt | fake_llm | StrOutputParser()


# pytest.ini 또는 pyproject.toml에 추가할 설정:
# [pytest]
# asyncio_mode = auto          # pytest-asyncio 자동 모드
# testpaths = tests            # 테스트 디렉토리
# filterwarnings = ignore::DeprecationWarning
```

```python
# tests/test_with_fixtures.py
import pytest


def test_simple_chain_invoke(simple_chain):
    """픽스처를 활용한 체인 테스트입니다."""
    result = simple_chain.invoke({"question": "안녕하세요"})
    assert result == "테스트 응답"


def test_fake_llm_factory(fake_llm_factory):
    """팩토리 픽스처로 커스텀 응답을 설정합니다."""
    from langchain_core.messages import HumanMessage

    custom_llm = fake_llm_factory(["커스텀 응답 A", "커스텀 응답 B"])

    response1 = custom_llm.invoke([HumanMessage(content="첫 번째")])
    response2 = custom_llm.invoke([HumanMessage(content="두 번째")])

    assert "커스텀 응답 A" in response1.content
    assert "커스텀 응답 B" in response2.content
```

### 예제 6: CI 통합 개념 (GitHub Actions)

```yaml
# .github/workflows/test.yml
name: LLM App Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  unit-tests:
    name: 단위 테스트 (Fake LLM)
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Python 설정
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: UV 설치 및 의존성 설치
        run: |
          pip install uv
          uv sync --group dev

      - name: 단위 테스트 실행 (API 키 불필요)
        run: uv run pytest tests/ -v -k "not integration and not e2e"
        env:
          # LangSmith 트레이싱 비활성화
          LANGSMITH_TRACING: "false"

  integration-tests:
    name: 통합 테스트 (실제 LLM)
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'  # main 브랜치에서만

    steps:
      - uses: actions/checkout@v4

      - name: Python 설정
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: UV 설치 및 의존성 설치
        run: |
          pip install uv
          uv sync --group dev

      - name: 통합 테스트 실행
        run: uv run pytest tests/ -v -k "integration"
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
          LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
          LANGSMITH_TRACING: "true"
          LANGSMITH_PROJECT: "ci-integration-tests"
```

---

## ✏️ 실습 과제

### 과제 1: LCEL 체인 단위 테스트 스위트 (필수)

Phase 07에서 만든 LCEL 체인에 대한 단위 테스트를 작성합니다:
- 정상 입력 테스트 (최소 3개)
- 경계 조건 테스트 (빈 입력, 매우 긴 입력)
- 출력 형식 검증 테스트

### 과제 2: LangGraph 테스트 (중급)

Phase 22(체크포인터)에서 만든 그래프를 테스트합니다:
- 노드 단위 테스트 (각 노드 함수를 독립적으로 테스트)
- 상태 전환 테스트 (노드 실행 후 상태가 올바르게 변경되는지)
- 오류 케이스 테스트 (LLM이 예상치 못한 응답을 반환할 때)

### 과제 3: 비동기 테스트 (심화)

Phase 08(스트리밍/비동기)에서 배운 `ainvoke`, `astream`을 테스트합니다:
- pytest-asyncio로 비동기 테스트 설정
- 스트리밍 청크의 순서와 완전성 검증
- 병렬 `abatch` 실행 성능 테스트

---

## ⚠️ 흔한 함정

### 1. FakeListChatModel 응답 소진

```python
# 오류: 응답 목록 소진 후 다시 호출
fake_llm = FakeListChatModel(responses=["응답 1"])
chain.invoke({"question": "질문 1"})  # OK
chain.invoke({"question": "질문 2"})  # StopIteration 오류!

# 해결: 충분한 응답 제공 또는 cycle 파라미터 사용
# (FakeListChatModel이 cycle을 지원하지 않으면 더 많은 응답 제공)
fake_llm = FakeListChatModel(responses=["응답"] * 100)
```

### 2. 테스트에서 실제 API 호출 방지

```python
# conftest.py에서 API 키가 없으면 실제 LLM 테스트를 건너뜁니다
import pytest

def pytest_configure(config):
    """커스텀 마커를 등록합니다."""
    config.addinivalue_line("markers", "integration: 실제 LLM이 필요한 통합 테스트")

@pytest.fixture(autouse=True)
def check_integration_marker(request):
    """integration 마커가 있는 테스트는 API 키가 없으면 건너뜁니다."""
    if request.node.get_closest_marker("integration"):
        if not os.environ.get("OPENROUTER_API_KEY"):
            pytest.skip("OPENROUTER_API_KEY가 없어 건너뜁니다.")
```

### 3. 비동기 테스트 설정 누락

```python
# pyproject.toml에 asyncio_mode 설정이 없으면 비동기 테스트가 실패합니다
# pyproject.toml:
# [tool.pytest.ini_options]
# asyncio_mode = "auto"
```

---

## ✅ 셀프 체크

- [ ] LLM 앱 테스트가 기존 테스트와 다른 이유를 3가지 설명할 수 있습니다.
- [ ] `FakeListChatModel`로 LCEL 체인의 단위 테스트를 작성했습니다.
- [ ] `GenericFakeChatModel`로 구조화된 출력(JSON)을 테스트했습니다.
- [ ] LangGraph 노드 함수를 독립적으로 단위 테스트했습니다.
- [ ] pytest 픽스처(`conftest.py`)로 공통 설정을 중앙화했습니다.
- [ ] `@pytest.mark.asyncio`로 비동기 체인을 테스트했습니다.
- [ ] CI 파이프라인에서 단위 테스트와 통합 테스트를 분리하는 전략을 이해합니다.
- [ ] 테스트 중 LangSmith 트레이싱을 비활성화하는 방법을 구현했습니다.

---

## 🔗 참고 자료

- [LangChain 테스트 가이드](https://python.langchain.com/docs/contributing/testing/)
- [FakeChatModel API](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.fake_chat_models.FakeListChatModel.html)
- [pytest-asyncio 문서](https://pytest-asyncio.readthedocs.io/)
- [pytest 공식 문서](https://docs.pytest.org/)

> **API 변동 안내**: `GenericFakeChatModel`과 `FakeListChatModel`의 임포트 경로는 LangChain 버전에 따라 다를 수 있습니다. `langchain_core.language_models.fake_chat_models` 또는 `langchain_core.language_models.fake`를 시도해보고, 없으면 [공식 문서](https://python.langchain.com)에서 최신 경로를 확인하세요.

---

← [Phase 37: 프롬프트 관리](37-prompt-management.md) | [Phase 39: 비용·캐싱 최적화](39-cost-caching-optimization.md) →
