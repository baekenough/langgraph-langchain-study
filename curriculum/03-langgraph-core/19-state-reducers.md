# Phase 19: 상태와 리듀서

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 90분 |
| 난이도 | ★★★☆☆ |
| 선행 학습 | Phase 18 (LangGraph 입문과 StateGraph) |

---

## 🎯 학습 목표

- `TypedDict`로 그래프 상태를 올바르게 정의할 수 있습니다.
- 노드가 부분 업데이트(partial update)를 반환하는 방식을 이해합니다.
- 기본 덮어쓰기(overwrite)와 리듀서(reducer)의 차이를 설명할 수 있습니다.
- `Annotated[list, add_messages]`를 사용해 메시지 목록을 올바르게 누적합니다.
- 커스텀 리듀서를 작성하고 활용할 수 있습니다.

---

## 📚 핵심 개념

### 상태(State)란?

LangGraph에서 상태는 그래프 실행 중 **모든 노드가 공유하는 데이터 저장소**입니다. 각 노드는 상태를 읽어서 처리하고, 변경이 필요한 필드만 딕셔너리로 반환합니다. LangGraph 런타임이 이를 받아 현재 상태에 **병합(merge)**합니다.

```
노드 실행 흐름:
┌──────────┐    state 전달    ┌──────────┐    부분 업데이트 반환    ┌──────────────┐
│  현재     │ ──────────────► │  node    │ ──────────────────────► │  새로운 상태  │
│  state   │                 │  함수    │   {"field": new_value}   │  (병합 결과)  │
└──────────┘                 └──────────┘                          └──────────────┘
```

### TypedDict로 상태 정의

```python
from typing import TypedDict

class MyState(TypedDict):
    # 기본 타입 필드
    name: str
    count: int
    active: bool
    # 복합 타입 필드
    items: list[str]
    metadata: dict[str, str]
```

**TypedDict 사용 이유**:
1. 타입 힌트로 IDE 자동완성 지원
2. LangGraph 런타임이 상태 스키마를 파악할 수 있음
3. 코드 가독성 향상

### 기본 동작: 덮어쓰기(Overwrite)

리듀서를 지정하지 않으면 LangGraph는 **새 값으로 완전히 교체(overwrite)**합니다.

```python
class CounterState(TypedDict):
    count: int
    message: str

def increment_node(state: CounterState) -> dict:
    return {"count": state["count"] + 1}
    # "message"는 반환하지 않으므로 기존 값 유지
    # "count"는 새 값으로 교체됨
```

덮어쓰기는 간단하지만, **목록(list)** 필드에서 문제가 생깁니다.

```python
class BadState(TypedDict):
    items: list[str]   # 리듀서 없음 → 덮어쓰기

def add_item_node(state: BadState) -> dict:
    # ❌ 이 방식은 매번 새 목록으로 교체됨
    return {"items": ["new_item"]}  # 기존 항목 모두 사라짐!
```

### 리듀서(Reducer): 업데이트 방식을 커스터마이즈

리듀서는 **현재 값과 새 값을 받아서 최종 값을 결정하는 함수**입니다. `Annotated` 타입 힌트로 필드에 붙입니다.

```python
from typing import Annotated

def my_reducer(current_value, new_value):
    # current_value: 현재 상태의 값
    # new_value: 노드가 반환한 새 값
    # 반환값: 업데이트 후 상태에 저장될 최종 값
    return current_value + new_value  # 예: 숫자 누적

class AccumulatedState(TypedDict):
    total: Annotated[int, my_reducer]  # 리듀서 적용
    label: str                         # 리듀서 없음 → 덮어쓰기
```

### add_messages: 메시지 목록 전용 리듀서

`add_messages`는 LangGraph가 제공하는 내장 리듀서로, 대화 메시지 목록을 안전하게 관리합니다.

```python
from langgraph.graph.message import add_messages
from typing import Annotated

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
```

`add_messages`의 동작 방식:
- 노드가 새 메시지를 반환하면 기존 목록에 **추가(append)**합니다.
- 동일한 `id`를 가진 메시지가 반환되면 **업데이트(update)**합니다.
- `HumanMessage`, `AIMessage`, `SystemMessage` 등 LangChain 메시지 타입을 모두 지원합니다.

```
add_messages 동작:
기존: [HumanMessage("안녕"), AIMessage("반갑습니다")]
새 값: AIMessage("어떻게 도와드릴까요?")
결과: [HumanMessage("안녕"), AIMessage("반갑습니다"), AIMessage("어떻게 도와드릴까요?")]
```

---

## 💻 코드 예제

### 예제 1: 기본 덮어쓰기 vs 리듀서 비교

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END


# --- 덮어쓰기 방식 (리듀서 없음) ---
class OverwriteState(TypedDict):
    history: list[str]  # 매번 교체됨


def step1(state: OverwriteState) -> dict:
    return {"history": ["step1"]}  # 기존 목록 교체


def step2(state: OverwriteState) -> dict:
    return {"history": ["step2"]}  # 또 교체


builder = StateGraph(OverwriteState)
builder.add_node("step1", step1)
builder.add_node("step2", step2)
builder.add_edge(START, "step1")
builder.add_edge("step1", "step2")
builder.add_edge("step2", END)
graph_overwrite = builder.compile()

result = graph_overwrite.invoke({"history": []})
print(f"덮어쓰기 결과: {result['history']}")
# 출력: ['step2']  ← step1 결과가 사라짐!


# --- 리듀서 방식 (누적) ---
def list_append(current: list, new: list) -> list:
    """두 목록을 이어붙이는 커스텀 리듀서."""
    return current + new


class ReducerState(TypedDict):
    history: Annotated[list[str], list_append]  # 리듀서 적용


def step1_r(state: ReducerState) -> dict:
    return {"history": ["step1"]}  # 기존 목록에 추가됨


def step2_r(state: ReducerState) -> dict:
    return {"history": ["step2"]}  # 또 추가됨


builder2 = StateGraph(ReducerState)
builder2.add_node("step1", step1_r)
builder2.add_node("step2", step2_r)
builder2.add_edge(START, "step1")
builder2.add_edge("step1", "step2")
builder2.add_edge("step2", END)
graph_reducer = builder2.compile()

result2 = graph_reducer.invoke({"history": []})
print(f"리듀서 결과: {result2['history']}")
# 출력: ['step1', 'step2']  ← 두 단계 모두 누적됨
```

### 예제 2: add_messages를 활용한 멀티턴 대화

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI


class ConversationState(TypedDict):
    messages: Annotated[list, add_messages]
    turn_count: int   # 덮어쓰기 (리듀서 없음)


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SYSTEM_PROMPT = SystemMessage(content="당신은 친절한 AI 어시스턴트입니다.")


def chat_node(state: ConversationState) -> dict:
    """LLM에 전체 대화 히스토리를 전달해 응답 생성."""
    # 시스템 메시지 + 기존 대화 전체
    messages_with_system = [SYSTEM_PROMPT] + state["messages"]
    response = llm.invoke(messages_with_system)
    return {
        "messages": response,              # add_messages가 누적
        "turn_count": state["turn_count"] + 1,  # 단순 덮어쓰기
    }


builder = StateGraph(ConversationState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
graph = builder.compile()

# 첫 번째 대화 턴
state = graph.invoke({
    "messages": [HumanMessage(content="파이썬이란 무엇인가요?")],
    "turn_count": 0,
})
print(f"Turn {state['turn_count']}: {state['messages'][-1].content[:100]}...")

# 두 번째 대화 턴 (이전 대화 계속)
state = graph.invoke({
    "messages": state["messages"] + [HumanMessage(content="장점은?")],
    "turn_count": state["turn_count"],
})
print(f"Turn {state['turn_count']}: {state['messages'][-1].content[:100]}...")
```

### 예제 3: 커스텀 리듀서 다양하게 만들기

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END


# 커스텀 리듀서 1: 최댓값 유지
def max_reducer(current: int, new: int) -> int:
    """항상 더 큰 값을 유지합니다."""
    return max(current, new)


# 커스텀 리듀서 2: 집합 합집합
def set_union(current: set, new: set) -> set:
    """두 집합의 합집합을 반환합니다."""
    return current | new


# 커스텀 리듀서 3: 딕셔너리 병합 (새 키가 기존을 덮어쓰지 않음)
def safe_merge(current: dict, new: dict) -> dict:
    """기존 키를 보존하고 새 키만 추가합니다."""
    return {**new, **current}  # current가 우선순위 높음


class MultiReducerState(TypedDict):
    high_score: Annotated[int, max_reducer]
    visited: Annotated[set, set_union]
    metadata: Annotated[dict, safe_merge]
    name: str  # 리듀서 없음 → 덮어쓰기


def node_a(state: MultiReducerState) -> dict:
    return {
        "high_score": 150,
        "visited": {"page_a"},
        "metadata": {"source": "node_a", "version": "1"},
    }


def node_b(state: MultiReducerState) -> dict:
    return {
        "high_score": 200,
        "visited": {"page_b"},
        "metadata": {"source": "node_b", "extra": "data"},
    }


builder = StateGraph(MultiReducerState)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)
graph = builder.compile()

result = graph.invoke({
    "high_score": 100,
    "visited": {"page_home"},
    "metadata": {"app": "my_app"},
    "name": "test",
})

print(f"high_score: {result['high_score']}")   # 200 (최댓값)
print(f"visited: {result['visited']}")          # {'page_home', 'page_a', 'page_b'}
print(f"metadata: {result['metadata']}")        # {'source': 'node_a', 'app': 'my_app', 'version': '1'}
# safe_merge는 current(node_a 결과)를 new(node_b 결과)보다 우선함
```

### 예제 4: 여러 키를 가진 현실적인 상태 설계

```python
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """에이전트 워크플로우를 위한 종합 상태."""

    # 대화 메시지 누적 (add_messages 리듀서)
    messages: Annotated[list[BaseMessage], add_messages]

    # 단순 값 (덮어쓰기)
    user_id: str
    session_id: str
    current_step: str

    # 오류 추적 (덮어쓰기, None 가능)
    last_error: Optional[str]

    # 수집된 데이터 (커스텀 리듀서로 누적)
    retrieved_docs: Annotated[list[str], lambda a, b: a + b]

    # 카운터 (커스텀 리듀서)
    retry_count: Annotated[int, lambda a, b: a + b]
```

---

## ✏️ 실습 과제

### 과제 1: 점수 집계 시스템

다음 요구사항의 StateGraph를 구현하세요.

- 상태:
  - `scores: Annotated[list[int], ...]` — 점수들을 누적하는 리듀서 포함
  - `max_score: Annotated[int, ...]` — 최대 점수만 유지하는 리듀서
  - `total_rounds: int` — 단순 덮어쓰기
- `round1_node`, `round2_node`, `round3_node`: 각각 다른 점수를 반환
- 최종 상태에서 모든 점수 목록과 최고 점수 출력

### 과제 2: add_messages 동작 실험

`add_messages` 리듀서가 있는 상태에서:
1. 같은 `id`를 가진 메시지를 두 번 반환하면 어떻게 되는지 확인하세요.
2. `AIMessage`에 `id="msg-001"`을 지정하고, 두 번째 노드에서 같은 `id`로 내용이 다른 `AIMessage`를 반환해 보세요.

### 과제 3 (도전): 딕셔너리 깊은 병합 리듀서

중첩 딕셔너리를 재귀적으로 병합하는 리듀서를 작성하세요.
```python
# 기대 동작:
current = {"a": {"x": 1, "y": 2}, "b": 3}
new = {"a": {"z": 4}, "c": 5}
result = deep_merge(current, new)
# 결과: {"a": {"x": 1, "y": 2, "z": 4}, "b": 3, "c": 5}
```

---

## ⚠️ 흔한 함정

### 함정 1: 리스트 필드에 리듀서 없이 추가하려는 실수

```python
# ❌ 잘못된 예: 이 방식은 기존 목록을 덮어씁니다
class BadState(TypedDict):
    items: list[str]  # 리듀서 없음!

def add_node(state: BadState) -> dict:
    # 기존 항목 읽어서 더하려는 의도
    return {"items": state["items"] + ["new"]}  # 매번 전체 목록 전달 필요

# ✓ 올바른 예: 리듀서 사용
def append_reducer(a: list, b: list) -> list:
    return a + b

class GoodState(TypedDict):
    items: Annotated[list[str], append_reducer]

def add_node_good(state: GoodState) -> dict:
    return {"items": ["new"]}  # 리듀서가 자동으로 붙여줌
```

### 함정 2: 리듀서가 None을 처리하지 못하는 문제

```python
# ❌ 문제: 초기값이 None일 때 오류
def bad_reducer(a: list, b: list) -> list:
    return a + b  # a가 None이면 TypeError!

# ✓ 해결: None 처리 추가
def safe_reducer(a: list | None, b: list | None) -> list:
    result_a = a or []
    result_b = b or []
    return result_a + result_b
```

### 함정 3: add_messages에 단일 메시지 대신 목록 반환 혼용

```python
from langchain_core.messages import AIMessage

# ✓ 둘 다 동작합니다 — add_messages는 단일/목록 모두 처리
return {"messages": AIMessage(content="...")}        # 단일 메시지
return {"messages": [AIMessage(content="...")]}      # 목록 형태
```

### 함정 4: 리듀서 함수를 람다로 간단히 쓸 때 직렬화 주의

```python
# ⚠️ 주의: 람다는 체크포인터 직렬화에 문제가 생길 수 있음
class State(TypedDict):
    count: Annotated[int, lambda a, b: a + b]  # 직렬화 불가능할 수 있음

# ✓ 명명 함수 사용 권장 (직렬화 안전)
def add_counts(a: int, b: int) -> int:
    return a + b

class SafeState(TypedDict):
    count: Annotated[int, add_counts]
```

---

## ✅ 셀프 체크

- [ ] `TypedDict`로 상태를 정의하고 여러 타입의 필드를 선언할 수 있다.
- [ ] 리듀서 없이 노드가 부분 업데이트를 반환하면 해당 필드만 덮어쓰임을 이해한다.
- [ ] `Annotated[타입, 리듀서함수]` 문법을 올바르게 사용할 수 있다.
- [ ] `add_messages`가 메시지를 어떻게 누적하고 업데이트하는지 설명할 수 있다.
- [ ] 커스텀 리듀서(최댓값, 집합 합집합, 딕셔너리 병합 등)를 직접 작성할 수 있다.
- [ ] 리스트 필드에 리듀서 없이 항목을 추가하면 어떤 문제가 생기는지 안다.

---

## 🔗 참고 자료

- [LangGraph - State Management](https://langchain-ai.github.io/langgraph/concepts/low_level/#state)
- [LangGraph - Reducers](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers)
- [LangGraph - add_messages API](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.message.add_messages)
- [Python typing - Annotated](https://docs.python.org/3/library/typing.html#typing.Annotated)

> **참고**: LangGraph API는 빠르게 발전하고 있습니다. 코드 실행 전 반드시 최신 공식 문서를 확인하세요.

---

## 네비게이션

| 이전 | 다음 |
|------|------|
| [Phase 18: LangGraph 입문과 StateGraph](./18-langgraph-intro-stategraph.md) | [Phase 20: 노드·엣지·라우팅](./20-nodes-edges-routing.md) |
