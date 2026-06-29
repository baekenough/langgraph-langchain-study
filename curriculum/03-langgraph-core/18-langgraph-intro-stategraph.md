# Phase 18: LangGraph 입문과 StateGraph

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 90분 |
| 난이도 | ★★☆☆☆ |
| 선행 학습 | Phase 17 (RAG 평가), LangChain 기초 (Phase 04~10) |

---

## 🎯 학습 목표

- 체인(Chain)이 아닌 그래프(Graph)가 필요한 이유를 설명할 수 있습니다.
- StateGraph의 멘탈 모델(노드·엣지·상태)을 이해합니다.
- 가장 단순한 StateGraph를 직접 만들고 실행할 수 있습니다.
- 그래프 구조를 시각화하는 방법을 알고 있습니다.

---

## 📚 핵심 개념

### 왜 체인이 아니라 그래프인가?

LangChain의 LCEL 체인은 **선형(linear)** 파이프라인입니다. `A | B | C` 형태로 데이터가 한 방향으로만 흐릅니다. 이는 단순한 RAG 파이프라인이나 단일 LLM 호출에는 충분하지만, 다음 상황에서 한계에 부딪힙니다.

| 필요한 기능 | LCEL 체인 | LangGraph |
|------------|----------|-----------|
| 순환 (루프, 재시도) | ✗ 불가 | ✓ 기본 지원 |
| 조건 분기 | △ 제한적 | ✓ 자유로운 라우팅 |
| 공유 상태 | ✗ 없음 | ✓ State 객체 |
| 실행 제어 (일시정지/재개) | ✗ 불가 | ✓ Checkpoint |
| 병렬 실행 | △ 제한적 | ✓ 팬아웃/팬인 |
| 인간 개입 (Human-in-the-loop) | ✗ 불가 | ✓ interrupt() |

**LangGraph의 핵심 설계 철학**: 복잡한 에이전트 워크플로우는 **상태를 가진 방향 그래프(stateful directed graph)** 로 표현할 때 가장 자연스럽습니다.

### 멘탈 모델: 세 가지 구성 요소

```
┌─────────────────────────────────────────────────────────┐
│                     StateGraph                           │
│                                                         │
│   State (공유 데이터)                                    │
│   ┌────────────────────────────────┐                    │
│   │  messages: [...]               │                    │
│   │  count: 0                      │                    │
│   │  result: ""                    │                    │
│   └────────────────────────────────┘                    │
│                                                         │
│   Node A ──────────────────────────► Node B             │
│   (함수)        엣지 (흐름)           (함수)             │
│     │                                                   │
│     └──── 조건부 엣지 ─────────────► Node C             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**노드(Node)**: 작업 단위. 파이썬 함수입니다. 현재 상태를 받아서 업데이트할 상태를 반환합니다.

```python
def my_node(state: MyState) -> dict:
    # state에서 데이터를 읽고
    # 처리 후 업데이트할 필드만 반환
    return {"result": "processed"}
```

**엣지(Edge)**: 노드 간 이동 경로. 단순 연결(일반 엣지)과 조건에 따른 분기(조건부 엣지)가 있습니다.

**상태(State)**: 그래프 전체에서 공유되는 데이터 구조. `TypedDict`로 정의합니다. 노드는 상태 전체를 받고, 변경할 키만 포함한 딕셔너리를 반환합니다.

### START와 END

`START`와 `END`는 LangGraph 내장 특수 노드입니다.

- `START`: 그래프 실행의 시작점. 여기서 첫 번째 노드로 엣지를 연결합니다.
- `END`: 그래프 실행의 종료점. 여기로 엣지가 이어지면 실행을 마칩니다.

---

## 💻 코드 예제

### 예제 1: 가장 단순한 StateGraph

```python
# 최신 LangGraph API 기준 (API는 빠르게 변하니 공식 문서도 확인하세요)
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# 1단계: 상태 정의
class SimpleState(TypedDict):
    message: str
    processed: bool


# 2단계: 노드 정의 (파이썬 함수)
def process_node(state: SimpleState) -> dict:
    """메시지를 처리하는 노드."""
    original = state["message"]
    updated_message = original.upper()  # 대문자 변환
    return {
        "message": updated_message,
        "processed": True,
    }


def summarize_node(state: SimpleState) -> dict:
    """결과를 요약하는 노드."""
    msg = state["message"]
    print(f"[summarize] 최종 메시지: {msg}, 처리됨: {state['processed']}")
    return {}  # 변경 없을 때는 빈 딕셔너리 반환


# 3단계: 그래프 빌드
builder = StateGraph(SimpleState)

# 노드 등록
builder.add_node("process", process_node)
builder.add_node("summarize", summarize_node)

# 엣지 연결
builder.add_edge(START, "process")       # 시작 → process
builder.add_edge("process", "summarize") # process → summarize
builder.add_edge("summarize", END)       # summarize → 종료

# 4단계: 컴파일
graph = builder.compile()

# 5단계: 실행
initial_state = {"message": "hello langgraph", "processed": False}
result = graph.invoke(initial_state)

print(f"결과: {result}")
# 출력:
# [summarize] 최종 메시지: HELLO LANGGRAPH, 처리됨: True
# 결과: {'message': 'HELLO LANGGRAPH', 'processed': True}
```

**그래프 구조 (텍스트 다이어그램)**:
```
START
  │
  ▼
process  (message를 대문자로 변환)
  │
  ▼
summarize  (결과 출력)
  │
  ▼
END
```

### 예제 2: 그래프 시각화

```python
# Mermaid 다이어그램으로 그래프 구조 출력
print(graph.get_graph().draw_mermaid())
```

출력 예시:
```
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
	__start__([<p>__start__</p>]):::first
	process([process])
	summarize([summarize])
	__end__([<p>__end__</p>]):::last
	__start__ --> process;
	process --> summarize;
	summarize --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0 stroke-opacity:0
	classDef last fill:#bfb6fc
```

```python
# ASCII 형태로 출력 (터미널 친화적)
graph.get_graph().print_ascii()
```

### 예제 3: LLM을 포함한 기본 대화 그래프

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI


# 메시지 목록을 누적하는 상태
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def chat_node(state: ChatState) -> dict:
    """LLM에 메시지를 보내고 응답을 받는 노드."""
    response = llm.invoke(state["messages"])
    return {"messages": response}  # add_messages가 기존 목록에 추가해 줌


# 그래프 구성
builder = StateGraph(ChatState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

graph = builder.compile()

# 실행
from langchain_core.messages import HumanMessage

result = graph.invoke({"messages": [HumanMessage(content="안녕하세요!")]})

for msg in result["messages"]:
    print(f"[{msg.type}] {msg.content}")
```

**그래프 구조**:
```
START
  │
  ▼
chat  (LLM 호출)
  │
  ▼
END
```

---

## ✏️ 실습 과제

### 과제 1: 두 노드 파이프라인

다음 조건을 만족하는 StateGraph를 만드세요.

- 상태: `{"number": int, "result": str}`
- `double_node`: `number`를 2배로 만드는 노드
- `describe_node`: "숫자 X의 2배는 Y입니다" 형식의 `result` 문자열 생성 노드
- 실행 순서: `START → double_node → describe_node → END`

### 과제 2: 시각화 확인

과제 1에서 만든 그래프의 Mermaid 다이어그램을 출력하고, 어떤 노드와 엣지가 표시되는지 확인하세요.

### 과제 3 (도전): 세 노드 파이프라인

- `fetch_node`: 외부 데이터를 가져오는 역할을 시뮬레이션 (고정된 텍스트 반환)
- `clean_node`: 텍스트에서 특수문자를 제거
- `count_node`: 단어 수를 셈

세 노드를 연결하고 최종 상태를 출력하세요.

---

## ⚠️ 흔한 함정

### 함정 1: 상태 전체를 반환하려는 실수

```python
# ❌ 잘못된 예: 변경이 없는 필드도 모두 반환
def my_node(state: MyState) -> MyState:
    return {
        "message": state["message"].upper(),
        "processed": state["processed"],  # 변경 없는데 굳이 포함
        "count": state["count"],          # 변경 없는데 굳이 포함
    }

# ✓ 올바른 예: 변경할 필드만 반환
def my_node(state: MyState) -> dict:
    return {"message": state["message"].upper()}
```

### 함정 2: START/END를 임포트하지 않는 실수

```python
# ❌ 잘못된 예: 문자열 "__start__"를 직접 사용
builder.add_edge("__start__", "my_node")

# ✓ 올바른 예: 상수 임포트해서 사용
from langgraph.graph import StateGraph, START, END
builder.add_edge(START, "my_node")
```

### 함정 3: compile() 전에 invoke() 호출

```python
# ❌ 잘못된 예
builder = StateGraph(MyState)
builder.add_node("node", my_func)
# compile()을 잊고 바로 실행하면 AttributeError
result = builder.invoke({"key": "value"})  # 오류!

# ✓ 올바른 예
graph = builder.compile()  # 반드시 compile() 먼저
result = graph.invoke({"key": "value"})
```

### 함정 4: TypedDict 타입과 실제 값 불일치

```python
class MyState(TypedDict):
    count: int

# ❌ 잘못된 예: 문자열을 int 필드에 할당
graph.invoke({"count": "0"})  # 런타임 오류 가능성

# ✓ 올바른 예
graph.invoke({"count": 0})
```

---

## ✅ 셀프 체크

- [ ] 선형 체인과 그래프의 차이점을 세 가지 이상 말할 수 있다.
- [ ] `StateGraph(State)`, `add_node()`, `add_edge()`, `compile()`, `invoke()`의 역할을 각각 설명할 수 있다.
- [ ] 노드 함수의 시그니처 `(state: MyState) -> dict`를 이해하고, 변경할 필드만 반환해야 함을 안다.
- [ ] `START`와 `END`의 역할을 알고, 올바르게 임포트할 수 있다.
- [ ] `get_graph().draw_mermaid()`로 그래프를 시각화할 수 있다.
- [ ] `Annotated[list, add_messages]`가 무엇을 하는지 기본적으로 설명할 수 있다 (자세한 내용은 Phase 19).

---

## 🔗 참고 자료

- [LangGraph 공식 문서 - Quick Start](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [LangGraph - StateGraph API](https://langchain-ai.github.io/langgraph/reference/graphs/)
- [LangGraph - Concepts: Graph](https://langchain-ai.github.io/langgraph/concepts/low_level/)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)

> **참고**: LangGraph API는 빠르게 발전하고 있습니다. 코드 실행 전 반드시 최신 공식 문서를 확인하세요.

---

## 네비게이션

| 이전 | 다음 |
|------|------|
| [Phase 17: RAG 평가](../02-rag/17-rag-evaluation.md) | [Phase 19: 상태와 리듀서](./19-state-reducers.md) |
