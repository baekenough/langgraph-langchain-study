# Phase 23: 스트리밍

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 90분 |
| 난이도 | ★★★☆☆ |
| 선행 학습 | Phase 22 (영속성과 체크포인터) |

---

## 🎯 학습 목표

- `stream_mode` 옵션(`values`, `updates`, `messages`, `custom`, `debug`)의 차이를 설명할 수 있습니다.
- `graph.stream()`과 `graph.astream()`으로 실시간 결과를 받을 수 있습니다.
- LLM 토큰 단위 스트리밍을 구현할 수 있습니다.
- `get_stream_writer()`로 노드 내부에서 커스텀 데이터를 스트리밍할 수 있습니다.
- 여러 `stream_mode`를 동시에 사용하는 멀티 모드 스트리밍을 이해합니다.

---

## 📚 핵심 개념

### 왜 스트리밍인가?

`graph.invoke()`는 그래프 실행이 **전부 완료된 후** 최종 결과를 반환합니다. LLM 호출이 여러 단계 있는 에이전트라면 사용자는 수 초씩 빈 화면을 바라봐야 합니다.

스트리밍은 이 문제를 해결합니다:

```
invoke() 방식:          stream() 방식:
  [실행 중...]            노드1 완료 → 즉시 전달
  [실행 중...]            LLM 토큰 → 즉시 전달
  [실행 중...]            노드2 완료 → 즉시 전달
  ↓ 결과 (5초 후)        ↓ 계속 흘러옴 (실시간)
```

### stream_mode 다섯 가지

| 모드 | 방출 시점 | 데이터 형태 | 주요 용도 |
|------|----------|------------|----------|
| `values` | 노드 실행 후 | 전체 상태 딕셔너리 | 각 단계별 전체 상태 확인 |
| `updates` | 노드 실행 후 | `{노드명: 변경된 키만}` | 변경분만 추적 |
| `messages` | LLM 토큰 생성 시 | `(AIMessageChunk, 메타데이터)` | 채팅 UI 실시간 표시 |
| `custom` | `get_stream_writer()` 호출 시 | 노드가 직접 emit한 값 | 커스텀 진행 상황 |
| `debug` | 모든 이벤트 | 상세 이벤트 딕셔너리 | 디버깅 |

### values 모드: 단계별 전체 상태

```
graph.stream(input, stream_mode="values") 출력:
  chunk_1 = {"count": 0, "messages": [HumanMessage(...)]}        ← 초기 상태
  chunk_2 = {"count": 1, "messages": [HumanMessage(...), AIMessage(...)]}   ← node_a 후
  chunk_3 = {"count": 2, "messages": [..., AIMessage(...)]}      ← node_b 후
```

### updates 모드: 변경분만

```
graph.stream(input, stream_mode="updates") 출력:
  chunk_1 = {"node_a": {"count": 1, "messages": [AIMessage(...)]}}
  chunk_2 = {"node_b": {"count": 2, "messages": [AIMessage(...)]}}
```

노드가 어떤 키를 업데이트했는지 추적할 때 유용합니다.

### messages 모드: LLM 토큰 스트리밍

`messages` 모드는 그래프 안의 LLM이 토큰을 생성할 때마다 즉시 방출합니다.

```
"안" → "녕" → "하" → "세" → "요" → ...
```

각 chunk는 `(AIMessageChunk, 메타데이터)` 튜플입니다. `AIMessageChunk.content`로 토큰을 읽습니다.

### custom 모드: 노드 내부에서 직접 방출

```python
from langgraph.config import get_stream_writer

def my_node(state):
    writer = get_stream_writer()  # 현재 컨텍스트에서 writer 획득
    writer({"step": "데이터 로딩 중..."})
    # ... 처리 ...
    writer({"step": "완료", "count": 42})
    return state
```

`stream_mode="custom"`으로 스트리밍하면 `writer()`에 전달한 값들이 순서대로 방출됩니다.

---

## 💻 코드 예제

### 예제 1: values vs updates 비교

```python
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# ─── 모델 설정 (OpenRouter) ───
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int


def node_a(state: State) -> dict:
    """첫 번째 처리 노드."""
    return {"step_count": state["step_count"] + 1}


def node_b(state: State) -> dict:
    """LLM 호출 노드."""
    response = llm.invoke(state["messages"])
    return {
        "messages": response,
        "step_count": state["step_count"] + 1,
    }


builder = StateGraph(State)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)
graph = builder.compile()

initial_state = {
    "messages": [HumanMessage(content="LangGraph가 무엇인가요? 한 문장으로.")],
    "step_count": 0,
}

# ── values 모드: 각 노드 후 전체 상태 출력 ──
print("=== stream_mode='values' ===")
for chunk in graph.stream(initial_state, stream_mode="values"):
    print(f"step_count={chunk['step_count']}, messages={len(chunk['messages'])}개")
    print()

# ── updates 모드: 변경된 키만 출력 ──
print("=== stream_mode='updates' ===")
for chunk in graph.stream(initial_state, stream_mode="updates"):
    # chunk: {"노드명": {변경된 키: 값}}
    for node_name, updated in chunk.items():
        print(f"[{node_name}] 업데이트: {list(updated.keys())}")
```

### 예제 2: messages 모드로 LLM 토큰 스트리밍

```python
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": response}


builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
graph = builder.compile()

# ── messages 모드: 토큰 단위 스트리밍 ──
print("AI: ", end="", flush=True)
for chunk, metadata in graph.stream(
    {"messages": [HumanMessage(content="파이썬의 장점을 3가지 알려주세요.")]},
    stream_mode="messages",
):
    # chunk는 AIMessageChunk (LLM 토큰) 또는 다른 메시지 타입
    if isinstance(chunk, AIMessageChunk) and chunk.content:
        print(chunk.content, end="", flush=True)

print()  # 줄바꿈
```

### 예제 3: custom 모드로 진행 상황 스트리밍

```python
import os
import time
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


class ResearchState(TypedDict):
    query: str
    findings: list[str]
    report: str


def search_node(state: ResearchState) -> dict:
    """검색 단계 — 진행 상황을 실시간으로 방출."""
    writer = get_stream_writer()

    topics = ["배경", "현황", "전망"]
    findings = []

    for i, topic in enumerate(topics, 1):
        writer({"event": "searching", "topic": topic, "progress": f"{i}/{len(topics)}"})
        time.sleep(0.1)  # 실제로는 API 호출
        findings.append(f"{topic}에 관한 검색 결과")

    writer({"event": "search_complete", "count": len(findings)})
    return {"findings": findings}


def report_node(state: ResearchState) -> dict:
    """보고서 작성 단계."""
    writer = get_stream_writer()
    writer({"event": "generating_report"})

    prompt = f"다음 내용으로 간단한 보고서를 작성하세요:\n" + "\n".join(state["findings"])
    response = llm.invoke(prompt)

    writer({"event": "report_complete"})
    return {"report": response.content}


builder = StateGraph(ResearchState)
builder.add_node("search", search_node)
builder.add_node("report", report_node)
builder.add_edge(START, "search")
builder.add_edge("search", "report")
builder.add_edge("report", END)
graph = builder.compile()

# ── custom 모드: 커스텀 이벤트만 수신 ──
print("=== 리서치 진행 상황 ===")
for event in graph.stream(
    {"query": "LangGraph 최신 동향", "findings": [], "report": ""},
    stream_mode="custom",
):
    print(f"[이벤트] {event}")
```

### 예제 4: 멀티 모드 스트리밍

여러 `stream_mode`를 리스트로 전달하면 모든 모드의 이벤트를 받을 수 있습니다.
각 chunk는 `(namespace, mode, data)` 튜플로 반환됩니다.

```python
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: State) -> dict:
    return {"messages": llm.invoke(state["messages"])}


builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
graph = builder.compile()

# ── 여러 모드 동시 사용 ──
for namespace, mode, chunk in graph.stream(
    {"messages": [HumanMessage(content="안녕하세요!")]},
    stream_mode=["updates", "messages"],  # 두 모드 동시
):
    if mode == "updates":
        print(f"[UPDATE] 노드 완료: {list(chunk.keys())}")
    elif mode == "messages":
        msg, meta = chunk
        if isinstance(msg, AIMessageChunk) and msg.content:
            print(f"[TOKEN] {msg.content}", end="", flush=True)

print()
```

### 예제 5: astream — 비동기 스트리밍

FastAPI 등 비동기 환경에서는 `astream()`을 사용합니다.

```python
import os
import asyncio
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: State) -> dict:
    return {"messages": llm.invoke(state["messages"])}


builder = StateGraph(State)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
graph = builder.compile()


async def stream_response(question: str) -> None:
    """비동기 스트리밍으로 응답 출력."""
    print("AI: ", end="", flush=True)
    async for chunk, _meta in graph.astream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            print(chunk.content, end="", flush=True)
    print()


# 실행
asyncio.run(stream_response("파이썬이 왜 인기 있나요?"))
```

---

## ✏️ 실습 과제

### 과제 1: 노드별 타임스탬프 추적기

`updates` 모드를 사용하여 각 노드가 완료된 시각과 소요 시간을 추적하는 코드를 작성하세요.

```python
import time

start = time.time()
for chunk in graph.stream(input_data, stream_mode="updates"):
    elapsed = time.time() - start
    for node_name in chunk:
        print(f"[{elapsed:.2f}s] {node_name} 완료")
```

### 과제 2: 스트리밍 채팅 CLI

다음 요구사항을 만족하는 CLI 채팅 프로그램을 작성하세요:
1. 사용자 입력을 받아 LLM에 전달
2. LLM 응답을 `messages` 모드로 토큰 단위 출력
3. `Ctrl+C`로 종료 가능
4. 멀티턴 대화 지원 (이전 내용 기억)

### 과제 3: 진행률 바

`custom` 모드와 `get_stream_writer()`를 활용하여 멀티 스텝 파이프라인에서 터미널 진행률 바를 구현하세요.

---

## ⚠️ 흔한 함정

### 1. messages 모드에서 chunk 타입 확인 누락

```python
# ❌ 잘못된 예: 모든 chunk에 .content 접근 시도
for chunk, meta in graph.stream(input, stream_mode="messages"):
    print(chunk.content)  # ToolMessage 등은 content가 다를 수 있음

# ✅ 올바른 예: 타입 확인 후 처리
from langchain_core.messages import AIMessageChunk
for chunk, meta in graph.stream(input, stream_mode="messages"):
    if isinstance(chunk, AIMessageChunk) and chunk.content:
        print(chunk.content, end="", flush=True)
```

### 2. invoke()와 stream()의 반환 타입 혼동

```python
# invoke(): 최종 상태 딕셔너리 반환
result = graph.invoke(input)
print(result["messages"])  # 직접 접근 가능

# stream(stream_mode="values"): 이터레이터 반환 — 마지막 값이 최종 상태
final_state = None
for chunk in graph.stream(input, stream_mode="values"):
    final_state = chunk
print(final_state["messages"])
```

### 3. get_stream_writer()를 stream_mode="custom" 없이 사용

`get_stream_writer()`로 방출한 데이터는 반드시 `stream_mode="custom"` (또는 멀티 모드에 `"custom"` 포함)이어야 수신됩니다. 다른 모드에서는 무시됩니다.

### 4. 동기 그래프에서 astream() 사용

`astream()`은 `async for`와 함께 `asyncio` 이벤트 루프 안에서만 동작합니다. 동기 환경에서는 `stream()`을 사용하세요.

---

## ✅ 셀프 체크

- [ ] `stream_mode="values"`와 `"updates"`의 출력 형태 차이를 설명할 수 있다.
- [ ] `messages` 모드로 LLM 토큰 단위 스트리밍을 구현할 수 있다.
- [ ] `get_stream_writer()`를 노드 안에서 호출하고 `"custom"` 모드로 수신할 수 있다.
- [ ] `astream()`을 `async for`로 올바르게 사용할 수 있다.
- [ ] 멀티 모드 스트리밍에서 `(namespace, mode, chunk)` 튜플을 처리할 수 있다.

---

## 🔗 참고 자료

- [LangGraph Streaming 공식 문서](https://langchain-ai.github.io/langgraph/concepts/streaming/)
- [stream_mode 상세 설명](https://langchain-ai.github.io/langgraph/how-tos/streaming/)
- [astream 비동기 가이드](https://langchain-ai.github.io/langgraph/how-tos/streaming-async/)

> **API 변동 주의**: `stream_mode="messages"` 반환 형태와 `get_stream_writer()` 임포트 경로는 LangGraph 버전에 따라 달라질 수 있습니다. 최신 변경사항은 [공식 CHANGELOG](https://github.com/langchain-ai/langgraph/blob/main/CHANGELOG.md)를 확인하세요.

---

⬅️ [Phase 22: 영속성과 체크포인터](./22-persistence-checkpointers.md) | ➡️ [Phase 24: Human-in-the-Loop](./24-human-in-the-loop.md)
