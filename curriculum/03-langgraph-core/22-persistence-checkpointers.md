# Phase 22: 영속성과 체크포인터

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 120분 |
| 난이도 | ★★★★☆ |
| 선행 학습 | Phase 21 (순환과 반복) |

---

## 🎯 학습 목표

- 체크포인터(checkpointer)의 개념과 역할을 설명할 수 있습니다.
- `MemorySaver`와 `SqliteSaver`를 상황에 맞게 선택해 사용합니다.
- `thread_id`로 독립적인 대화 세션을 관리합니다.
- `get_state()` / `get_state_history()`로 실행 이력을 조회합니다.
- 중단된 그래프를 재개하는 패턴을 구현합니다.
- 멀티턴 대화를 체크포인터로 구현합니다.

---

## 📚 핵심 개념

### 체크포인터란?

체크포인터는 **그래프 실행의 각 스텝 후 상태를 저장**하는 메커니즘입니다. 이를 통해:

1. **멀티턴 대화**: 이전 대화 내용을 기억
2. **중단 후 재개**: 실행 중간에 멈췄다가 다시 시작
3. **Human-in-the-loop**: 사람의 승인을 기다리는 동안 상태 보존
4. **Time travel**: 과거 특정 시점의 상태로 되돌아가기

```
그래프 실행 흐름:
invoke() 호출
    │
    ▼
node_1 실행 ─► 상태 저장 (checkpoint_1)
    │
    ▼
node_2 실행 ─► 상태 저장 (checkpoint_2)
    │
    ▼
node_3 실행 ─► 상태 저장 (checkpoint_3)
    │
    ▼
결과 반환
```

### thread_id: 대화 세션 구분

체크포인터는 `thread_id`로 여러 독립적인 대화 세션을 구분합니다.

```python
# 사용자 A의 대화
config_a = {"configurable": {"thread_id": "user_a_session_1"}}
graph.invoke(state, config=config_a)

# 사용자 B의 대화 (A와 완전히 독립)
config_b = {"configurable": {"thread_id": "user_b_session_1"}}
graph.invoke(state, config=config_b)
```

`thread_id`는 임의의 문자열로, 보통 사용자 ID + 세션 ID 조합을 사용합니다.

### MemorySaver vs SqliteSaver

| 항목 | MemorySaver | SqliteSaver |
|------|-------------|-------------|
| 저장 위치 | 메모리 내 | SQLite 파일 |
| 프로세스 재시작 | 데이터 사라짐 | 유지됨 |
| 적합한 용도 | 개발/테스트 | 프로덕션/실제 서비스 |
| 설정 복잡도 | 매우 간단 | 간단 (경로 지정) |

```python
# MemorySaver: 메모리에 저장 (프로세스 종료 시 사라짐)
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

# SqliteSaver: SQLite 파일에 저장 (영구 보존)
from langgraph.checkpoint.sqlite import SqliteSaver
with SqliteSaver.from_conn_string("./checkpoints.db") as saver:
    graph = builder.compile(checkpointer=saver)
    ...
```

### 상태 조회: get_state와 get_state_history

```python
# 현재 (최신) 상태 조회
state_snapshot = graph.get_state(config)
print(state_snapshot.values)   # 상태 딕셔너리
print(state_snapshot.next)     # 다음 실행될 노드 (있다면)
print(state_snapshot.config)   # 현재 config (checkpoint_id 포함)

# 전체 이력 조회 (최신순)
for snapshot in graph.get_state_history(config):
    print(snapshot.config["configurable"]["checkpoint_id"])
    print(snapshot.values)
```

---

## 💻 코드 예제

### 예제 1: MemorySaver로 멀티턴 대화

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def chat_node(state: ConversationState) -> dict:
    """전체 대화 히스토리를 LLM에 전달."""
    system = SystemMessage(content="당신은 도움이 되는 AI 어시스턴트입니다.")
    response = llm.invoke([system] + state["messages"])
    return {"messages": response}


# 그래프 빌드
builder = StateGraph(ConversationState)
builder.add_node("chat", chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# 체크포인터와 함께 컴파일
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 동일한 thread_id로 여러 번 invoke → 대화 이어짐
config = {"configurable": {"thread_id": "user-001"}}

# 첫 번째 턴
graph.invoke(
    {"messages": [HumanMessage(content="제 이름은 홍길동입니다.")]},
    config=config,
)

# 두 번째 턴 (이전 대화 자동으로 기억)
result = graph.invoke(
    {"messages": [HumanMessage(content="제 이름이 뭐라고 했죠?")]},
    config=config,
)
print(result["messages"][-1].content)  # "홍길동"을 기억

# 상태 확인
state = graph.get_state(config)
print(f"총 메시지 수: {len(state.values['messages'])}")

# 다른 사용자는 독립적인 대화
config_other = {"configurable": {"thread_id": "user-002"}}
result2 = graph.invoke(
    {"messages": [HumanMessage(content="제 이름이 뭐죠?")]},
    config=config_other,
)
print(result2["messages"][-1].content)  # 이름을 모름 (독립 세션)
```

### 예제 2: SqliteSaver로 영구 보존 대화

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage

# SqliteSaver는 컨텍스트 매니저로 사용
with SqliteSaver.from_conn_string("./my_conversations.db") as saver:
    graph = builder.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "persistent-001"}}

    # 첫 번째 실행
    graph.invoke(
        {"messages": [HumanMessage(content="파이썬 제너레이터를 설명해줘")]},
        config=config,
    )
    print("첫 번째 질문 완료")

    # 나중에 (심지어 프로세스 재시작 후에도) 같은 thread_id로 계속
    result = graph.invoke(
        {"messages": [HumanMessage(content="방금 설명한 내용 예제 보여줘")]},
        config=config,
    )
    print(result["messages"][-1].content)
```

### 예제 3: get_state_history로 이력 탐색

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage


# 이전 예제와 동일한 그래프 사용
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "history-demo"}}

# 여러 번 대화
for question in [
    "1 + 1은?",
    "그 두 배는?",
    "그 세 배는?",
]:
    graph.invoke({"messages": [HumanMessage(content=question)]}, config=config)

# 실행 이력 조회
print("=== 대화 이력 ===")
history = list(graph.get_state_history(config))

for i, snapshot in enumerate(reversed(history)):  # 오래된 것부터
    checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
    msg_count = len(snapshot.values.get("messages", []))
    next_node = snapshot.next
    print(f"[스텝 {i}] checkpoint: {checkpoint_id[:12]}... | 메시지: {msg_count}개 | 다음: {next_node}")

# 특정 시점으로 되돌아가서 재개
# 두 번째 대화 이후 시점의 checkpoint_id 가져오기
target_snapshot = list(reversed(history))[2]  # 두 번째 대화 후
target_config = target_snapshot.config

print(f"\n두 번째 대화 후 상태에서 재개:")
print(f"메시지 수: {len(target_snapshot.values['messages'])}")

# 그 시점에서 새로운 방향으로 분기
branch_config = target_config.copy()
# thread_id를 새것으로 바꿔 새 스레드로 분기 (time travel)
branch_config["configurable"]["thread_id"] = "branch-from-step2"

result = graph.invoke(
    {"messages": [HumanMessage(content="그 절반은?")]},
    config={
        "configurable": {
            "thread_id": "branch-from-step2",
            "checkpoint_id": target_snapshot.config["configurable"]["checkpoint_id"],
        }
    }
)
print(f"분기 결과: {result['messages'][-1].content}")
```

### 예제 4: 중단 후 재개 패턴

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
import time


class LongTaskState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step: int
    data: str


def step_1(state: LongTaskState) -> dict:
    print("[단계 1] 데이터 수집 중...")
    time.sleep(0.1)  # 실제로는 오래 걸리는 작업
    return {"step": 1, "data": "수집된 데이터"}


def step_2(state: LongTaskState) -> dict:
    print("[단계 2] 데이터 처리 중...")
    time.sleep(0.1)
    return {"step": 2, "data": f"{state['data']} → 처리됨"}


def step_3(state: LongTaskState) -> dict:
    print("[단계 3] 결과 저장 중...")
    return {"step": 3, "data": f"{state['data']} → 저장됨"}


memory = MemorySaver()
builder = StateGraph(LongTaskState)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# interrupt_before로 특정 노드 전에 일시 정지
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["step_3"],  # step_3 직전에 멈춤
)

config = {"configurable": {"thread_id": "task-001"}}

# 첫 실행: step_1, step_2까지만 실행 후 일시정지
result = graph.invoke(
    {"messages": [], "step": 0, "data": ""},
    config=config,
)
print(f"일시정지됨. 현재 데이터: {result['data']}")

# 현재 상태 확인
current_state = graph.get_state(config)
print(f"다음 실행 예정 노드: {current_state.next}")  # ('step_3',)

# (여기서 사람이 검토하거나 추가 작업 수행)
print("\n[사람 검토 중... 승인!]\n")

# step_3 실행을 재개 (None을 입력으로 전달)
final_result = graph.invoke(None, config=config)
print(f"최종 데이터: {final_result['data']}")
```

---

## ✏️ 실습 과제

### 과제 1: 개인화 챗봇

`MemorySaver`를 사용해서:
1. 사용자 이름과 선호도를 기억하는 챗봇을 만드세요.
2. `thread_id`를 `f"user_{user_id}"` 형식으로 사용하세요.
3. 세 번의 대화 후 `get_state()`로 메시지 목록 수를 출력하세요.

### 과제 2: 이력 탐색

위 과제 1의 챗봇에서:
1. `get_state_history()`로 전체 이력을 출력하세요.
2. 두 번째 대화 시점의 `checkpoint_id`를 찾으세요.
3. 그 시점으로 돌아가 다른 질문을 해보세요.

### 과제 3 (도전): SqliteSaver 영구 대화

`SqliteSaver`를 사용해서:
1. 프로그램을 처음 실행하면 새 대화 시작
2. 두 번째 실행에서 같은 `thread_id`로 이어서 대화
3. `get_state()`로 이전 대화가 기억되는지 확인

---

## ⚠️ 흔한 함정

### 함정 1: config 없이 invoke() 호출

```python
# ❌ 잘못된 예: checkpointer가 있어도 config 없으면 저장 안 됨
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
result = graph.invoke(state)  # thread_id 없음 → 저장 안 됨

# ✓ 올바른 예
result = graph.invoke(state, config={"configurable": {"thread_id": "my-thread"}})
```

### 함정 2: SqliteSaver를 컨텍스트 매니저 없이 사용

```python
# ❌ 잘못된 예: 연결이 제대로 관리되지 않음
saver = SqliteSaver.from_conn_string("./db.sqlite")
graph = builder.compile(checkpointer=saver)

# ✓ 올바른 예: with 문으로 연결 관리
with SqliteSaver.from_conn_string("./db.sqlite") as saver:
    graph = builder.compile(checkpointer=saver)
    graph.invoke(state, config=config)
```

### 함정 3: 재개 시 초기 상태 재전달

```python
# ❌ 잘못된 예: interrupt_before 이후 재개할 때 초기 state 재전달
graph.invoke(
    {"messages": [], "step": 0},  # 이미 저장된 상태를 덮어씀!
    config=config,
)

# ✓ 올바른 예: None을 전달해 저장된 상태에서 재개
graph.invoke(None, config=config)
```

### 함정 4: thread_id 없이 여러 사용자 대화 혼용

```python
# ❌ 잘못된 예: 모든 사용자가 같은 스레드 공유
SHARED_CONFIG = {"configurable": {"thread_id": "shared"}}

# 사용자 A, B, C 모두 같은 config 사용 → 대화 섞임!

# ✓ 올바른 예: 사용자별 고유 thread_id
def get_config(user_id: str) -> dict:
    return {"configurable": {"thread_id": f"user_{user_id}"}}
```

### 함정 5: 체크포인터 없이 interrupt_before 설정

```python
# ❌ 잘못된 예: checkpointer 없으면 interrupt_before 무의미
graph = builder.compile(interrupt_before=["my_node"])  # checkpointer 없음!
# interrupt는 발생하지 않거나 재개 불가

# ✓ 올바른 예
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["my_node"],
)
```

---

## ✅ 셀프 체크

- [ ] 체크포인터가 무엇이고 왜 필요한지 세 가지 이상 이유를 말할 수 있다.
- [ ] `MemorySaver`와 `SqliteSaver`의 차이와 적합한 사용 시나리오를 안다.
- [ ] `config={"configurable": {"thread_id": "..."}}` 형식으로 스레드를 구분할 수 있다.
- [ ] `get_state(config)`로 현재 상태를 조회하고 `.values`, `.next` 필드를 해석할 수 있다.
- [ ] `get_state_history(config)`로 과거 체크포인트를 순회할 수 있다.
- [ ] `interrupt_before`로 일시정지 후 `graph.invoke(None, config=config)`로 재개하는 패턴을 구현할 수 있다.
- [ ] 멀티턴 대화에서 새 메시지만 전달하면 이전 메시지는 체크포인터가 자동으로 합쳐줌을 이해한다.

---

## 🔗 참고 자료

- [LangGraph - Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [LangGraph - MemorySaver](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.memory.MemorySaver)
- [LangGraph - SqliteSaver](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.sqlite.SqliteSaver)
- [LangGraph - How-to: Persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
- [LangGraph - Time Travel](https://langchain-ai.github.io/langgraph/how-tos/time-travel/)

> **참고**: LangGraph API는 빠르게 발전하고 있습니다. 코드 실행 전 반드시 최신 공식 문서를 확인하세요.

---

## 네비게이션

| 이전 | 다음 |
|------|------|
| [Phase 21: 순환과 반복](./21-cycles-iteration.md) | [Phase 23: 스트리밍](./23-streaming.md) |
