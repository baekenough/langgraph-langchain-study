# Phase 24: Human-in-the-Loop

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 120분 |
| 난이도 | ★★★★☆ |
| 선행 학습 | Phase 23 (스트리밍) |

---

## 🎯 학습 목표

- `interrupt()`로 그래프 실행을 일시정지하고 사람의 입력을 기다릴 수 있습니다.
- `Command(resume=...)`로 일시정지된 그래프를 재개할 수 있습니다.
- `interrupt_before` / `interrupt_after` 브레이크포인트를 설정할 수 있습니다.
- `update_state()`로 재개 전에 상태를 수정할 수 있습니다.
- 과거 체크포인트로 돌아가는 **Time Travel**을 구현할 수 있습니다.

---

## 📚 핵심 개념

### 왜 Human-in-the-Loop인가?

완전 자동화 에이전트는 강력하지만, 다음 상황에서는 사람의 판단이 필요합니다:

- 에이전트가 중요한 API를 호출하기 전 확인 (예: 이메일 발송, DB 삭제)
- 에이전트가 불확실한 상황에서 방향을 물어볼 때
- 출력 품질을 검토하고 수정할 때
- 규제/컴플라이언스 요구사항으로 사람 승인이 필수일 때

LangGraph는 이를 **일급 기능**으로 지원합니다. 그래프를 재설계하지 않고도 임의의 지점에서 실행을 멈출 수 있습니다.

### interrupt(): 노드 내부에서 멈추기

`interrupt()`는 노드 실행 도중 그래프를 일시정지합니다. 현재 상태가 체크포인터에 저장되고, `interrupt()`에 전달한 값이 외부로 반환됩니다.

```
graph.invoke() 호출
    │
    ▼
node_a 실행 ─── 완료 ───────────────── 상태 저장
    │
    ▼
review_node 실행
    │
    ▼
  interrupt("승인 여부를 알려주세요") ← 여기서 멈춤!
    │ (외부에 값 반환, 상태 저장)
    │
    ▼
  (사람이 검토 후 resume)
    │
    ▼
  resume 값이 interrupt() 반환값이 됨
    │
    ▼
node_b 실행 → 완료
```

```python
from langgraph.types import interrupt

def review_node(state):
    # interrupt()를 호출하면 실행이 여기서 멈춥니다.
    # 전달한 딕셔너리는 invoke()/stream()의 반환값에 포함됩니다.
    user_decision = interrupt({
        "question": "이 결과를 승인하시겠습니까?",
        "data": state["draft"],
    })
    # user_decision은 Command(resume=...)에 전달한 값
    return {"approved": user_decision == "yes"}
```

### Command(resume=...): 재개

`interrupt()`로 멈춘 그래프는 동일한 `thread_id`의 `config`와 함께 `Command(resume=값)`을 전달하여 재개합니다.

```python
from langgraph.types import Command

# 그래프 재개: interrupt()의 반환값으로 "yes"를 전달
graph.invoke(Command(resume="yes"), config=config)
```

재개 후 `interrupt()`는 `"yes"`를 반환하고 노드 실행이 이어집니다.

### interrupt_before / interrupt_after: 브레이크포인트

노드 코드를 수정하지 않고도 **컴파일 시점**에 브레이크포인트를 설정할 수 있습니다.

```python
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["dangerous_node"],  # 이 노드 실행 전에 일시정지
    interrupt_after=["review_node"],      # 이 노드 실행 후에 일시정지
)
```

브레이크포인트로 일시정지된 그래프는 `graph.get_state(config).next`로 다음 실행될 노드를 확인할 수 있습니다.

```python
state_snapshot = graph.get_state(config)
print(state_snapshot.next)  # ('dangerous_node',) — 아직 실행 전
```

### update_state(): 재개 전 상태 수정

일시정지 중에 상태를 수정한 뒤 재개할 수 있습니다.

```python
# 상태 수정 (as_node로 어떤 노드가 업데이트한 것처럼 처리)
graph.update_state(
    config,
    {"draft": "사람이 수정한 버전"},
    as_node="review_node",
)

# 수정된 상태로 재개
graph.invoke(None, config=config)
```

`as_node`를 지정하면 해당 노드가 update를 반환한 것으로 처리되어 리듀서가 정상 적용됩니다.

### Time Travel: 과거로 돌아가기

모든 체크포인트는 `checkpoint_id`를 가집니다. 특정 시점의 `config`를 사용하면 그 지점으로 "되돌아가서" 다른 경로로 실행할 수 있습니다.

```
실행 히스토리:
  checkpoint_0 (초기) → checkpoint_1 (node_a) → checkpoint_2 (node_b) → 완료

Time Travel:
  checkpoint_1로 되돌아가 다른 입력으로 재실행
  → checkpoint_1 → checkpoint_3 (다른 경로)
```

---

## 💻 코드 예제

### 예제 1: interrupt()로 사람 승인 받기

```python
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    draft: str
    approved: bool


def generate_node(state: State) -> dict:
    """초안을 생성하는 노드."""
    response = llm.invoke(
        state["messages"] + [HumanMessage(content="짧은 마케팅 이메일 초안을 작성해주세요.")]
    )
    return {"draft": response.content}


def human_review_node(state: State) -> dict:
    """사람이 초안을 검토하는 노드."""
    print(f"\n📧 생성된 초안:\n{state['draft']}\n")

    # 이 지점에서 그래프가 일시정지됩니다.
    decision = interrupt({
        "message": "위 초안을 승인하시겠습니까?",
        "options": ["yes (승인)", "no (거부)", "edit (수정 요청)"],
        "draft": state["draft"],
    })

    if decision == "yes":
        return {"approved": True}
    elif decision == "no":
        return {"approved": False, "draft": ""}
    else:
        # 수정 요청: decision에 수정 지시사항이 포함된다고 가정
        return {"approved": False, "draft": f"수정 요청: {decision}"}


def send_node(state: State) -> dict:
    """승인된 이메일을 발송하는 노드."""
    if state["approved"]:
        print("✅ 이메일 발송 완료!")
        return {"messages": AIMessage(content=f"이메일이 발송되었습니다:\n{state['draft']}")}
    else:
        print("❌ 이메일 발송 취소됨")
        return {"messages": AIMessage(content="이메일 발송이 취소되었습니다.")}


# 그래프 구성
builder = StateGraph(State)
builder.add_node("generate", generate_node)
builder.add_node("human_review", human_review_node)
builder.add_node("send", send_node)
builder.add_edge(START, "generate")
builder.add_edge("generate", "human_review")
builder.add_edge("human_review", "send")
builder.add_edge("send", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "email-campaign-1"}}

# ─── 1단계: 그래프 실행 (human_review에서 일시정지) ───
print("=== 1단계: 초안 생성 ===")
result = graph.invoke(
    {
        "messages": [HumanMessage(content="여름 세일 이메일을 보내고 싶습니다.")],
        "draft": "",
        "approved": False,
    },
    config=config,
)
# interrupt() 발생 시 result에는 interrupt 값이 포함됨

# 현재 상태 확인
state_snapshot = graph.get_state(config)
print(f"\n현재 멈춘 노드 이후: {state_snapshot.next}")

# ─── 2단계: 사람이 결정하고 재개 ───
print("\n=== 2단계: 승인 후 재개 ===")
final_result = graph.invoke(
    Command(resume="yes"),  # "yes"를 interrupt()의 반환값으로 전달
    config=config,
)
print(f"최종 메시지: {final_result['messages'][-1].content}")
```

### 예제 2: interrupt_before 브레이크포인트

```python
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


class State(TypedDict):
    query: str
    sql: str
    result: str


def generate_sql_node(state: State) -> dict:
    """자연어를 SQL로 변환."""
    response = llm.invoke(f"다음 질문을 SQL로 변환하세요 (PostgreSQL): {state['query']}")
    print(f"생성된 SQL:\n{response.content}")
    return {"sql": response.content}


def execute_sql_node(state: State) -> dict:
    """SQL 실행 (위험한 작업!)."""
    # 실제로는 DB에 쿼리를 실행하는 코드
    print(f"SQL 실행 중: {state['sql']}")
    return {"result": "실행 완료 (시뮬레이션)"}


builder = StateGraph(State)
builder.add_node("generate_sql", generate_sql_node)
builder.add_node("execute_sql", execute_sql_node)
builder.add_edge(START, "generate_sql")
builder.add_edge("generate_sql", "execute_sql")
builder.add_edge("execute_sql", END)

memory = MemorySaver()
# execute_sql 실행 전에 자동으로 멈춤
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["execute_sql"],
)

config = {"configurable": {"thread_id": "sql-review-1"}}

# ─── SQL 생성 후 자동 일시정지 ───
print("=== SQL 생성 단계 ===")
graph.invoke(
    {"query": "지난 30일간 주문 건수가 가장 많은 상위 5개 제품을 조회해주세요.", "sql": "", "result": ""},
    config=config,
)

state_snapshot = graph.get_state(config)
print(f"\n다음 실행 예정 노드: {state_snapshot.next}")
print("→ execute_sql 실행 전에 자동으로 멈췄습니다.")

# ─── 검토 후 재개 또는 SQL 수정 ───
print("\n=== SQL 수정 후 재개 ===")
# SQL이 잘못됐다면 update_state로 수정 가능
graph.update_state(
    config,
    {"sql": "SELECT product_id, COUNT(*) as orders FROM orders WHERE created_at >= NOW() - INTERVAL '30 days' GROUP BY product_id ORDER BY orders DESC LIMIT 5;"},
    as_node="generate_sql",
)

# None 전달 = 현재 상태에서 계속 실행
final = graph.invoke(None, config=config)
print(f"\n결과: {final['result']}")
```

### 예제 3: update_state와 Time Travel

```python
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
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

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "time-travel-demo"}}

# ─── 여러 턴 대화 ───
turns = [
    "파이썬과 자바의 차이를 설명해주세요.",
    "그럼 어떤 상황에 파이썬이 더 좋나요?",
    "데이터 과학 분야에서는요?",
]

for msg in turns:
    graph.invoke({"messages": [HumanMessage(content=msg)]}, config=config)
    print(f"Q: {msg}")
    state = graph.get_state(config)
    print(f"A: {state.values['messages'][-1].content[:100]}...\n")

# ─── 체크포인트 히스토리 조회 ───
print("=== 실행 히스토리 ===")
snapshots = list(graph.get_state_history(config))
for i, snapshot in enumerate(snapshots):
    checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
    msg_count = len(snapshot.values.get("messages", []))
    print(f"  [{i}] checkpoint_id={checkpoint_id[:8]}... | 메시지 수={msg_count}")

# ─── Time Travel: 첫 번째 질문 시점으로 되돌아가기 ───
# snapshots[-1]이 가장 오래된 체크포인트 (초기 상태)
# snapshots[-2]가 첫 번째 turn 완료 시점
past_config = snapshots[-2].config  # 첫 번째 대화 직후

print(f"\n=== Time Travel: 첫 번째 turn 시점으로 되돌아가기 ===")
print(f"되돌아갈 체크포인트: {past_config['configurable']['checkpoint_id'][:8]}...")

# 다른 질문으로 분기 (새 thread_id 권장)
fork_config = {"configurable": {"thread_id": "time-travel-fork"}}
graph.update_state(
    fork_config,
    snapshots[-2].values,  # 과거 상태를 새 thread에 복사
)
# 이후 다른 방향으로 대화 이어가기
graph.invoke(
    {"messages": [HumanMessage(content="그러면 러스트와 비교하면 어떤가요?")]},
    config=fork_config,
)
fork_state = graph.get_state(fork_config)
print(f"분기된 대화: {fork_state.values['messages'][-1].content[:100]}...")
```

---

## ✏️ 실습 과제

### 과제 1: 이중 승인 워크플로우

다음 워크플로우를 구현하세요:
1. 에이전트가 보고서 초안 작성
2. **1차 검토자** 승인 (`interrupt`)
3. **2차 검토자** 승인 (`interrupt`)
4. 둘 다 승인하면 발행, 하나라도 거부면 재작성

### 과제 2: 대화형 SQL 생성기

자연어 질문 → SQL 생성 → 사람 검토 → (수정 또는 실행) 루프를 구현하세요. 사람이 "수정: ..." 형태로 입력하면 LLM이 SQL을 재생성합니다.

### 과제 3: Time Travel 실험

같은 `thread_id`로 3번 대화한 뒤, 두 번째 대화 시점으로 되돌아가 다른 질문을 했을 때의 결과와 원래 결과를 비교하는 코드를 작성하세요.

---

## ⚠️ 흔한 함정

### 1. checkpointer 없이 interrupt() 사용

```python
# ❌ 오류: checkpointer 없으면 interrupt()가 작동하지 않음
graph = builder.compile()  # checkpointer 없음
graph.invoke(...)  # interrupt() 호출 시 NodeInterrupt 예외 발생

# ✅ 올바른 방법
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
```

### 2. 재개 시 다른 thread_id 사용

```python
# ❌ 잘못됨: 다른 thread_id로 재개하면 새 실행이 시작됨
graph.invoke(Command(resume="yes"), config={"configurable": {"thread_id": "other-id"}})

# ✅ 올바름: 동일한 thread_id의 config 사용
graph.invoke(Command(resume="yes"), config=config)  # 처음과 같은 config
```

### 3. update_state의 as_node 누락

`add_messages` 리듀서가 있는 `messages` 키를 `update_state()`로 수정할 때 `as_node`를 지정하지 않으면 리듀서가 적용되지 않아 예상과 다른 결과가 나올 수 있습니다.

```python
# as_node를 지정하여 해당 노드의 출력으로 처리
graph.update_state(config, {"messages": [new_message]}, as_node="chat_node")
```

### 4. interrupt_before와 interrupt()를 같은 노드에 혼용

한 노드에 `interrupt_before`와 내부 `interrupt()` 호출이 모두 있으면 두 번 멈춥니다. 용도를 명확히 분리하세요.

---

## ✅ 셀프 체크

- [ ] `interrupt()`의 반환값이 `Command(resume=...)`에서 전달한 값임을 이해했다.
- [ ] `interrupt_before`로 코드 수정 없이 브레이크포인트를 설정할 수 있다.
- [ ] `update_state()`로 재개 전에 상태를 수정할 수 있다.
- [ ] `get_state_history()`로 과거 체크포인트 목록을 조회할 수 있다.
- [ ] Time Travel로 과거 시점에서 다른 경로로 분기할 수 있다.

---

## 🔗 참고 자료

- [LangGraph Human-in-the-Loop 공식 문서](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [interrupt() API 가이드](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/)
- [Time Travel 가이드](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/)

> **API 변동 주의**: `interrupt()` API는 LangGraph 0.2에서 도입되었습니다. 이전 버전의 `NodeInterrupt` 방식과 다르므로 반드시 버전을 확인하세요. `from langgraph.types import interrupt`가 정상 임포트되지 않는 경우 공식 문서를 참조하세요.

---

⬅️ [Phase 23: 스트리밍](./23-streaming.md) | ➡️ [Phase 25: 서브그래프](./25-subgraphs.md)
