# Phase 26: 장기 메모리 스토어

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 105분 |
| 난이도 | ★★★★☆ |
| 선행 학습 | Phase 25 (서브그래프) |

---

## 🎯 학습 목표

- 단기 메모리(체크포인터)와 장기 메모리(Store)의 차이를 설명할 수 있습니다.
- `InMemoryStore`를 사용하여 스레드를 초월한 정보를 저장하고 조회할 수 있습니다.
- `put` / `get` / `search` API를 사용할 수 있습니다.
- 네임스페이스 패턴으로 사용자별 메모리를 분리할 수 있습니다.
- 노드 함수에서 Store에 접근하는 방법을 이해합니다.

---

## 📚 핵심 개념

### 단기 메모리 vs 장기 메모리

LangGraph에는 두 종류의 기억 메커니즘이 있습니다:

```
단기 메모리 (Checkpointer)          장기 메모리 (Store)
────────────────────────            ──────────────────────
• thread_id에 종속             vs   • thread_id에 독립
• 대화 내 상태 보존                 • 여러 대화에 걸쳐 보존
• 자동 저장 (노드 실행 후)          • 명시적 저장 (put() 호출)
• 구조: 직렬화된 상태 스냅샷        • 구조: 키-값 문서 저장소
• 예: 이번 대화의 메시지 이력       • 예: 사용자 선호도, 프로필
```

**비유**: 단기 메모리는 현재 대화 중에 사용하는 노트, 장기 메모리는 사용자 카드함입니다.

```
Thread A (사용자 철수)                   Thread B (사용자 철수)
┌────────────────────┐                   ┌────────────────────┐
│ Checkpointer       │  서로 독립         │ Checkpointer       │
│ - 이번 대화 메시지 │ ──────────────── │ - 이번 대화 메시지 │
└────────────────────┘                   └────────────────────┘
         │                                        │
         └──────────────────┬─────────────────────┘
                            ▼
              ┌─────────────────────────┐
              │ Store (사용자 철수)     │  ← thread 공유
              │ - 이름: 철수            │
              │ - 선호 언어: 한국어     │
              │ - 관심사: AI, 파이썬    │
              └─────────────────────────┘
```

### BaseStore API

```python
from langgraph.store.base import BaseStore

# put: 데이터 저장
# namespace: 계층적 구분자 (튜플)
# key: 항목 식별자 (문자열)
# value: 저장할 딕셔너리
store.put(("user", "cheolsu", "preferences"), "language", {"lang": "ko"})

# get: 단일 항목 조회
item = store.get(("user", "cheolsu", "preferences"), "language")
if item:
    print(item.value)  # {"lang": "ko"}

# search: 네임스페이스 내 검색
# query가 있으면 의미 검색 (Store가 지원하는 경우)
# query가 없으면 전체 나열
items = store.search(("user", "cheolsu", "preferences"))
for item in items:
    print(item.key, item.value)

# delete: 항목 삭제
store.delete(("user", "cheolsu", "preferences"), "language")
```

### 네임스페이스 설계 패턴

네임스페이스는 파일 시스템 경로처럼 계층 구조를 만듭니다:

```python
# 사용자별 분리
("user", user_id, "profile")      # 사용자 프로필
("user", user_id, "preferences")  # 사용자 설정
("user", user_id, "memories")     # 대화에서 학습한 내용

# 세션 + 사용자
("session", session_id, "context")

# 전역 설정
("global", "config")
```

### 노드에서 Store 접근

그래프를 `store=store`로 컴파일하면 노드 함수의 매개변수에 `BaseStore` 타입을 선언하는 것만으로 자동 주입됩니다:

```python
from langgraph.store.base import BaseStore

def my_node(state: State, store: BaseStore) -> dict:
    # store가 자동으로 주입됨
    items = store.search(("user", state["user_id"]))
    ...
```

---

## 💻 코드 예제

### 예제 1: InMemoryStore 기본 사용

```python
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

# InMemoryStore 생성 (프로세스 내에서 유지)
store = InMemoryStore()

# ─── 기본 CRUD ───
# put: 저장
store.put(("users", "cheolsu"), "profile", {
    "name": "홍철수",
    "age": 30,
    "interests": ["파이썬", "AI"],
})
store.put(("users", "cheolsu"), "preference", {
    "language": "한국어",
    "response_style": "친근하게",
})

# get: 단일 조회
profile_item = store.get(("users", "cheolsu"), "profile")
if profile_item:
    print(f"프로필: {profile_item.value}")
    print(f"마지막 업데이트: {profile_item.updated_at}")

# search: 네임스페이스 내 전체 조회
print("\n철수의 모든 데이터:")
for item in store.search(("users", "cheolsu")):
    print(f"  [{item.key}] {item.value}")

# 없는 항목 조회 → None 반환
missing = store.get(("users", "unknown_user"), "profile")
print(f"\n없는 항목: {missing}")  # None

# delete: 삭제
store.delete(("users", "cheolsu"), "preference")
print("\n삭제 후 항목 수:", len(list(store.search(("users", "cheolsu")))))
```

### 예제 2: 그래프에서 Store 활용 (사용자 기억)

```python
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
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
    user_id: str


def load_memories_node(state: State, store: BaseStore) -> dict:
    """대화 시작 시 사용자 기억을 로드하여 시스템 프롬프트에 반영."""
    namespace = ("user_memories", state["user_id"])
    memories = store.search(namespace)

    if memories:
        memory_text = "\n".join([
            f"- {item.key}: {item.value['content']}"
            for item in memories
        ])
        system_msg = SystemMessage(
            content=f"당신은 도움이 되는 AI 어시스턴트입니다.\n\n"
                    f"이 사용자에 대해 알고 있는 정보:\n{memory_text}"
        )
    else:
        system_msg = SystemMessage(content="당신은 도움이 되는 AI 어시스턴트입니다.")

    print(f"[메모리 로드] {len(memories)}개 기억 로드됨")
    return {"messages": system_msg}


def chat_node(state: State) -> dict:
    """LLM으로 응답 생성."""
    response = llm.invoke(state["messages"])
    return {"messages": response}


def save_memories_node(state: State, store: BaseStore) -> dict:
    """대화에서 새로운 정보를 추출하여 저장."""
    namespace = ("user_memories", state["user_id"])

    # 마지막 사용자 메시지에서 기억할 만한 정보 추출
    user_messages = [
        m for m in state["messages"]
        if isinstance(m, HumanMessage)
    ]
    if not user_messages:
        return {}

    last_user_msg = user_messages[-1].content
    extraction_response = llm.invoke(
        f"다음 대화에서 사용자에 대한 중요한 개인 정보나 선호도를 추출하세요. "
        f"없으면 'NONE'이라고만 답하세요.\n\n메시지: {last_user_msg}"
    )
    extracted = extraction_response.content.strip()

    if extracted != "NONE" and len(extracted) > 5:
        import hashlib
        key = hashlib.md5(last_user_msg.encode()).hexdigest()[:8]
        store.put(namespace, key, {"content": extracted})
        print(f"[메모리 저장] 새 기억: {extracted[:50]}...")

    return {}


# ─── 그래프 구성 ───
builder = StateGraph(State)
builder.add_node("load_memories", load_memories_node)
builder.add_node("chat", chat_node)
builder.add_node("save_memories", save_memories_node)
builder.add_edge(START, "load_memories")
builder.add_edge("load_memories", "chat")
builder.add_edge("chat", "save_memories")
builder.add_edge("save_memories", END)

memory = MemorySaver()
store = InMemoryStore()
graph = builder.compile(checkpointer=memory, store=store)  # 둘 다 전달


# ─── 시뮬레이션: 사용자 철수와 두 번의 독립 대화 ───
user_id = "cheolsu"

# 첫 번째 대화 (thread 1)
print("=== 첫 번째 대화 ===")
config1 = {"configurable": {"thread_id": "session-001"}}
graph.invoke(
    {"messages": [HumanMessage(content="안녕하세요! 저는 파이썬 개발자이고 머신러닝에 관심이 많습니다.")], "user_id": user_id},
    config=config1,
)
state1 = graph.get_state(config1)
print(f"AI: {state1.values['messages'][-2].content[:80]}...\n")

# 두 번째 대화 (thread 2 — 다른 세션이지만 같은 store)
print("=== 두 번째 대화 (새 세션) ===")
config2 = {"configurable": {"thread_id": "session-002"}}
result2 = graph.invoke(
    {"messages": [HumanMessage(content="저에게 좋은 학습 자료를 추천해주세요.")], "user_id": user_id},
    config=config2,
)
# load_memories_node가 첫 번째 대화에서 저장된 기억을 자동으로 로드함
print(f"AI: {result2['messages'][-2].content[:120]}...")
```

### 예제 3: 네임스페이스 기반 다중 사용자 메모리

```python
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


class State(TypedDict):
    user_id: str
    action: str   # "save" | "recall" | "list"
    data: dict
    result: str


def memory_manager_node(state: State, store: BaseStore) -> dict:
    """사용자별 메모리를 관리하는 노드."""
    namespace = ("personal", state["user_id"])
    action = state["action"]

    if action == "save":
        key = state["data"].get("key", "default")
        store.put(namespace, key, state["data"])
        return {"result": f"저장 완료: {key}"}

    elif action == "recall":
        key = state["data"].get("key", "")
        item = store.get(namespace, key)
        if item:
            return {"result": f"조회 성공: {item.value}"}
        return {"result": f"키 '{key}'를 찾을 수 없습니다."}

    elif action == "list":
        items = store.search(namespace)
        summary = [f"  {item.key}: {str(item.value)[:50]}" for item in items]
        return {"result": "저장된 항목:\n" + "\n".join(summary) if summary else "저장된 항목 없음"}

    return {"result": "알 수 없는 액션"}


builder = StateGraph(State)
builder.add_node("manage_memory", memory_manager_node)
builder.add_edge(START, "manage_memory")
builder.add_edge("manage_memory", END)

store = InMemoryStore()
graph = builder.compile(store=store)  # checkpointer 없이 store만 전달 가능

# ─── 사용자 A의 데이터 ───
graph.invoke({"user_id": "alice", "action": "save", "data": {"key": "goal", "content": "AI 엔지니어 취업"}, "result": ""})
graph.invoke({"user_id": "alice", "action": "save", "data": {"key": "skill", "content": "파이썬, LangGraph"}, "result": ""})

# ─── 사용자 B의 데이터 ───
graph.invoke({"user_id": "bob", "action": "save", "data": {"key": "goal", "content": "스타트업 창업"}, "result": ""})

# ─── 각 사용자 데이터 조회 ───
result_a = graph.invoke({"user_id": "alice", "action": "list", "data": {}, "result": ""})
print(f"Alice의 데이터:\n{result_a['result']}\n")

result_b = graph.invoke({"user_id": "bob", "action": "list", "data": {}, "result": ""})
print(f"Bob의 데이터:\n{result_b['result']}")

# ─── 네임스페이스 독립성 확인 ───
# Alice의 데이터가 Bob에게 보이지 않음
cross_check = store.get(("personal", "bob"), "skill")  # Alice의 skill 키를 Bob 네임스페이스에서 조회
print(f"\n네임스페이스 독립성 확인: {cross_check}")  # None
```

### 예제 4: 체크포인터 + Store 조합 패턴

```python
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# ─── 두 메모리 메커니즘의 역할 분담 ───
#
# Checkpointer (thread-scoped)     Store (cross-thread)
# ─────────────────────────────    ─────────────────────
# 이번 대화의 메시지 이력           사용자 장기 프로필
# 현재 처리 중인 상태               학습된 선호도
# 중단 후 재개를 위한 스냅샷        여러 세션의 공통 지식
# ─────────────────────────────    ─────────────────────

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


class PersonalAssistantState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str


def personalized_chat_node(state: PersonalAssistantState, store: BaseStore) -> dict:
    """Store에서 사용자 정보를 읽어 맞춤형 응답을 생성."""
    user_id = state["user_id"]
    namespace = ("assistant", user_id)

    # 사용자 프로필 로드
    profile_item = store.get(namespace, "profile")
    history_items = store.search(namespace)

    # 컨텍스트 구성
    context_parts = []
    if profile_item:
        context_parts.append(f"사용자 프로필: {profile_item.value}")
    if history_items:
        facts = [item.value.get("fact", "") for item in history_items if item.key != "profile"]
        if facts:
            context_parts.append(f"알려진 사실: {'; '.join(facts[:3])}")

    context = "\n".join(context_parts) if context_parts else "신규 사용자"

    # LLM 호출
    system_prompt = f"개인 비서입니다. 사용자 정보: {context}"
    all_messages = [
        AIMessage(content=system_prompt),  # 간단한 시스템 컨텍스트
    ] + state["messages"]

    response = llm.invoke(all_messages)

    # 대화에서 새로운 사실 추출 후 저장
    last_user_msg = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if last_user_msg:
        extract = llm.invoke(
            f"메시지에서 사실 정보 1개만 추출하세요. 없으면 NONE: {last_user_msg[-1].content}"
        )
        if extract.content.strip() != "NONE":
            import time
            store.put(namespace, f"fact_{int(time.time())}", {"fact": extract.content.strip()})

    return {"messages": response}


builder = StateGraph(PersonalAssistantState)
builder.add_node("chat", personalized_chat_node)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

memory = MemorySaver()
store = InMemoryStore()

# 사용자 초기 프로필 사전 입력
store.put(("assistant", "user_001"), "profile", {
    "name": "김개발",
    "role": "백엔드 개발자",
    "expertise": ["Python", "FastAPI"],
})

graph = builder.compile(checkpointer=memory, store=store)

# ─── 여러 세션에서 맞춤형 대화 ───
user_id = "user_001"

# 세션 1
config_s1 = {"configurable": {"thread_id": "session-a"}}
r1 = graph.invoke(
    {"messages": [HumanMessage(content="LangGraph를 배우고 싶습니다.")], "user_id": user_id},
    config=config_s1,
)
print(f"세션 1 응답: {r1['messages'][-1].content[:100]}...")

# 세션 2 (다른 thread, Store 공유)
config_s2 = {"configurable": {"thread_id": "session-b"}}
r2 = graph.invoke(
    {"messages": [HumanMessage(content="어제 대화 기억하시나요?")], "user_id": user_id},
    config=config_s2,
)
print(f"세션 2 응답: {r2['messages'][-1].content[:100]}...")

# Store에 축적된 정보 확인
print("\n=== Store에 저장된 정보 ===")
for item in store.search(("assistant", user_id)):
    print(f"  [{item.key}] {item.value}")
```

---

## ✏️ 실습 과제

### 과제 1: 선호도 학습 챗봇

사용자가 대화할수록 선호도(음식, 관심사, 언어 스타일 등)를 Store에 학습하고, 다음 세션에서 자동으로 맞춤형 응답을 제공하는 챗봇을 만드세요.

### 과제 2: 팀 지식베이스

여러 사용자가 공유하는 `("team", "knowledge")` 네임스페이스에 정보를 저장하고, 어떤 사용자의 질문에도 팀 공통 지식으로 응답하는 에이전트를 구현하세요.

### 과제 3: 메모리 만료 시뮬레이션

`item.updated_at`을 활용하여 30일 이상 된 메모리는 자동으로 삭제하는 메모리 정리 노드를 작성하세요.

---

## ⚠️ 흔한 함정

### 1. Store 없이 그래프 컴파일 후 Store 주입 시도

```python
# ❌ 오류: store 없이 컴파일된 그래프에서 Store 주입 안 됨
graph = builder.compile(checkpointer=memory)  # store 누락

def my_node(state, store: BaseStore):  # store가 None으로 들어옴
    store.put(...)  # AttributeError!

# ✅ 올바름: 컴파일 시 store 전달
store = InMemoryStore()
graph = builder.compile(checkpointer=memory, store=store)
```

### 2. get() 반환값의 None 처리 누락

`store.get()`은 키가 없으면 `None`을 반환합니다. `.value`에 바로 접근하면 `AttributeError`가 발생합니다.

```python
# ❌ 위험: None 확인 없이 .value 접근
item = store.get(namespace, "key")
print(item.value)  # item이 None이면 AttributeError

# ✅ 안전: None 확인 후 접근
item = store.get(namespace, "key")
if item:
    print(item.value)
else:
    print("항목 없음")
```

### 3. InMemoryStore의 프로세스 종속성

`InMemoryStore`는 프로세스가 종료되면 모든 데이터가 사라집니다. 개발/테스트에는 적합하지만 프로덕션에서는 `AsyncPostgresStore` 등 영구 저장소를 사용하세요.

### 4. search()와 get()의 반환 타입 차이

```python
# get(): 단일 Item 또는 None
item = store.get(namespace, "key")  # Item | None

# search(): Item 리스트 (빈 리스트 가능)
items = store.search(namespace)  # list[Item]
for item in items:
    print(item.value)
```

---

## ✅ 셀프 체크

- [ ] Checkpointer(단기)와 Store(장기)의 차이를 설명할 수 있다.
- [ ] `put()` / `get()` / `search()`의 시그니처와 반환 타입을 안다.
- [ ] 네임스페이스 튜플로 사용자별 데이터를 분리할 수 있다.
- [ ] 노드 함수 매개변수에 `store: BaseStore`를 선언하여 자동 주입받을 수 있다.
- [ ] `get()` 반환값의 `None` 처리를 안전하게 할 수 있다.

---

## 🔗 참고 자료

- [LangGraph Memory Store 공식 문서](https://langchain-ai.github.io/langgraph/concepts/memory/)
- [InMemoryStore API 레퍼런스](https://langchain-ai.github.io/langgraph/reference/store/)
- [장기 메모리 How-to 가이드](https://langchain-ai.github.io/langgraph/how-tos/memory/manage-conversation-history/)

> **API 변동 주의**: `BaseStore`의 `search()` 의미 검색 기능은 Store 구현체에 따라 다릅니다. `InMemoryStore`는 필터 기반, PostgreSQL 기반 Store는 벡터 검색을 지원할 수 있습니다. 버전별 지원 기능은 공식 문서를 확인하세요.

---

⬅️ [Phase 25: 서브그래프](./25-subgraphs.md) | ➡️ [Phase 27: ReAct 에이전트](../04-agents/27-react-agent.md)
