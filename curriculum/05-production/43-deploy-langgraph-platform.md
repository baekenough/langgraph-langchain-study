# Phase 43: LangGraph Platform / Server

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 2시간 30분 |
| 난이도 | ★★★☆☆ |
| 선행 학습 | Phase 18~26 (LangGraph Core), Phase 22 (체크포인터), Phase 42 (FastAPI 배포) |

---

## 🎯 학습 목표

- LangGraph Platform이 제공하는 기능과 순수 FastAPI 배포와의 차이를 설명할 수 있습니다.
- `langgraph.json` 설정 파일을 작성하여 그래프를 배포 단위로 패키징할 수 있습니다.
- `langgraph dev`로 로컬에서 LangGraph Server를 실행하고 Studio에서 시각화할 수 있습니다.
- Assistant API와 Thread API의 개념을 이해하고 기본 요청을 보낼 수 있습니다.

> **참고**: LangGraph Platform API는 활발히 발전 중입니다. 이 페이즈의 내용은 집필 시점 기준이며, 최신 정보는 [공식 문서](https://langchain-ai.github.io/langgraph/cloud/)를 반드시 확인하세요.

---

## 📚 핵심 개념

### 1. LangGraph Platform이란?

LangGraph Platform은 LangGraph 그래프를 프로덕션 환경에 쉽게 배포하기 위한 런타임·인프라 레이어입니다. 순수 FastAPI 배포(Phase 42)와 비교하면 다음과 같습니다.

| 기능 | 순수 FastAPI | LangGraph Platform |
|------|-------------|-------------------|
| 체크포인터·영속성 | 직접 구현 | 내장 (자동 PostgreSQL) |
| 스트리밍 | 직접 구현 | 내장 (SSE/WebSocket) |
| Thread/Assistant API | 직접 구현 | 내장 |
| LangGraph Studio | 없음 | 내장 (시각적 디버깅) |
| 수평 확장 | 직접 설계 | 자동 |
| 배포 복잡도 | 낮음(유연) | 낮음(관리형) |

**언제 사용하나요?**

- 그래프 로직에만 집중하고 인프라를 최소화하고 싶을 때
- LangGraph Studio로 팀과 실시간 디버깅·시각화가 필요할 때
- Thread 기반 멀티 사용자·멀티 세션 관리가 필요할 때

### 2. 핵심 추상화

```
Assistant  ── 배포된 그래프의 논리적 인스턴스 (구성·버전 포함)
    │
Thread     ── 하나의 대화 세션 (메시지 히스토리 + 상태 저장)
    │
Run        ── Thread에서 발생한 단일 그래프 실행
```

### 3. 배포 옵션

| 옵션 | 설명 | 적합한 경우 |
|------|------|-----------|
| **로컬 개발** | `langgraph dev` — 인메모리 실행 | 개발·프로토타이핑 |
| **Self-Hosted** | Docker 컨테이너로 자체 인프라 배포 | 데이터 주권·비용 최적화 |
| **LangGraph Cloud** | LangChain Inc.의 관리형 SaaS | 운영 부담 최소화 |

---

## 💻 코드 예제

### 예제 1: 프로젝트 구조 준비

```
my_langgraph_app/
├── .env                    # API keys
├── langgraph.json          # LangGraph Platform config
├── requirements.txt        # Dependencies
└── src/
    └── my_agent/
        ├── __init__.py
        └── graph.py        # Graph definition
```

### 예제 2: 그래프 정의 (`src/my_agent/graph.py`)

```python
# src/my_agent/graph.py
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

# NOTE: LangGraph 그래프 정의는 langgraph.json의 "graphs" 키로 참조됩니다.
# API 변동 → 공식 문서 확인: https://langchain-ai.github.io/langgraph/cloud/


class AgentState(TypedDict):
    """State for the conversational agent."""
    messages: Annotated[list[BaseMessage], add_messages]


def get_llm() -> ChatOpenAI:
    """Create LLM using OpenRouter."""
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )


async def chatbot_node(state: AgentState) -> AgentState:
    """Main chatbot node that calls the LLM."""
    llm = get_llm()
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}


def build_graph() -> StateGraph:
    """Build and compile the conversational graph."""
    builder = StateGraph(AgentState)
    builder.add_node("chatbot", chatbot_node)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    # LangGraph Platform automatically provides a checkpointer at runtime.
    # Do NOT compile with a checkpointer here — let the platform inject it.
    return builder.compile()


# The compiled graph — referenced by langgraph.json
graph = build_graph()
```

### 예제 3: `langgraph.json` 설정 파일

```json
{
  "dependencies": ["."],
  "graphs": {
    "my_agent": "./src/my_agent/graph.py:graph"
  },
  "env": ".env"
}
```

**필드 설명:**

| 필드 | 설명 |
|------|------|
| `dependencies` | 패키지 경로 (pip install 대상) |
| `graphs` | `"assistant_name": "module_path:graph_object"` 형식 |
| `env` | 환경 변수 파일 경로 |

여러 그래프를 등록할 수 있습니다.

```json
{
  "dependencies": ["."],
  "graphs": {
    "chatbot": "./src/my_agent/graph.py:graph",
    "rag_agent": "./src/rag/graph.py:rag_graph"
  },
  "env": ".env"
}
```

### 예제 4: 로컬 개발 서버 실행

```bash
# langgraph-cli 설치 (langgraph-cli[inmem]은 인메모리 체크포인터 포함)
pip install "langgraph-cli[inmem]"

# 로컬 LangGraph Server 시작
langgraph dev

# 출력 예시:
# Ready!
# - API: http://localhost:2024
# - Docs: http://localhost:2024/docs
# - LangGraph Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

`langgraph dev`는 코드 변경을 감지하여 자동으로 재시작합니다.

### 예제 5: Assistant API 및 Thread API 사용

```python
# client_example.py
import asyncio
import httpx
import json

BASE_URL = "http://localhost:2024"

# NOTE: LangGraph Platform REST API는 변경될 수 있습니다.
# 항상 /docs 엔드포인트에서 현재 스키마를 확인하세요.


async def create_assistant() -> str:
    """Create an assistant for the 'my_agent' graph."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/assistants",
            json={"graph_id": "my_agent"},
        )
        response.raise_for_status()
        assistant = response.json()
        print(f"Assistant created: {assistant['assistant_id']}")
        return assistant["assistant_id"]


async def create_thread(assistant_id: str) -> str:
    """Create a new conversation thread."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/threads",
            json={},
        )
        response.raise_for_status()
        thread = response.json()
        print(f"Thread created: {thread['thread_id']}")
        return thread["thread_id"]


async def run_and_stream(assistant_id: str, thread_id: str, message: str) -> None:
    """Send a message and stream the response via SSE."""
    payload = {
        "assistant_id": assistant_id,
        "input": {
            "messages": [{"role": "human", "content": message}]
        },
        "stream_mode": "messages",
    }

    print(f"\nUser: {message}")
    print("Assistant: ", end="", flush=True)

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/threads/{thread_id}/runs/stream",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    # Message events contain token chunks
                    if isinstance(data, list):
                        for item in data:
                            if item.get("type") == "AIMessageChunk":
                                print(item.get("content", ""), end="", flush=True)
                except json.JSONDecodeError:
                    pass

    print()  # newline after stream ends


async def get_thread_history(thread_id: str) -> None:
    """Retrieve full conversation history for a thread."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/threads/{thread_id}/state")
        response.raise_for_status()
        state = response.json()
        messages = state.get("values", {}).get("messages", [])
        print(f"\n--- Thread History ({len(messages)} messages) ---")
        for msg in messages:
            role = msg.get("type", "unknown")
            content = msg.get("content", "")
            print(f"[{role}]: {content[:80]}...")


async def main() -> None:
    """Full example: create assistant → thread → run → history."""
    # Step 1: Create assistant
    assistant_id = await create_assistant()

    # Step 2: Create thread (= new conversation session)
    thread_id = await create_thread(assistant_id)

    # Step 3: First message
    await run_and_stream(assistant_id, thread_id, "안녕하세요! LangGraph Platform이 뭔가요?")

    # Step 4: Follow-up (same thread — context is preserved)
    await run_and_stream(assistant_id, thread_id, "방금 설명하신 내용 중 배포 옵션을 더 자세히 설명해주세요.")

    # Step 5: Inspect conversation history
    await get_thread_history(thread_id)


if __name__ == "__main__":
    asyncio.run(main())
```

### 예제 6: `requirements.txt`

```
# requirements.txt
langchain>=0.3.0
langchain-openai>=0.2.0
langgraph>=0.2.0
python-dotenv>=1.0.0

# Development only
langgraph-cli[inmem]
httpx
```

---

## ✏️ 실습 과제

**과제 1 — ReAct 에이전트를 Platform에 배포**

Phase 27에서 만든 ReAct 에이전트를 `langgraph.json`에 등록하고 `langgraph dev`로 실행하세요. LangGraph Studio에서 실행 흐름을 시각화하세요.

**과제 2 — 여러 Thread 병렬 관리**

`asyncio.gather()`를 사용하여 3개의 서로 다른 thread에서 동시에 질문을 전송하고, 각각 독립적으로 대화 히스토리가 유지되는지 확인하세요.

```python
async def parallel_sessions() -> None:
    """Run 3 independent conversations concurrently."""
    assistant_id = await create_assistant()
    threads = await asyncio.gather(*[create_thread(assistant_id) for _ in range(3)])

    questions = [
        "Python에 대해 알려주세요",
        "FastAPI가 뭔가요?",
        "LangGraph의 장점은?",
    ]

    await asyncio.gather(*[
        run_and_stream(assistant_id, tid, q)
        for tid, q in zip(threads, questions)
    ])
```

**과제 3 — Self-Hosted 배포 조사**

[공식 셀프호스팅 가이드](https://langchain-ai.github.io/langgraph/cloud/deployment/self_hosted/)를 읽고, Docker Compose 기반 배포에서 어떤 서비스(컨테이너)가 필요한지 목록을 작성하세요.

---

## ⚠️ 흔한 함정

**1. 그래프를 체크포인터와 함께 컴파일하는 실수**

`langgraph.json`으로 배포할 때는 `builder.compile(checkpointer=...)` 형태로 체크포인터를 직접 주입하면 안 됩니다. Platform이 런타임에 자동으로 체크포인터를 주입합니다.

**2. `langgraph.json`의 그래프 경로 오류**

`"./src/my_agent/graph.py:graph"` 형식에서 콜론(`:`) 뒤는 파이썬 모듈 내 변수 이름입니다. 함수가 아닌 컴파일된 그래프 객체여야 합니다.

**3. `langgraph-cli` 없이 `langgraph dev` 실행 시도**

`langgraph-cli`는 `langgraph` 패키지에 포함되지 않습니다. `pip install "langgraph-cli[inmem]"`을 별도로 설치해야 합니다.

**4. API 스키마가 바뀌었는데 오래된 예제 사용**

LangGraph Platform REST API는 활발히 업데이트됩니다. 로컬 실행 후 `http://localhost:2024/docs`에서 현재 API 스키마를 항상 먼저 확인하세요.

**5. 로컬 개발과 Cloud 배포의 환경 변수 차이**

`langgraph dev`는 `.env` 파일을 자동으로 읽지만, Cloud 배포에서는 별도로 환경 변수를 설정해야 합니다.

---

## ✅ 셀프 체크

- [ ] `langgraph.json`을 올바르게 작성했다.
- [ ] `langgraph dev` 명령으로 로컬 서버가 `http://localhost:2024`에서 실행된다.
- [ ] LangGraph Studio URL에서 그래프 실행 흐름을 시각화했다.
- [ ] Assistant API로 어시스턴트를 생성했다.
- [ ] Thread API로 세션을 생성하고 메시지를 전송했다.
- [ ] 같은 thread에 두 번 이상 메시지를 전송하면 대화 맥락이 유지됨을 확인했다.
- [ ] Self-Hosted와 Cloud 배포 옵션의 차이를 설명할 수 있다.

---

## 🔗 참고 자료

- [LangGraph Platform 공식 문서](https://langchain-ai.github.io/langgraph/cloud/)
- [LangGraph CLI](https://langchain-ai.github.io/langgraph/cloud/reference/cli/)
- [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/)
- [Self-Hosted 배포 가이드](https://langchain-ai.github.io/langgraph/cloud/deployment/self_hosted/)
- [LangGraph Platform REST API Reference](https://langchain-ai.github.io/langgraph/cloud/reference/api/api_ref/)

---

## 🔗 네비게이션

| 이전 | 현재 | 다음 |
|------|------|------|
| [Phase 42: FastAPI로 배포](42-deploy-fastapi.md) | **Phase 43: LangGraph Platform** | [Phase 44: 관측성·모니터링](44-observability-monitoring.md) |
