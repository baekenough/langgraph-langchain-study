# Phase 42: FastAPI로 배포

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 3시간 |
| 난이도 | ★★★☆☆ |
| 선행 학습 | Phase 07 (LCEL), Phase 08 (스트리밍·비동기), Phase 22 (체크포인터) |

---

## 🎯 학습 목표

- LangGraph/LCEL 앱을 FastAPI 비동기 엔드포인트로 서빙할 수 있습니다.
- SSE(Server-Sent Events)로 토큰 단위 스트리밍 응답을 구현할 수 있습니다.
- Pydantic으로 요청·응답 스키마를 정의하고 자동 검증할 수 있습니다.
- `thread_id`를 이용한 세션 기반 대화 관리를 구현할 수 있습니다.
- CORS 설정 및 환경 변수 관리를 적용할 수 있습니다.

> **참고**: LangServe는 현재 LangChain 팀이 권장하지 않으며, 이 페이즈에서는 순수 FastAPI를 직접 사용합니다.

---

## 📚 핵심 개념

### 1. 왜 FastAPI인가?

FastAPI는 Python의 비동기 웹 프레임워크로, LangGraph/LCEL의 `async` API와 자연스럽게 통합됩니다.

| 기능 | 설명 |
|------|------|
| `async def` 엔드포인트 | 동시 요청을 블로킹 없이 처리 |
| `StreamingResponse` | 토큰 스트리밍을 SSE로 전달 |
| Pydantic 통합 | 요청 자동 검증·타입 힌트·자동 문서화 |
| 자동 OpenAPI 문서 | `/docs`에서 인터랙티브 테스트 가능 |

### 2. Server-Sent Events (SSE)

SSE는 서버에서 클라이언트로 단방향 실시간 데이터를 푸시하는 HTTP 기반 프로토콜입니다. LLM 토큰 스트리밍에 적합합니다.

```
Client: GET /chat/stream
Server: data: {"token": "안"}
        data: {"token": "녕"}
        data: {"token": "하"}
        data: [DONE]
```

Phase 08(스트리밍·비동기)에서 배운 `astream()` 메서드를 FastAPI `StreamingResponse`에 연결하면 됩니다.

### 3. 세션 관리 with thread_id

LangGraph의 체크포인터(Phase 22)는 `thread_id`로 대화 히스토리를 저장합니다. 클라이언트가 동일한 `thread_id`를 전달하면 이전 대화가 자동으로 복원됩니다.

```
Client ──thread_id: "session-abc"──► FastAPI ──config──► LangGraph
                                                         ▼
                                                   Checkpointer (SQLite/Postgres)
                                                         ▼
                                                   이전 메시지 자동 로드
```

---

## 💻 코드 예제

### 예제 1: 기본 FastAPI 서버 구조

```python
# app/main.py
import os
import json
import asyncio
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, SecretStr

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

load_dotenv()

# NOTE: FastAPI/LangChain API는 변경될 수 있습니다. 공식 문서를 확인하세요.
# https://fastapi.tiangolo.com  /  https://python.langchain.com

app = FastAPI(
    title="LangChain API",
    description="LangGraph/LCEL powered chat API",
    version="1.0.0",
)

# CORS — allow all origins in development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response schemas ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Chat request payload."""
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    thread_id: str = Field(default="default", description="Session identifier")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    """Non-streaming chat response."""
    answer: str
    thread_id: str


# ── LLM factory ──────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Create a ChatOpenAI instance using OpenRouter."""
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
    )


PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer concisely in Korean."),
    ("human", "{message}"),
])


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Non-streaming chat endpoint."""
    llm = get_llm(temperature=request.temperature)
    chain = PROMPT | llm

    try:
        response = await chain.ainvoke({"message": request.message})
        return ChatResponse(
            answer=response.content,
            thread_id=request.thread_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 예제 2: SSE 스트리밍 엔드포인트

```python
# app/streaming.py  (위 main.py에 추가하거나 별도 라우터로 분리)
import os
import json
import asyncio
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

router = APIRouter(prefix="/chat", tags=["streaming"])


class StreamRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str = Field(default="default")


async def token_generator(message: str) -> AsyncGenerator[str, None]:
    """
    Async generator yielding SSE-formatted tokens.

    SSE format: "data: {json}\\n\\n"
    """
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        streaming=True,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer in Korean."),
        ("human", "{message}"),
    ])
    chain = prompt | llm

    try:
        # astream() yields AIMessageChunk objects — see Phase 08
        async for chunk in chain.astream({"message": message}):
            token = chunk.content
            if token:
                payload = json.dumps({"token": token}, ensure_ascii=False)
                yield f"data: {payload}\n\n"
                # Small sleep prevents overwhelming the client
                await asyncio.sleep(0)

        # Signal stream completion
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_payload = json.dumps({"error": str(e)})
        yield f"data: {error_payload}\n\n"


@router.post("/stream")
async def stream_chat(request: StreamRequest) -> StreamingResponse:
    """
    Streaming chat endpoint using Server-Sent Events.

    Client reads:
        const es = new EventSource('/chat/stream');
        es.onmessage = (e) => {
            if (e.data === '[DONE]') return es.close();
            const { token } = JSON.parse(e.data);
            output += token;
        };
    """
    return StreamingResponse(
        token_generator(request.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
```

### 예제 3: LangGraph 앱 통합 + thread_id 세션 관리

```python
# app/graph_endpoint.py
import os
import json
import uuid
from typing import Annotated, AsyncGenerator

from dotenv import load_dotenv
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, SecretStr

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

load_dotenv()

router = APIRouter(prefix="/graph", tags=["langgraph"])


# ── LangGraph application ─────────────────────────────────────────────────────

class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph() -> StateGraph:
    """Build a simple conversational LangGraph."""
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )

    async def chat_node(state: ConversationState) -> ConversationState:
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}

    builder = StateGraph(ConversationState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)

    # MemorySaver persists conversations per thread_id
    # In production, replace with SqliteSaver or PostgresSaver (Phase 22)
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# Build graph once at startup — shared across requests
GRAPH = build_graph()


# ── API schemas ───────────────────────────────────────────────────────────────

class GraphChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    thread_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session ID. Auto-generated if not provided.",
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/chat")
async def graph_chat(request: GraphChatRequest) -> dict:
    """
    Invoke LangGraph with session persistence via thread_id.

    Same thread_id resumes the previous conversation.
    """
    config = {"configurable": {"thread_id": request.thread_id}}
    input_state = {"messages": [HumanMessage(content=request.message)]}

    result = await GRAPH.ainvoke(input_state, config=config)
    last_message = result["messages"][-1]

    return {
        "answer": last_message.content,
        "thread_id": request.thread_id,
    }


async def graph_stream_generator(
    message: str, thread_id: str
) -> AsyncGenerator[str, None]:
    """Stream tokens from LangGraph using astream_events."""
    config = {"configurable": {"thread_id": thread_id}}
    input_state = {"messages": [HumanMessage(content=message)]}

    # astream_events yields fine-grained events including token chunks
    async for event in GRAPH.astream_events(input_state, config=config, version="v2"):
        kind = event.get("event")
        if kind == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk and chunk.content:
                payload = json.dumps({"token": chunk.content}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/stream")
async def graph_stream(request: GraphChatRequest) -> StreamingResponse:
    """Stream LangGraph responses via SSE with session persistence."""
    return StreamingResponse(
        graph_stream_generator(request.message, request.thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

### 예제 4: 애플리케이션 진입점 및 실행

```python
# app/main_full.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Assume graph_endpoint.py is in the same package
# from app.graph_endpoint import router as graph_router

load_dotenv()

app = FastAPI(title="LangChain Production API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.include_router(graph_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# ── Run with uvicorn ──────────────────────────────────────────────────────────
# uvicorn app.main_full:app --host 0.0.0.0 --port 8000 --reload
#
# Docker (한 줄 언급):
# docker build -t langchain-api . && docker run -p 8000:8000 --env-file .env langchain-api
```

### 예제 5: 클라이언트 테스트

```python
# test_client.py
import asyncio
import httpx
import json


async def test_streaming() -> None:
    """Test the SSE streaming endpoint."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Non-streaming
        response = await client.post(
            "http://localhost:8000/graph/chat",
            json={"message": "안녕하세요!", "thread_id": "test-session-1"},
        )
        print("Non-streaming:", response.json())

        # Streaming — follow-up in same session
        print("\nStreaming (follow-up):")
        async with client.stream(
            "POST",
            "http://localhost:8000/graph/stream",
            json={"message": "제 이름을 물어봤나요?", "thread_id": "test-session-1"},
        ) as stream_resp:
            async for line in stream_resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        print("\n[Stream complete]")
                        break
                    parsed = json.loads(data)
                    print(parsed.get("token", ""), end="", flush=True)


if __name__ == "__main__":
    asyncio.run(test_streaming())
```

---

## ✏️ 실습 과제

**과제 1 — 미들웨어로 API 키 인증 추가**

`Authorization: Bearer <token>` 헤더를 검증하는 FastAPI 의존성(Dependency)을 구현하세요.

```python
from fastapi import Depends, HTTPException, Header

async def verify_api_key(authorization: str = Header(...)) -> str:
    """Validate bearer token."""
    expected = os.environ.get("API_SECRET_KEY", "")
    if not authorization.startswith("Bearer ") or authorization[7:] != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return authorization[7:]

# Usage: @router.post("/chat", dependencies=[Depends(verify_api_key)])
```

**과제 2 — 요청 로깅 미들웨어**

모든 요청의 경로, 메서드, 소요 시간을 기록하는 `BaseHTTPMiddleware`를 작성하세요.

**과제 3 — 스트리밍 취소 처리**

클라이언트가 연결을 끊으면 서버 측에서 `asyncio.CancelledError`를 잡아 LLM 스트리밍을 정상 종료하는 코드를 추가하세요.

---

## ⚠️ 흔한 함정

**1. 동기 LangChain 호출을 async 엔드포인트에서 직접 사용**

`chain.invoke()` 같은 동기 메서드를 `async def` 엔드포인트 안에서 직접 호출하면 이벤트 루프가 블로킹됩니다. 반드시 `await chain.ainvoke()`를 사용하세요.

**2. `streaming=True`를 설정하지 않고 `astream()` 사용**

`ChatOpenAI`에서 `streaming=True`를 설정하지 않으면 `astream()`이 전체 응답을 한 번에 반환합니다.

**3. GRAPH를 요청마다 재생성**

`build_graph()`를 엔드포인트 안에서 매번 호출하면 체크포인터가 재초기화되어 세션 기억이 사라집니다. 애플리케이션 수준의 전역 변수나 `@app.on_event("startup")`으로 한 번만 초기화하세요.

**4. SSE 응답에 `media_type="text/event-stream"` 누락**

이를 설정하지 않으면 브라우저가 스트리밍을 처리하지 못하고 전체 응답을 기다립니다.

**5. nginx 앞에서 버퍼링 문제**

프로덕션에서 nginx를 reverse proxy로 사용할 때 `proxy_buffering off;`를 설정하거나 응답 헤더에 `X-Accel-Buffering: no`를 추가해야 합니다.

---

## ✅ 셀프 체크

- [ ] FastAPI 앱이 `/health`, `/graph/chat`, `/graph/stream` 엔드포인트로 실행된다.
- [ ] `StreamingResponse`가 SSE 형식으로 토큰을 전달한다.
- [ ] 동일한 `thread_id`로 두 번 요청하면 이전 대화 내용을 기억한다.
- [ ] CORS 미들웨어가 설정되어 있다.
- [ ] 환경 변수는 `.env` 파일로 관리된다.
- [ ] 테스트 클라이언트로 스트리밍 응답을 확인했다.

---

## 🔗 참고 자료

- [FastAPI 공식 문서](https://fastapi.tiangolo.com)
- [FastAPI — StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [LangChain Async API](https://python.langchain.com/docs/concepts/async)
- [LangGraph Streaming (Phase 23)](../03-langgraph-core/23-streaming.md)

---

## 🔗 네비게이션

| 이전 | 현재 | 다음 |
|------|------|------|
| [Phase 41: 보안과 가드레일](41-security-guardrails.md) | **Phase 42: FastAPI로 배포** | [Phase 43: LangGraph Platform](43-deploy-langgraph-platform.md) |
