# Phase 08: 스트리밍과 비동기

> 예상 소요시간: 60분 | 난이도: ★★★☆☆ | 선행 페이즈: [07-lcel-runnables](07-lcel-runnables.md)

---

## 🎯 학습 목표

- `.stream()`으로 토큰 단위 실시간 스트리밍을 구현할 수 있습니다.
- `astream()`, `ainvoke()`로 비동기 LLM 호출을 작성할 수 있습니다.
- `astream_events()`로 체인의 내부 이벤트를 추적할 수 있습니다.
- 동기와 비동기 호출을 언제 사용해야 하는지 판단할 수 있습니다.

---

## 📚 핵심 개념

### 1. 스트리밍이 필요한 이유

LLM은 응답을 한 번에 생성하지 않고 **토큰 단위로 순차 생성**합니다.
`invoke()`는 전체 응답이 완성될 때까지 기다립니다(수 초 이상 지연 가능).
`stream()`은 토큰이 생성되는 즉시 전달하여 **체감 응답 속도를 크게 향상**시킵니다.

```
invoke:  [----------- 3초 대기 -----------] → 전체 응답 한 번에 출력
stream:  [토] → [큰] → [단] → [위] → [로] → 점진적 출력 (체감 즉각)
```

### 2. 동기 vs 비동기

| 방식 | 메서드 | 사용 시점 |
|------|--------|----------|
| 동기 | `invoke`, `stream`, `batch` | 스크립트, 간단한 CLI 도구 |
| 비동기 | `ainvoke`, `astream`, `abatch` | FastAPI, WebSocket, 다중 동시 요청 |

비동기는 Python의 `asyncio` 이벤트 루프 위에서 동작합니다.
`async def` 함수 안에서 `await`와 함께 사용합니다.

### 3. astream_events

체인 내부 모든 단계에서 발생하는 이벤트를 스트리밍합니다.
`on_chain_start`, `on_chat_model_stream`, `on_chain_end` 등의 이벤트로 세밀한 추적이 가능합니다.

---

## 💻 코드 예제

### 예제 1: 동기 스트리밍 — stream()

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "자세한 설명을 제공합니다."),
    ("human", "{topic}을 설명해주세요."),
])

chain = prompt | llm | StrOutputParser()

print("=== 스트리밍 출력 ===")
full_response = ""
for chunk in chain.stream({"topic": "파이썬의 비동기 프로그래밍"}):
    print(chunk, end="", flush=True)
    full_response += chunk
print()  # 개행

print(f"\n총 길이: {len(full_response)}자")
```

### 예제 2: AIMessage 청크 스트리밍 (파서 없이)

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# 파서 없이 스트리밍 — AIMessageChunk 반환
for chunk in llm.stream([HumanMessage(content="1부터 5까지 천천히 세어주세요.")]):
    print(f"청크 타입: {type(chunk).__name__}, 내용: {repr(chunk.content)}")
```

### 예제 3: ainvoke — 비동기 단건 호출

```python
import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "간결하게 답변합니다."),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()

async def ask_question(question: str) -> str:
    """비동기로 LLM에 질문합니다."""
    return await chain.ainvoke({"question": question})

async def main() -> None:
    result = await ask_question("asyncio의 장점은?")
    print(result)

asyncio.run(main())
```

### 예제 4: abatch — 여러 질문 비동기 병렬 처리

```python
import os
import asyncio
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | llm
    | StrOutputParser()
)

questions = [
    {"question": "Python이란?"},
    {"question": "FastAPI란?"},
    {"question": "LangChain이란?"},
    {"question": "asyncio란?"},
    {"question": "Pydantic이란?"},
]

async def main() -> None:
    start = time.time()
    # 5개 질문을 비동기로 동시에 처리
    results = await chain.abatch(questions)
    elapsed = time.time() - start

    for q, r in zip(questions, results):
        print(f"Q: {q['question']}")
        print(f"A: {r[:80]}...")
        print()

    print(f"총 소요 시간: {elapsed:.2f}초 (동기 순차 실행 대비 약 {len(questions)}배 빠름)")

asyncio.run(main())
```

### 예제 5: astream — 비동기 스트리밍

```python
import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)

chain = (
    ChatPromptTemplate.from_messages([("human", "{topic}을 설명해주세요.")])
    | llm
    | StrOutputParser()
)

async def stream_response(topic: str) -> None:
    """비동기로 스트리밍하며 출력합니다."""
    print(f"=== {topic} ===")
    async for chunk in chain.astream({"topic": topic}):
        print(chunk, end="", flush=True)
    print()

async def main() -> None:
    # 두 스트림을 동시에 처리 (asyncio.gather)
    await asyncio.gather(
        stream_response("코루틴"),
        stream_response("이벤트 루프"),
    )

asyncio.run(main())
```

### 예제 6: astream_events — 이벤트 기반 추적

```python
import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | llm
    | StrOutputParser()
)

async def main() -> None:
    # astream_events로 체인 내부 이벤트 스트리밍
    async for event in chain.astream_events(
        {"question": "Python의 GIL이란?"},
        version="v2",  # v2 권장
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            # LLM이 토큰을 생성할 때마다 발생
            chunk_content = event["data"]["chunk"].content
            if chunk_content:
                print(chunk_content, end="", flush=True)

        elif kind == "on_chain_start":
            print(f"\n[시작] {event['name']}")

        elif kind == "on_chain_end":
            print(f"\n[완료] {event['name']}")

asyncio.run(main())
```

### 예제 7: FastAPI에서의 스트리밍 (실전 패턴)

```python
# 실행하려면: pip install fastapi uvicorn
# uvicorn filename:app --reload

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)

chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | llm
    | StrOutputParser()
)

@app.get("/stream")
async def stream_endpoint(question: str):
    """LLM 응답을 스트리밍으로 반환하는 엔드포인트."""
    async def generate():
        async for chunk in chain.astream({"question": question}):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")
```

---

## ✏️ 실습 과제

### 과제 1: 타이핑 효과 구현

`stream()`과 `time.sleep(0.02)`를 조합하여 LLM 응답이 타이핑되는 것처럼 보이는 콘솔 UI를 구현하세요.

### 과제 2: 비동기 번역기

5개 문장을 `abatch()`로 동시에 번역하고, 동기 순차 처리(`batch()`)와 소요 시간을 비교하세요.

### 과제 3: astream_events 필터링

`astream_events()`를 사용하여 체인 시작/종료 이벤트만 로그로 출력하고, 토큰 스트리밍은 별도로 수집하는 함수를 구현하세요.

### 과제 4: 동시 멀티 스트림

두 개의 다른 토픽에 대한 스트리밍 응답을 `asyncio.gather()`로 동시에 수집하고, 각 응답을 파일에 저장하세요.

---

## ⚠️ 흔한 함정

**1. 동기 컨텍스트에서 await 사용**

```python
# 오류 — 동기 함수에서 await 불가
def wrong():
    result = await chain.ainvoke(...)  # SyntaxError

# 올바름
async def correct():
    result = await chain.ainvoke(...)
```

**2. 주피터 노트북에서 asyncio.run() 사용**

주피터는 이미 이벤트 루프를 실행 중이므로 `asyncio.run()`이 실패합니다.

```python
# 주피터에서는
import nest_asyncio
nest_asyncio.apply()

# 또는 await 직접 사용
result = await chain.ainvoke(...)
```

**3. stream() 결과를 list로 변환하면 스트리밍 의미가 없음**

```python
# 안티패턴 — 전체를 모아서 반환하면 stream의 이점 없음
chunks = list(chain.stream(input))

# 스트리밍의 의미: 청크를 받는 즉시 처리
for chunk in chain.stream(input):
    send_to_client(chunk)  # 즉시 전송
```

**4. API 변동 주의**

> LangChain은 빠르게 발전합니다. 스트리밍/비동기 관련 최신 사양은 [공식 문서](https://python.langchain.com/docs/concepts/streaming/)를 확인하세요.

---

## ✅ 셀프 체크

- [ ] `stream()`으로 토큰 단위 스트리밍을 구현할 수 있다.
- [ ] `ainvoke()`를 `async def` 함수에서 `await`와 함께 사용할 수 있다.
- [ ] `abatch()`로 여러 요청을 비동기로 병렬 처리할 수 있다.
- [ ] `astream_events()`로 체인 내부 이벤트를 필터링할 수 있다.
- [ ] FastAPI에서 `StreamingResponse`와 `astream()`을 연결하는 패턴을 설명할 수 있다.

---

## 🔗 참고 자료

- [LangChain Streaming 개요](https://python.langchain.com/docs/concepts/streaming/)
- [astream_events 가이드](https://python.langchain.com/docs/how_to/streaming/#using-stream-events)
- [FastAPI StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)

---

← [Phase 07: LCEL과 Runnable](07-lcel-runnables.md) | [Phase 09: 툴 호출 기초](09-tool-calling.md) →
