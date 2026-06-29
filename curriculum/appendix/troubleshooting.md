# 트러블슈팅 (Troubleshooting)

LangChain/LangGraph 학습 중 자주 마주치는 에러와 해결 방법을 정리했습니다.  
증상 → 원인 → 해결 순서로 정리되어 있습니다.

---

## API 키 관련

### OPENROUTER_API_KEY와 OPENAI_API_KEY 혼동

**증상**
```
openai.AuthenticationError: Incorrect API key provided: sk-or-...
```
또는
```
openai.AuthenticationError: No API key provided.
```

**원인**  
OpenRouter 채팅 모델을 초기화할 때 `OPENAI_API_KEY`를 사용하거나, 반대로 OpenAI 임베딩에 OpenRouter 키를 사용하면 발생합니다.

**해결**
```python
# 채팅 모델 → OPENROUTER_API_KEY 사용
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),  # OpenRouter 키
)

# 임베딩 → OPENAI_API_KEY 사용
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),  # OpenAI 키
)
```

---

### API 키가 None으로 읽힘

**증상**
```
openai.AuthenticationError: No API key provided.
```

**원인**  
`.env` 파일이 없거나, `load_dotenv()`를 호출하지 않았거나, `.env` 파일의 위치가 다릅니다.

**해결**
```python
from dotenv import load_dotenv
import os

# 스크립트 최상단에서 호출해야 합니다
load_dotenv()

# 키가 제대로 로드되었는지 확인
print(os.getenv("OPENROUTER_API_KEY"))  # None이면 .env 위치 문제
```

```bash
# .env 파일이 프로젝트 루트에 있는지 확인
ls -la .env
cat .env | head -3
```

---

### OpenRouter에서 임베딩 호출 실패

**증상**
```
openai.BadRequestError: 404 Not Found
# 또는
NotImplementedError: Embeddings are not supported by this provider
```

**원인**  
OpenRouter는 채팅 완성(Chat Completions) API만 지원합니다. 임베딩 엔드포인트(`/embeddings`)를 OpenRouter로 호출하면 실패합니다.

**해결**  
임베딩은 반드시 직접 OpenAI API를 사용해야 합니다.

```python
# 잘못된 방법
embeddings = OpenAIEmbeddings(
    base_url="https://openrouter.ai/api/v1",  # OpenRouter로 임베딩 시도 → 실패
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# 올바른 방법
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),  # OpenAI 직접 사용
)
```

---

## Import 에러

### ModuleNotFoundError

**증상**
```
ModuleNotFoundError: No module named 'langchain_openai'
ModuleNotFoundError: No module named 'langchain_chroma'
```

**원인**  
패키지가 설치되지 않았거나, 가상 환경이 활성화되지 않았습니다.

**해결**
```bash
# UV로 패키지 설치
uv add langchain-openai langchain-chroma langchain-community

# 또는 pyproject.toml에 추가 후
uv sync

# pip를 사용하는 경우
pip install langchain-openai langchain-chroma
```

---

### ImportError: cannot import name '...'

**증상**
```
ImportError: cannot import name 'HumanMessage' from 'langchain.schema'
ImportError: cannot import name 'ChatOpenAI' from 'langchain.chat_models'
```

**원인**  
LangChain v0.1+ 이후 모듈 경로가 변경되었습니다. 오래된 튜토리얼의 import 경로를 그대로 사용하면 발생합니다.

**해결**
```python
# 구버전 (0.0.x) → 신버전 (0.1+, 0.2+, 0.3+) 경로 변경

# 메시지
# 구버전: from langchain.schema import HumanMessage
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ChatOpenAI
# 구버전: from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

# 프롬프트
# 구버전: from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# 출력 파서
# 구버전: from langchain.schema import StrOutputParser
from langchain_core.output_parsers import StrOutputParser

# 커뮤니티 통합 (문서 로더 등)
from langchain_community.document_loaders import PyPDFLoader
```

---

### langchain_chroma import 에러

**증상**
```
ModuleNotFoundError: No module named 'langchain_chroma'
# 또는
ImportError: cannot import name 'Chroma' from 'langchain.vectorstores'
```

**해결**
```bash
uv add langchain-chroma chromadb
```

```python
# 올바른 import
from langchain_chroma import Chroma
```

---

## UV 관련

### .venv가 인식되지 않음

**증상**  
VSCode에서 Python 인터프리터가 `.venv`를 찾지 못하거나, `import langchain`이 실패합니다.

**해결**
```bash
# 프로젝트 루트에서 UV 환경 초기화
uv init

# 의존성 설치
uv sync --dev

# 가상 환경 경로 확인
uv run python -c "import sys; print(sys.executable)"
# 출력: /path/to/project/.venv/bin/python
```

VSCode에서: `Cmd+Shift+P` → "Python: Select Interpreter" → `.venv/bin/python` 선택

---

### uv sync 실패

**증상**
```
error: Failed to download and build `package-name`
```

**해결**
```bash
# 캐시 초기화 후 재시도
uv cache clean
uv sync --dev

# Python 버전 명시
uv python pin 3.11
uv sync --dev
```

---

### uv run vs 직접 실행 차이

**증상**  
`python script.py`는 실패하는데 `uv run python script.py`는 성공합니다.

**원인**  
시스템 Python과 UV 관리 가상 환경의 Python이 다릅니다.

**해결**
```bash
# 항상 UV로 실행하거나
uv run python script.py
uv run jupyter notebook

# 또는 가상 환경을 직접 활성화
source .venv/bin/activate
python script.py
```

---

## LangGraph 관련

### RecursionError / recursion_limit 초과

**증상**
```
langgraph.errors.GraphRecursionError: Recursion limit of 25 reached
```

**원인**  
그래프가 종료 조건 없이 계속 순환하거나, 예상보다 많은 반복이 발생했습니다.

**해결**
```python
# 재귀 한도 늘리기
config = {"recursion_limit": 50}
graph.invoke(input, config=config)

# 또는 종료 조건 확인
def should_continue(state):
    # 최대 반복 횟수 체크
    if state.get("iterations", 0) > 10:
        return END
    # tool_calls 없으면 종료
    if not state["messages"][-1].tool_calls:
        return END
    return "tools"
```

---

### interrupt가 동작하지 않음

**증상**  
`interrupt()` 호출 후 그래프가 멈추지 않고 계속 실행됩니다.

**원인**  
체크포인터가 설정되지 않았습니다. `interrupt`는 체크포인터 없이는 동작하지 않습니다.

**해결**
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
# compile() 시 반드시 checkpointer 지정
graph = graph_builder.compile(checkpointer=checkpointer)

# thread_id도 필수
config = {"configurable": {"thread_id": "session-1"}}
graph.invoke(input, config=config)
```

---

### State 업데이트가 반영되지 않음

**증상**  
노드에서 상태를 업데이트했는데 다음 노드에서 변경이 반영되지 않습니다.

**원인**  
노드 함수가 업데이트할 키만 포함한 딕셔너리를 반환해야 합니다. 전체 State를 반환하면 안 됩니다.

**해결**
```python
# 잘못된 방법: 전체 상태 반환 시도
def my_node(state: State) -> State:
    state["key"] = "new_value"
    return state  # 이렇게 하면 안 됩니다

# 올바른 방법: 변경할 키만 포함한 딕셔너리 반환
def my_node(state: State) -> dict:
    return {"key": "new_value"}  # 변경할 필드만 반환
```

---

### add_messages 없이 메시지가 덮어써짐

**증상**  
대화를 여러 번 해도 마지막 메시지만 남습니다.

**원인**  
`messages` 필드에 `add_messages` 리듀서를 지정하지 않아 덮어쓰기가 됩니다.

**해결**
```python
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    # Annotated[list, add_messages]로 선언해야 누적됩니다
    messages: Annotated[list, add_messages]  # 올바른 방법
    # messages: list  # 이렇게 하면 덮어써짐
```

---

## LangSmith 관련

### 트레이싱이 기록되지 않음

**증상**  
코드를 실행해도 LangSmith 대시보드에 아무것도 나타나지 않습니다.

**해결**
```bash
# .env 확인
LANGSMITH_TRACING=true   # "true" 문자열이어야 합니다 (True 아님)
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=my-project  # 프로젝트명 (없으면 default)
```

```python
# 코드에서 직접 확인
import os
print("Tracing:", os.getenv("LANGSMITH_TRACING"))
print("Key:", os.getenv("LANGSMITH_API_KEY", "NOT SET"))
print("Project:", os.getenv("LANGSMITH_PROJECT", "default"))

# load_dotenv()가 코드 최상단에 있는지 확인
from dotenv import load_dotenv
load_dotenv()  # 이 줄이 import langchain보다 먼저 와야 합니다
```

---

### LangSmith 인증 에러

**증상**
```
httpx.HTTPStatusError: Client error '401 Unauthorized'
```

**해결**  
`LANGSMITH_API_KEY`의 값이 올바른지 확인합니다.  
[smith.langchain.com](https://smith.langchain.com) → Settings → API Keys에서 키를 재발급받으세요.

---

## 비동기(Async) 관련

### RuntimeError: This event loop is already running

**증상**
```
RuntimeError: This event loop is already running.
```

**원인**  
Jupyter Notebook이나 이미 실행 중인 이벤트 루프에서 `asyncio.run()`을 호출했습니다.

**해결**
```python
# Jupyter Notebook에서는 await를 직접 사용
result = await chain.ainvoke({"question": "..."})

# 또는 nest_asyncio 설치 후 사용
import nest_asyncio
nest_asyncio.apply()

import asyncio
asyncio.run(my_async_function())
```

```bash
uv add nest-asyncio
```

---

### async 함수를 sync처럼 호출하는 실수

**증상**
```
RuntimeWarning: coroutine 'ainvoke' was never awaited
```

**해결**
```python
# 비동기 함수는 await 필요
result = await chain.ainvoke({"question": "..."})  # 올바름

# 동기 함수 사용
result = chain.invoke({"question": "..."})  # 올바름

# 잘못된 방법 (await 없이 비동기 호출)
result = chain.ainvoke({"question": "..."})  # 잘못됨 → coroutine 객체만 반환
```

---

## VSCode 인터프리터 관련

### VSCode에서 import 에러가 표시됨 (실행은 됨)

**증상**  
에디터에서 빨간 밑줄이 표시되지만, 터미널에서 실행하면 정상 동작합니다.

**원인**  
VSCode가 UV 가상 환경의 Python을 인터프리터로 선택하지 않았습니다.

**해결**
1. `Cmd+Shift+P` (macOS) 또는 `Ctrl+Shift+P` (Windows/Linux)
2. "Python: Select Interpreter" 선택
3. `.venv/bin/python` (macOS/Linux) 또는 `.venv\Scripts\python.exe` (Windows) 선택

또는 프로젝트 루트에 `.vscode/settings.json` 생성:
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"
}
```

---

## Rate Limit / 타임아웃

### RateLimitError

**증상**
```
openai.RateLimitError: You exceeded your current quota
# 또는
openai.RateLimitError: Rate limit exceeded
```

**해결**
```python
# 재시도 로직 추가
llm = llm.with_retry(
    retry_if_exception_type=(Exception,),
    wait_exponential_jitter=True,
    stop_after_attempt=3,
)

# 또는 배치 처리 시 딜레이 추가
import time

for doc in documents:
    result = chain.invoke(doc)
    time.sleep(0.5)  # 요청 간격 추가
```

---

### 타임아웃 에러

**증상**
```
openai.Timeout: Request timed out.
httpx.ConnectTimeout
```

**해결**
```python
# 타임아웃 시간 늘리기
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    timeout=60,  # 초 단위, 기본값 보통 10초
    max_retries=2,
)
```

---

## 버전 관련

### LangChain Deprecation Warning

**증상**
```
LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17...
```

**원인**  
오래된 `LLMChain`, `ConversationChain` 등의 클래스가 더 이상 권장되지 않습니다.

**해결**  
LCEL 방식으로 전환합니다.

```python
# 구버전 방식 (deprecated)
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(question="...")

# 신버전 방식 (LCEL)
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "..."})
```

---

### Pydantic v1 vs v2 경고

**증상**
```
PydanticUserError: `model_rebuild` is not supported in pydantic V1
# 또는
UserWarning: Field "model_fields" shadows an attribute in parent "BaseModel"
```

**해결**  
LangChain 0.3+는 Pydantic v2를 사용합니다. Pydantic v1 문법이 섞여 있는지 확인합니다.

```python
# Pydantic v2 방식
from pydantic import BaseModel, Field

class MyModel(BaseModel):
    name: str = Field(description="이름")
    age: int = Field(default=0, description="나이")

# 인스턴스화
obj = MyModel(name="홍길동", age=30)
# 딕셔너리로 변환
obj.model_dump()  # v2 (구버전: obj.dict())
```

---

## 체크리스트 (빠른 진단)

문제가 해결되지 않을 때 다음 순서로 확인하세요:

- [ ] `.env` 파일이 프로젝트 루트에 존재하는가?
- [ ] `.env`에 올바른 API 키가 입력되어 있는가?
- [ ] `load_dotenv()`가 코드 최상단에서 호출되고 있는가?
- [ ] `uv sync --dev`를 최근에 실행했는가?
- [ ] VSCode가 `.venv` 인터프리터를 사용하고 있는가?
- [ ] import 경로가 최신 LangChain(0.3+) 기준인가? (`langchain_core`, `langchain_openai` 등)
- [ ] OpenRouter는 채팅 모델에, OpenAI는 임베딩에 사용하고 있는가?
- [ ] LangGraph에서 `interrupt`를 사용한다면 체크포인터가 설정되어 있는가?
- [ ] LangSmith 트레이싱이 필요하다면 `LANGSMITH_TRACING=true`가 설정되어 있는가?

---

*이 목록에 없는 에러가 발생하면 [LangChain Discord](https://discord.gg/langchain) 또는 [GitHub Discussions](https://github.com/langchain-ai/langchain/discussions)에서 질문하세요.*
