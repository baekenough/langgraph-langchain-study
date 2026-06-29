# 치트시트 (Cheatsheet)

자주 쓰는 코드 스니펫을 복붙 가능한 형태로 정리했습니다.  
막혔을 때 빠르게 참조하세요.

---

## 환경 설정

### .env 파일 구성

```bash
# 채팅 모델 (필수)
OPENROUTER_API_KEY=sk-or-v1-...

# 임베딩 모델 (필수)
OPENAI_API_KEY=sk-...

# LangSmith 트레이싱 (선택)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=my-project

# Tavily 검색 (선택)
TAVILY_API_KEY=tvly-...
```

### 환경 변수 로드

```python
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일 로드

openrouter_key = os.getenv("OPENROUTER_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
```

---

## 채팅 모델 (OpenRouter)

### 기본 초기화

```python
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",       # OpenRouter 모델명
    base_url="https://openrouter.ai/api/v1",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    temperature=0,
)
```

### 다른 모델 사용 예

```python
from pydantic import SecretStr

# Claude (via OpenRouter)
llm_claude = ChatOpenAI(
    model="anthropic/claude-3.5-sonnet",
    base_url="https://openrouter.ai/api/v1",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
)

# Llama (via OpenRouter)
llm_llama = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",
    base_url="https://openrouter.ai/api/v1",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
)
```

### 기본 호출

```python
from langchain_core.messages import HumanMessage, SystemMessage

response = llm.invoke([
    SystemMessage(content="당신은 친절한 도우미입니다."),
    HumanMessage(content="Python이란 무엇인가요?"),
])
print(response.content)
```

---

## 임베딩 (OpenAI)

```python
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
)

# 단일 텍스트 임베딩
vector = embeddings.embed_query("검색할 텍스트")

# 여러 문서 임베딩
vectors = embeddings.embed_documents(["문서1", "문서2", "문서3"])
```

---

## 프롬프트 템플릿

### ChatPromptTemplate

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {domain} 전문가입니다."),
    ("human", "{question}"),
])

# 변수 채우기
formatted = prompt.invoke({"domain": "Python", "question": "리스트와 튜플의 차이는?"})
```

### MessagesPlaceholder (대화 이력 포함)

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 도움이 되는 어시스턴트입니다."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
```

---

## LCEL 체인

### 기본 체인 구성

```python
from langchain_core.output_parsers import StrOutputParser

chain = prompt | llm | StrOutputParser()
result = chain.invoke({"domain": "Python", "question": "리스트와 튜플의 차이는?"})
```

### RunnablePassthrough (입력 그대로 전달)

```python
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
result = chain.invoke("Python이란?")
```

### RunnableLambda (임의 함수 감싸기)

```python
from langchain_core.runnables import RunnableLambda

def to_upper(text: str) -> str:
    return text.upper()

chain = llm | StrOutputParser() | RunnableLambda(to_upper)
```

---

## 출력 파서

### StrOutputParser (문자열)

```python
from langchain_core.output_parsers import StrOutputParser

chain = llm | StrOutputParser()
```

### JsonOutputParser (JSON)

```python
from langchain_core.output_parsers import JsonOutputParser

chain = llm | JsonOutputParser()
```

### PydanticOutputParser

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float

parser = PydanticOutputParser(pydantic_object=Answer)
# format_instructions를 프롬프트에 포함해야 합니다
instructions = parser.get_format_instructions()
```

---

## 구조화된 출력 (with_structured_output)

```python
from pydantic import BaseModel, Field

class Joke(BaseModel):
    setup: str = Field(description="개그의 설정 부분")
    punchline: str = Field(description="펀치라인")

structured_llm = llm.with_structured_output(Joke)
joke = structured_llm.invoke("파이썬에 관한 개그를 만들어주세요")
print(joke.setup)
print(joke.punchline)
```

---

## 툴 정의와 바인딩

### @tool 데코레이터

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 반환합니다."""
    return f"{city}의 날씨는 맑음, 기온 22도입니다."

# 여러 줄 설명 (Pydantic schema 자동 생성)
@tool
def calculate(expression: str) -> float:
    """
    수학 표현식을 계산합니다.
    
    Args:
        expression: 계산할 수식 (예: "2 + 3 * 4")
    """
    return eval(expression)
```

### 모델에 툴 바인딩

```python
tools = [get_weather, calculate]
llm_with_tools = llm.bind_tools(tools)

response = llm_with_tools.invoke("서울 날씨 알려줘")
# response.tool_calls 확인
if response.tool_calls:
    print(response.tool_calls)
```

---

## RAG 파이프라인

### 문서 로드, 분할, 임베딩, 저장

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 1. 문서 로드
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# 2. 텍스트 분할
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)

# 3. 임베딩 + 벡터 스토어 저장
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)
```

### 기본 RAG 체인

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "다음 컨텍스트를 바탕으로 질문에 답하세요:\n\n{context}"),
    ("human", "{question}"),
])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("문서에서 가장 중요한 내용은?")
```

---

## 스트리밍

### 동기 스트리밍

```python
for chunk in llm.stream("긴 이야기를 해주세요"):
    print(chunk.content, end="", flush=True)
```

### 비동기 스트리밍

```python
import asyncio

async def stream_response():
    async for chunk in llm.astream("긴 이야기를 해주세요"):
        print(chunk.content, end="", flush=True)

asyncio.run(stream_response())
```

### 체인 스트리밍

```python
async for chunk in chain.astream({"question": "Python이란?"}):
    print(chunk, end="", flush=True)
```

---

## 최소 StateGraph (LangGraph)

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

# 상태 정의
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 노드 정의
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 그래프 조립
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# 실행
result = graph.invoke({"messages": [("user", "안녕하세요")]})
print(result["messages"][-1].content)
```

---

## MessagesState 사용

```python
from langgraph.graph import MessagesState

# MessagesState는 messages 필드를 기본 제공합니다
# TypedDict 직접 정의 없이 사용 가능

graph_builder = StateGraph(MessagesState)

def my_node(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

---

## 체크포인터 (영속성)

### MemorySaver (메모리 기반)

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# thread_id로 대화 세션 식별
config = {"configurable": {"thread_id": "user-001"}}

# 첫 번째 호출
graph.invoke({"messages": [("user", "제 이름은 홍길동입니다")]}, config=config)

# 같은 thread_id로 호출 → 이전 대화 맥락 유지
graph.invoke({"messages": [("user", "제 이름이 뭐라고 했죠?")]}, config=config)
```

### SqliteSaver (파일 기반)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = graph_builder.compile(checkpointer=checkpointer)
    # ...
```

---

## 조건부 엣지

```python
def should_continue(state: State) -> str:
    """마지막 메시지에 tool_calls가 있으면 tools로, 없으면 END로."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

graph_builder.add_conditional_edges(
    "chatbot",
    should_continue,
    {"tools": "tools", END: END},
)
```

---

## interrupt (Human-in-the-Loop)

```python
from langgraph.types import interrupt

def review_node(state: State):
    """사람의 검토가 필요한 노드."""
    # 실행을 중단하고 사람의 입력을 기다립니다
    user_input = interrupt({
        "question": "이 내용을 승인하시겠습니까?",
        "content": state["draft"],
    })
    
    if user_input["approved"]:
        return {"status": "approved"}
    else:
        return {"status": "rejected", "feedback": user_input.get("feedback", "")}

# interrupt 재개: Command(resume=값)으로 전달
from langgraph.types import Command

graph.invoke(
    Command(resume={"approved": True}),
    config=config,
)
```

---

## create_react_agent (내장 ReAct 에이전트)

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """웹에서 정보를 검색합니다."""
    return f"'{query}'에 대한 검색 결과..."

tools = [search]
agent = create_react_agent(llm, tools)

# 실행
result = agent.invoke({
    "messages": [("user", "서울의 현재 날씨를 검색해줘")]
})
```

---

## 에러 처리 (with_retry, with_fallbacks)

### with_retry

```python
from langchain_core.runnables import RunnableConfig

llm_with_retry = llm.with_retry(
    retry_if_exception_type=(Exception,),
    wait_exponential_jitter=True,
    stop_after_attempt=3,
)
```

### with_fallbacks

```python
# 기본 모델 실패 시 대체 모델로 전환
primary_llm = ChatOpenAI(model="openai/gpt-4o", ...)
fallback_llm = ChatOpenAI(model="openai/gpt-4o-mini", ...)

llm_with_fallback = primary_llm.with_fallbacks([fallback_llm])
```

---

## LangGraph 스트리밍 (astream_events)

```python
async def stream_graph_events():
    async for event in graph.astream_events(
        {"messages": [("user", "안녕하세요")]},
        config=config,
        version="v2",
    ):
        kind = event["event"]
        
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            print(chunk.content, end="", flush=True)

asyncio.run(stream_graph_events())
```

---

## LangSmith 트레이싱 설정

```bash
# .env에 추가
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=my-langchain-project
```

```python
# 코드에서 수동 설정 (옵션)
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "ls__..."
os.environ["LANGSMITH_PROJECT"] = "my-project"

# 이후 모든 LangChain/LangGraph 호출이 자동으로 트레이싱됩니다
```

---

## 자주 쓰는 임포트 모음

```python
# Core
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.tools import tool

# OpenAI / OpenRouter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# RAG
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

# LangGraph
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.types import interrupt, Command

# Pydantic
from pydantic import BaseModel, Field
from typing import Annotated
from typing_extensions import TypedDict
```
