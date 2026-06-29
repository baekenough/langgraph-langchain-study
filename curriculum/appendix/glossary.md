# 용어집 (Glossary)

LangChain과 LangGraph를 학습하면서 자주 마주치는 핵심 용어를 한국어로 정의합니다.  
각 용어 옆의 링크를 클릭하면 관련 페이즈로 이동합니다.

> 알파벳 순서로 정렬되어 있습니다. 한국어 번역이 있는 경우 함께 표기했습니다.

---

## A

### Agent (에이전트)
LLM이 어떤 행동을 취할지 스스로 결정하는 시스템. 도구(Tool)를 사용하여 외부 세계와 상호작용하고, 목표를 달성할 때까지 추론-행동-관찰 루프를 반복합니다.  
→ [Phase 27: ReAct 에이전트](../04-agents/27-react-agent.md)

### Agentic RAG (에이전틱 RAG)
단순한 단일 검색이 아니라, 에이전트가 검색 전략을 스스로 결정하고 필요하면 재검색하거나 질문을 분해하여 RAG를 수행하는 방식.  
→ [Phase 34: 에이전틱 RAG](../04-agents/34-agentic-rag.md)

### add_messages (리듀서 함수)
LangGraph에서 메시지 리스트를 덮어쓰지 않고 **누적(append)** 하는 내장 리듀서. `Annotated[list, add_messages]`로 타입을 선언하면 적용됩니다.  
→ [Phase 19: 상태와 리듀서](../03-langgraph-core/19-state-reducers.md)

---

## C

### ChatModel (채팅 모델)
메시지 리스트를 입력받아 메시지를 반환하는 LLM 인터페이스. `BaseChatModel`을 상속하며, LangChain이 지원하는 모든 채팅 LLM의 공통 인터페이스입니다.  
예: `ChatOpenAI`, `ChatAnthropic`, `ChatOpenAI(base_url="https://openrouter.ai/api/v1")`.  
→ [Phase 04: 채팅 모델과 메시지](../01-langchain-core/04-chat-models-messages.md)

### Checkpointer (체크포인터)
LangGraph 그래프의 상태(State)를 저장소에 기록하여 대화 이력과 재시작을 지원하는 컴포넌트. `MemorySaver`(메모리), `SqliteSaver`(SQLite), `PostgresSaver`(PostgreSQL) 등이 있습니다.  
→ [Phase 22: 영속성과 체크포인터](../03-langgraph-core/22-persistence-checkpointers.md)

### Chunking (청킹 / 텍스트 분할)
긴 문서를 임베딩하고 검색하기 위해 작은 조각(청크)으로 나누는 과정. 청크 크기와 오버랩이 RAG 품질에 큰 영향을 줍니다.  
→ [Phase 12: 텍스트 분할](../02-rag/12-text-splitting.md)

### Conditional Edge (조건부 엣지)
LangGraph에서 현재 상태(State)를 검사하여 다음 노드를 동적으로 결정하는 엣지. `add_conditional_edges()`로 등록합니다.  
→ [Phase 20: 노드, 엣지, 라우팅](../03-langgraph-core/20-nodes-edges-routing.md)

### Context Window (컨텍스트 윈도우)
LLM이 한 번에 처리할 수 있는 최대 토큰 수. 모델마다 다르며(예: 128K, 200K), RAG에서 검색 청크 수를 결정할 때 중요한 제약 조건입니다.

---

## D

### Document (문서 객체)
LangChain에서 텍스트와 메타데이터를 함께 담는 기본 자료 구조. `page_content: str`과 `metadata: dict`로 구성됩니다.  
→ [Phase 11: 문서 로더](../02-rag/11-document-loaders.md)

### Document Loader (문서 로더)
PDF, 웹 페이지, CSV, 데이터베이스 등 다양한 소스에서 `Document` 객체를 읽어오는 LangChain 컴포넌트. `load()` 또는 `lazy_load()`로 문서를 가져옵니다.  
→ [Phase 11: 문서 로더](../02-rag/11-document-loaders.md)

---

## E

### Edge (엣지)
LangGraph의 그래프에서 노드 간 연결을 나타냅니다. 일반 엣지는 항상 같은 다음 노드로 이동하며, 조건부 엣지는 상태에 따라 분기합니다.  
→ [Phase 20: 노드, 엣지, 라우팅](../03-langgraph-core/20-nodes-edges-routing.md)

### Embedding (임베딩)
텍스트를 고차원 벡터(숫자 배열)로 변환하는 기술. 의미적으로 유사한 텍스트는 벡터 공간에서 가까운 위치에 놓입니다. RAG에서 문서와 질문을 동일한 공간으로 변환하여 유사도를 계산합니다.  
이 커리큘럼에서는 OpenAI `text-embedding-3-small`을 사용합니다.  
→ [Phase 13: 임베딩](../02-rag/13-embeddings.md)

### Evaluator (평가자)
LangSmith에서 LLM 응답의 품질을 자동으로 채점하는 함수. 정확도, 관련성, 독성 등 다양한 기준으로 평가할 수 있습니다.  
→ [Phase 36: LangSmith 평가](../05-production/36-langsmith-evaluation.md)

---

## G

### Guardrail (가드레일)
LLM이 유해하거나 규정 위반인 응답을 생성하지 않도록 제어하는 안전 장치. 입력/출력 필터링, 프롬프트 인젝션 방어 등을 포함합니다.  
→ [Phase 41: 보안과 가드레일](../05-production/41-security-guardrails.md)

---

## H

### Human-in-the-Loop (HITL, 인간 개입)
LangGraph 그래프 실행 도중 특정 시점에 사람이 확인하고 승인/거부/수정할 수 있도록 실행을 일시 중단하는 패턴.  
→ [Phase 24: Human-in-the-Loop](../03-langgraph-core/24-human-in-the-loop.md)

---

## I

### interrupt (인터럽트)
LangGraph에서 그래프 실행을 일시 중단하고 사람의 입력을 기다리는 함수. `interrupt(value)` 형태로 노드 내부에서 호출합니다. 체크포인터가 설정되어 있어야 작동합니다.  
→ [Phase 24: Human-in-the-Loop](../03-langgraph-core/24-human-in-the-loop.md)

---

## L

### LCEL (LangChain Expression Language)
LangChain 컴포넌트를 `|` 연산자로 연결하여 파이프라인을 구성하는 방법. 스트리밍, 비동기, 배치 처리를 통일된 인터페이스로 지원합니다.  
예: `chain = prompt | llm | output_parser`  
→ [Phase 07: LCEL과 Runnable](../01-langchain-core/07-lcel-runnables.md)

### LangSmith (랭스미스)
LangChain이 제공하는 LLM 애플리케이션 관측(Observability) 플랫폼. 각 LLM 호출의 입력/출력/지연시간/비용을 추적하고, 데이터셋 기반 평가를 지원합니다.  
→ [Phase 35: LangSmith 트레이싱](../05-production/35-langsmith-tracing.md), [Phase 36: LangSmith 평가](../05-production/36-langsmith-evaluation.md)

---

## M

### Message (메시지)
ChatModel과 대화할 때 사용하는 기본 단위. 역할(role)에 따라 `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage` 등으로 구분됩니다.  
→ [Phase 04: 채팅 모델과 메시지](../01-langchain-core/04-chat-models-messages.md)

### MessagesState (메시지 상태)
`messages: Annotated[list[BaseMessage], add_messages]` 필드를 기본 포함하는 LangGraph 내장 상태 클래스. 대부분의 챗봇/에이전트 그래프에서 기반 상태로 사용합니다.  
→ [Phase 18: LangGraph 소개와 StateGraph](../03-langgraph-core/18-langgraph-intro-stategraph.md)

---

## N

### Node (노드)
LangGraph 그래프에서 실제 작업을 수행하는 함수. 현재 `State`를 입력받아 업데이트된 상태 딕셔너리를 반환합니다. Python 함수 또는 `Runnable`을 노드로 등록할 수 있습니다.  
→ [Phase 20: 노드, 엣지, 라우팅](../03-langgraph-core/20-nodes-edges-routing.md)

---

## O

### OpenRouter (오픈라우터)
다양한 LLM 제공자(OpenAI, Anthropic, Meta, Google 등)의 모델을 단일 API 엔드포인트로 접근할 수 있는 통합 플랫폼.  
이 커리큘럼에서는 OpenAI Chat Completions 호환 API를 사용하여 `ChatOpenAI(base_url="https://openrouter.ai/api/v1")`로 초기화합니다.  
→ [Phase 03: 모델과 API 키](../00-foundations/03-models-and-keys.md)

### OutputParser (출력 파서)
LLM의 텍스트 출력을 구조화된 형태(Python 객체, JSON, 리스트 등)로 변환하는 컴포넌트.  
→ [Phase 06: 출력 파싱](../01-langchain-core/06-output-parsing.md)

---

## P

### Persistence (영속성)
LangGraph에서 그래프 실행 상태를 체크포인터를 통해 저장하고, 나중에 같은 `thread_id`로 재개할 수 있는 기능. 대화 이력 유지와 장애 복구에 사용됩니다.  
→ [Phase 22: 영속성과 체크포인터](../03-langgraph-core/22-persistence-checkpointers.md)

### Plan-and-Execute (계획 후 실행)
복잡한 작업을 먼저 단계별 계획으로 분해하고, 각 단계를 순서대로 실행하는 에이전트 패턴. 단일 ReAct 루프보다 장기 작업에 적합합니다.  
→ [Phase 30: Plan-and-Execute](../04-agents/30-plan-and-execute.md)

### PromptTemplate (프롬프트 템플릿)
변수를 포함한 프롬프트 문자열 템플릿. `{variable}` 형태의 플레이스홀더를 실행 시점에 실제 값으로 채웁니다. `ChatPromptTemplate`은 여러 메시지를 포함하는 대화형 템플릿입니다.  
→ [Phase 05: 프롬프트 템플릿](../01-langchain-core/05-prompt-templates.md)

---

## R

### RAG (Retrieval-Augmented Generation, 검색 증강 생성)
LLM이 답변을 생성할 때 외부 문서에서 관련 내용을 검색하여 컨텍스트로 제공하는 방식. 환각(hallucination)을 줄이고 최신 정보를 활용할 수 있습니다.  
→ [Phase 15: 리트리버와 기본 RAG](../02-rag/15-retrievers-basic-rag.md)

### ReAct (리액트)
**Re**asoning + **Act**ing의 약자. LLM이 생각(Thought) → 행동(Action) → 관찰(Observation) 루프를 반복하며 문제를 해결하는 에이전트 패턴.  
→ [Phase 27: ReAct 에이전트](../04-agents/27-react-agent.md)

### Reducer (리듀서)
LangGraph에서 노드가 반환한 업데이트를 현재 State에 어떻게 병합할지 결정하는 함수. 기본값은 마지막 값으로 덮어쓰기이며, `add_messages`는 누적하는 리듀서입니다.  
→ [Phase 19: 상태와 리듀서](../03-langgraph-core/19-state-reducers.md)

### Retriever (리트리버)
쿼리 문자열을 받아 관련 `Document` 리스트를 반환하는 인터페이스. 벡터 스토어, BM25, 앙상블 등 다양한 검색 방식을 통일된 인터페이스로 감쌉니다.  
→ [Phase 15: 리트리버와 기본 RAG](../02-rag/15-retrievers-basic-rag.md)

### Runnable (러너블)
LangChain의 핵심 인터페이스. `invoke()`, `ainvoke()`, `stream()`, `astream()`, `batch()` 메서드를 표준으로 제공합니다. `|` 연산자로 체인을 구성할 수 있습니다.  
→ [Phase 07: LCEL과 Runnable](../01-langchain-core/07-lcel-runnables.md)

---

## S

### State (상태)
LangGraph 그래프에서 노드 간에 공유되는 데이터 구조. 보통 `TypedDict` 또는 `dataclass`로 정의하며, 각 필드에 리듀서를 지정할 수 있습니다.  
→ [Phase 18: LangGraph 소개와 StateGraph](../03-langgraph-core/18-langgraph-intro-stategraph.md), [Phase 19: 상태와 리듀서](../03-langgraph-core/19-state-reducers.md)

### StateGraph (상태 그래프)
LangGraph의 핵심 클래스. 노드(Node)와 엣지(Edge)로 구성된 방향성 그래프를 정의하고, `compile()`로 실행 가능한 앱으로 변환합니다.  
→ [Phase 18: LangGraph 소개와 StateGraph](../03-langgraph-core/18-langgraph-intro-stategraph.md)

### Store (스토어 / 장기 메모리)
LangGraph에서 여러 `thread_id`(대화 세션)를 넘나들며 지속되는 장기 메모리 저장소. `InMemoryStore`, `AsyncPostgresStore` 등을 사용합니다.  
체크포인터가 단일 대화의 이력을 저장하는 반면, Store는 사용자 선호도 등 세션을 초월하는 정보를 저장합니다.  
→ [Phase 26: 메모리 스토어](../03-langgraph-core/26-memory-store.md)

### Streaming (스트리밍)
LLM이 응답을 완전히 생성하기 전에 토큰 단위로 점진적으로 반환하는 방식. 사용자 경험을 크게 향상시킵니다. LangChain에서는 `.stream()`, LangGraph에서는 `.astream_events()`를 사용합니다.  
→ [Phase 08: 스트리밍과 비동기](../01-langchain-core/08-streaming-async.md), [Phase 23: 스트리밍](../03-langgraph-core/23-streaming.md)

### Structured Output (구조화된 출력)
LLM이 자유 텍스트 대신 Pydantic 모델이나 JSON Schema에 맞는 형식으로 출력을 생성하도록 강제하는 기법. `model.with_structured_output(MySchema)`로 사용합니다.  
→ [Phase 10: 구조화된 출력](../01-langchain-core/10-structured-output.md)

### Subgraph (서브그래프)
하나의 LangGraph 그래프 안에 포함되는 또 다른 그래프. 복잡한 워크플로우를 모듈화하거나 독립적인 팀이 각자의 서브그래프를 개발할 때 사용합니다.  
→ [Phase 25: 서브그래프](../03-langgraph-core/25-subgraphs.md)

### Supervisor (슈퍼바이저)
멀티에이전트 시스템에서 사용자 요청을 받아 적절한 하위 에이전트에 작업을 분배하고 결과를 통합하는 조율 에이전트.  
→ [Phase 32: 멀티에이전트 슈퍼바이저](../04-agents/32-multiagent-supervisor.md)

### Swarm (스웜)
중앙 슈퍼바이저 없이 에이전트들이 peer-to-peer로 협력하는 멀티에이전트 패턴. LangGraph에서는 `handoff` 메커니즘을 통해 에이전트 간 작업 이전을 구현합니다.  
→ [Phase 33: 멀티에이전트 협업](../04-agents/33-multiagent-collaboration.md)

---

## T

### Thread (스레드)
LangGraph에서 하나의 대화 세션을 식별하는 단위. `{"configurable": {"thread_id": "..."}}` 형태로 체크포인터에 전달하면, 같은 `thread_id`로 호출할 때마다 이전 상태를 이어받습니다.  
→ [Phase 22: 영속성과 체크포인터](../03-langgraph-core/22-persistence-checkpointers.md)

### Tool (툴 / 도구)
에이전트가 외부 세계와 상호작용하기 위해 호출하는 함수. 웹 검색, 데이터베이스 조회, 계산기 등을 툴로 정의합니다. `@tool` 데코레이터나 `StructuredTool`로 생성합니다.  
→ [Phase 09: 툴 호출](../01-langchain-core/09-tool-calling.md), [Phase 29: 툴 설계 패턴](../04-agents/29-tool-design-patterns.md)

### Tool Calling (툴 호출)
LLM이 텍스트 대신 특정 함수 호출 명령(함수명 + 인자)을 출력하는 기능. OpenAI의 Function Calling을 일반화한 개념으로, 대부분의 최신 LLM이 지원합니다.  
→ [Phase 09: 툴 호출](../01-langchain-core/09-tool-calling.md)

### ToolNode (툴 노드)
LangGraph에서 LLM이 요청한 툴 호출을 실행하는 내장 노드 클래스. `AIMessage`에 포함된 `tool_calls`를 자동으로 처리하고 `ToolMessage`를 반환합니다.  
→ [Phase 27: ReAct 에이전트](../04-agents/27-react-agent.md)

### Trace (트레이스)
LangSmith에서 LLM 호출의 전체 실행 기록. 입력, 출력, 지연시간, 비용, 메타데이터를 포함하며 중첩된 실행(span)을 계층으로 시각화합니다.  
→ [Phase 35: LangSmith 트레이싱](../05-production/35-langsmith-tracing.md)

---

## V

### Vector Store (벡터 스토어)
임베딩 벡터를 저장하고 유사도 검색(ANN)을 수행하는 데이터베이스. Chroma, FAISS, Pinecone, pgvector 등이 있습니다. 이 커리큘럼에서는 로컬 개발에 ChromaDB를 주로 사용합니다.  
→ [Phase 14: 벡터 스토어](../02-rag/14-vector-stores.md)

---

*용어가 추가로 필요하거나 설명이 부정확한 경우 이슈로 알려주세요.*
