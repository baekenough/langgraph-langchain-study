# Phase 34: Agentic RAG

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 120분 |
| 난이도 | ★★★★★ |
| 선행 학습 | Phase 27~33 (에이전트 파트 전체), Phase 15 (기본 RAG), Phase 16 (고급 RAG) |

---

## 🎯 학습 목표

- 기본 RAG와 Agentic RAG의 차이를 설명할 수 있습니다.
- retriever를 도구로 변환하여 에이전트에게 제공할 수 있습니다.
- 에이전트가 검색 여부와 쿼리를 스스로 결정하는 구조를 구현할 수 있습니다.
- 문서 관련성 평가 후 재검색(Corrective RAG)을 구현할 수 있습니다.
- Self-RAG의 개념을 이해하고 단순화된 버전을 구현할 수 있습니다.
- Part 2 RAG 지식과 Part 4 에이전트 그래프를 결합할 수 있습니다.

---

## 📚 핵심 개념

### 기본 RAG vs Agentic RAG

| 항목 | 기본 RAG (Phase 15) | Agentic RAG |
|------|---------------------|-------------|
| 검색 여부 | 항상 검색 | 에이전트가 필요 시에만 검색 |
| 쿼리 생성 | 사용자 입력 그대로 | 에이전트가 최적 쿼리 생성 |
| 재검색 | 없음 | 관련성 낮으면 재검색 |
| 여러 검색 | 불가 | 여러 번, 다른 쿼리로 검색 가능 |
| 제어 흐름 | 선형 (검색→생성) | 동적 (에이전트가 판단) |

### Corrective RAG 개념

문서 관련성이 낮으면 쿼리를 개선하여 재검색합니다:

```
사용자 질문
     │
     ▼
[에이전트 판단: 검색 필요?]
  아니오 → 바로 답변
  예 ──────────────────┐
                       ▼
                   [검색 실행]
                       │
                       ▼
                [관련성 평가] ← 각 문서가 질문과 관련있는가?
                       │
           관련 있음 ──┤
                       │ 관련 없음
                       ▼
                [쿼리 개선 후 재검색]
                       │
                       ▼
                   [답변 생성]
                       │
                       ▼
                  [환각 검사] ← 답변이 문서에 근거하는가?
                       │
              근거 있음 ┤
                        │ 근거 없음 → 재생성
                        ▼
                     [최종 답변]
```

---

## 💻 코드 예제

### 예제 1: Retriever를 도구로 변환

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# 임베딩: OpenAI 직접 사용 (OpenRouter는 임베딩 미지원)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"],
)

# 샘플 문서 생성
docs = [
    Document(
        page_content="LangGraph는 LangChain 위에서 동작하는 그래프 기반 에이전트 프레임워크입니다. "
                     "StateGraph로 복잡한 워크플로우를 구성할 수 있으며, 체크포인터로 상태를 유지합니다.",
        metadata={"source": "langgraph_intro.md"},
    ),
    Document(
        page_content="ReAct 패턴은 Reasoning과 Acting을 교차하는 에이전트 패턴입니다. "
                     "LangGraph에서는 create_react_agent로 간단히 구현할 수 있습니다.",
        metadata={"source": "react_pattern.md"},
    ),
    Document(
        page_content="RAG(Retrieval-Augmented Generation)는 외부 지식을 검색하여 LLM의 답변을 보완합니다. "
                     "Vector Store에 문서를 저장하고 유사도 검색으로 관련 문서를 찾습니다.",
        metadata={"source": "rag_basics.md"},
    ),
    Document(
        page_content="LangChain의 LCEL(LangChain Expression Language)은 체인을 선언적으로 구성하는 방법입니다. "
                     "파이프 연산자(|)로 Runnable 컴포넌트를 연결합니다.",
        metadata={"source": "lcel_guide.md"},
    ),
    Document(
        page_content="멀티에이전트 시스템에서 Supervisor 패턴은 중앙 에이전트가 Worker들을 조율합니다. "
                     "Swarm 패턴은 에이전트들이 수평적으로 핸드오프합니다.",
        metadata={"source": "multiagent.md"},
    ),
]

# Vector Store 생성
vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# Retriever를 도구로 변환
@tool
def search_knowledge_base(query: str) -> str:
    """내부 지식 베이스에서 관련 정보를 검색합니다.
    LangChain, LangGraph, RAG, 에이전트 관련 기술 문서를 검색할 수 있습니다.

    Args:
        query: 검색할 질문 또는 키워드. 구체적일수록 좋습니다.
    """
    retrieved_docs = retriever.invoke(query)
    if not retrieved_docs:
        return "관련 문서를 찾지 못했습니다."

    results = []
    for i, doc in enumerate(retrieved_docs, 1):
        source = doc.metadata.get("source", "unknown")
        results.append(f"[문서 {i}] (출처: {source})\n{doc.page_content}")

    return "\n\n".join(results)


# 에이전트 생성 — retriever가 도구로 제공됨
agent = create_react_agent(
    model=llm,
    tools=[search_knowledge_base],
    state_modifier=(
        "당신은 기술 문서 도우미입니다. 사용자의 질문에 답변할 때 "
        "필요하면 search_knowledge_base 도구로 관련 정보를 검색하세요. "
        "검색이 필요 없는 간단한 질문은 직접 답변해도 됩니다. "
        "항상 한국어로 답변하세요."
    ),
)

# 테스트 1: 검색이 필요한 질문
print("=== 검색 필요 ===")
r1 = agent.invoke({"messages": [("user", "LangGraph의 StateGraph가 뭔가요?")]})
print(r1["messages"][-1].content)

# 테스트 2: 검색 없이 답변 가능한 질문
print("\n=== 검색 불필요 ===")
r2 = agent.invoke({"messages": [("user", "안녕하세요! 오늘 날씨 어때요?")]})
print(r2["messages"][-1].content)
```

### 예제 2: Corrective RAG — 관련성 평가 + 재검색

```python
import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"],
)

# 지식 베이스 구성
docs = [
    Document(page_content="Python은 1991년에 귀도 반 로섬이 개발한 프로그래밍 언어입니다. 가독성이 높고 다목적으로 사용됩니다."),
    Document(page_content="머신러닝(ML)은 데이터로부터 패턴을 학습하는 AI의 하위 분야입니다. 지도학습, 비지도학습, 강화학습으로 분류됩니다."),
    Document(page_content="딥러닝은 인공 신경망을 사용하는 머신러닝 기법입니다. 이미지 인식, NLP 등에서 뛰어난 성능을 보입니다."),
    Document(page_content="트랜스포머 아키텍처는 2017년 'Attention Is All You Need' 논문에서 제안되었습니다. NLP 분야를 혁신했습니다."),
]

vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# ─── 스키마 ────────────────────────────────────────────────────────────────────

class RelevanceScore(BaseModel):
    """문서 관련성 평가 결과."""
    score: Literal["relevant", "irrelevant"] = Field(
        description="'relevant': 질문과 관련됨, 'irrelevant': 관련 없음"
    )
    reason: str = Field(description="판단 이유")


class HallucinationCheck(BaseModel):
    """환각 여부 평가."""
    is_grounded: bool = Field(description="답변이 제공된 문서에 근거하면 True")
    reason: str = Field(description="판단 이유")


class CRAGState(TypedDict):
    question: str
    documents: list[Document]
    filtered_docs: list[Document]
    generation: str
    query_rewritten: bool
    iteration: int


# ─── 노드 ──────────────────────────────────────────────────────────────────────

def retrieve_node(state: CRAGState) -> dict:
    """질문으로 문서를 검색합니다."""
    print(f"\n[Retrieve] 검색: '{state['question']}'")
    docs = retriever.invoke(state["question"])
    print(f"  → {len(docs)}개 문서 검색됨")
    return {"documents": docs}


relevance_llm = llm.with_structured_output(RelevanceScore)


def grade_documents_node(state: CRAGState) -> dict:
    """검색된 문서의 관련성을 평가합니다."""
    print(f"[Grade] {len(state['documents'])}개 문서 관련성 평가 중...")
    filtered = []

    for doc in state["documents"]:
        result = relevance_llm.invoke([
            ("system", "다음 질문과 문서의 관련성을 평가하세요."),
            ("human", f"질문: {state['question']}\n\n문서: {doc.page_content}"),
        ])
        if result.score == "relevant":
            filtered.append(doc)
            print(f"  ✓ 관련 문서: {doc.page_content[:50]}...")
        else:
            print(f"  ✗ 무관 문서: {doc.page_content[:50]}...")

    return {"filtered_docs": filtered}


def rewrite_query_node(state: CRAGState) -> dict:
    """쿼리를 개선하여 재검색 준비를 합니다."""
    print(f"[Rewrite] 쿼리 개선 중...")
    improved = llm.invoke([
        ("system", "사용자의 질문을 더 나은 검색 쿼리로 변환하세요. 간결하게 작성하세요."),
        ("human", f"원래 질문: {state['question']}\n개선된 검색 쿼리를 작성하세요:"),
    ])
    new_query = improved.content.strip()
    print(f"  → 개선된 쿼리: '{new_query}'")
    return {
        "question": new_query,
        "query_rewritten": True,
        "iteration": state.get("iteration", 0) + 1,
    }


def generate_node(state: CRAGState) -> dict:
    """관련 문서를 바탕으로 답변을 생성합니다."""
    context = "\n\n".join(doc.page_content for doc in state["filtered_docs"])
    print(f"\n[Generate] {len(state['filtered_docs'])}개 관련 문서로 답변 생성 중...")

    response = llm.invoke([
        ("system",
         "다음 문서들을 참고하여 사용자의 질문에 정확하게 답변하세요. "
         "문서에 없는 내용은 추측하지 마세요. 한국어로 답변하세요."),
        ("human", f"문서:\n{context}\n\n질문: {state['question']}"),
    ])
    return {"generation": response.content}


hallucination_llm = llm.with_structured_output(HallucinationCheck)


def check_hallucination_node(state: CRAGState) -> dict:
    """답변이 문서에 근거하는지 확인합니다."""
    print(f"[HalluCheck] 환각 여부 확인 중...")
    context = "\n\n".join(doc.page_content for doc in state["filtered_docs"])

    result = hallucination_llm.invoke([
        ("system", "답변이 제공된 문서에 근거하는지 평가하세요."),
        ("human",
         f"문서:\n{context}\n\n"
         f"답변:\n{state['generation']}\n\n"
         f"이 답변이 문서에 근거하나요?"),
    ])
    print(f"  → 근거 있음: {result.is_grounded}")
    return {}  # 상태 변경 없음, 라우팅만 위해 사용


# ─── 라우팅 ────────────────────────────────────────────────────────────────────

def route_after_grading(state: CRAGState) -> str:
    """관련 문서가 있으면 생성, 없으면 쿼리 재작성."""
    if not state["filtered_docs"]:
        if state.get("iteration", 0) >= 2:
            print("[경고] 2번 재시도했지만 관련 문서 없음. 최선의 답변 시도.")
            return "generate"
        print(f"  → 관련 문서 없음. 쿼리 재작성 필요.")
        return "rewrite"
    return "generate"


def route_after_hallucination_check(state: CRAGState) -> str:
    """환각 없으면 종료, 있으면 재생성."""
    context = "\n\n".join(doc.page_content for doc in state["filtered_docs"])
    result = hallucination_llm.invoke([
        ("system", "답변이 제공된 문서에 근거하는지 평가하세요."),
        ("human",
         f"문서:\n{context}\n\n답변:\n{state['generation']}\n\n이 답변이 문서에 근거하나요?"),
    ])

    if result.is_grounded:
        return END
    else:
        print("  → 환각 감지! 재생성 시도...")
        return "generate"


# ─── 그래프 빌드 ───────────────────────────────────────────────────────────────

graph = StateGraph(CRAGState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("grade", grade_documents_node)
graph.add_node("rewrite", rewrite_query_node)
graph.add_node("generate", generate_node)
graph.add_node("hallucination_check", check_hallucination_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", route_after_grading)
graph.add_edge("rewrite", "retrieve")
graph.add_edge("generate", "hallucination_check")
graph.add_conditional_edges("hallucination_check", route_after_hallucination_check)

crag = graph.compile()

# 테스트
print("=== Corrective RAG ===\n")
result = crag.invoke({
    "question": "트랜스포머 아키텍처는 언제 처음 소개되었나요?",
    "documents": [],
    "filtered_docs": [],
    "generation": "",
    "query_rewritten": False,
    "iteration": 0,
})

print(f"\n=== 최종 답변 ===")
print(result["generation"])
```

### 예제 3: Self-RAG — 에이전트가 검색 필요 여부를 판단

```python
import os
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"],
)

# 지식 베이스
docs = [
    Document(page_content="LangGraph의 MemorySaver는 인메모리 체크포인터로, 그래프 실행 상태를 저장합니다."),
    Document(page_content="LangGraph의 MessagesState는 메시지 리스트를 관리하는 기본 상태 타입입니다."),
]
vectorstore = InMemoryVectorStore.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})


class RetrievalDecision(BaseModel):
    """검색 필요 여부 판단."""
    needs_retrieval: bool = Field(
        description="외부 지식이 필요하면 True, 일반 지식으로 충분하면 False"
    )
    reason: str = Field(description="판단 이유")


class SelfRAGState(TypedDict):
    question: str
    context: str
    answer: str


retrieval_decision_llm = llm.with_structured_output(RetrievalDecision)


def router_node(state: SelfRAGState) -> dict:
    """검색이 필요한지 먼저 판단합니다 (Self-RAG 핵심)."""
    print(f"\n[Router] 질문 분석: '{state['question']}'")

    decision = retrieval_decision_llm.invoke([
        ("system",
         "사용자의 질문에 답변하기 위해 외부 문서 검색이 필요한지 판단하세요. "
         "LangGraph, LangChain 등 특정 기술 문서가 필요한 질문에는 True, "
         "일반적인 상식이나 간단한 질문에는 False를 반환하세요."),
        ("human", state["question"]),
    ])

    print(f"  → 검색 필요: {decision.needs_retrieval} ({decision.reason})")
    return {"context": "NEEDS_RETRIEVAL" if decision.needs_retrieval else "NO_RETRIEVAL"}


def retrieve_node(state: SelfRAGState) -> dict:
    print(f"[Retrieve] 관련 문서 검색 중...")
    docs = retriever.invoke(state["question"])
    context = "\n".join(doc.page_content for doc in docs) if docs else "관련 문서 없음"
    print(f"  → {len(docs)}개 문서 검색됨")
    return {"context": context}


def generate_with_context_node(state: SelfRAGState) -> dict:
    """검색된 문서를 바탕으로 답변을 생성합니다."""
    print(f"[Generate] 문서 기반 답변 생성 중...")
    response = llm.invoke([
        ("system", "다음 문서를 참고하여 질문에 답변하세요. 한국어로 답변하세요."),
        ("human", f"문서:\n{state['context']}\n\n질문: {state['question']}"),
    ])
    return {"answer": response.content}


def generate_direct_node(state: SelfRAGState) -> dict:
    """검색 없이 직접 답변을 생성합니다."""
    print(f"[Generate] 직접 답변 생성 중...")
    response = llm.invoke([
        ("system", "사용자의 질문에 직접 답변하세요. 한국어로 답변하세요."),
        ("human", state["question"]),
    ])
    return {"answer": response.content}


def route_by_retrieval_need(state: SelfRAGState) -> str:
    """검색 필요 여부에 따라 라우팅합니다."""
    if state.get("context") == "NEEDS_RETRIEVAL":
        return "retrieve"
    return "generate_direct"


graph = StateGraph(SelfRAGState)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate_with_context", generate_with_context_node)
graph.add_node("generate_direct", generate_direct_node)

graph.add_edge(START, "router")
graph.add_conditional_edges("router", route_by_retrieval_need)
graph.add_edge("retrieve", "generate_with_context")
graph.add_edge("generate_with_context", END)
graph.add_edge("generate_direct", END)

self_rag = graph.compile()

# 테스트 1: 검색이 필요한 질문
print("=== Self-RAG 테스트 1: 기술 문서 필요 ===")
r1 = self_rag.invoke({
    "question": "LangGraph의 MemorySaver는 어떤 역할을 하나요?",
    "context": "",
    "answer": "",
})
print(f"답변: {r1['answer']}")

# 테스트 2: 검색이 불필요한 질문
print("\n=== Self-RAG 테스트 2: 일반 지식으로 충분 ===")
r2 = self_rag.invoke({
    "question": "파이썬에서 리스트를 정렬하는 방법은?",
    "context": "",
    "answer": "",
})
print(f"답변: {r2['answer']}")
```

### 예제 4: 멀티-retriever Agentic RAG

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.vectorstores import InMemoryVectorStore
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"],
)

# 두 개의 분리된 지식 베이스
tech_docs = [
    Document(page_content="Python 3.12는 타입 파라미터 문법을 지원하고 성능이 향상되었습니다."),
    Document(page_content="FastAPI는 Python으로 빠른 API를 구축하는 현대적인 웹 프레임워크입니다."),
]

business_docs = [
    Document(page_content="2024년 AI 시장 규모는 2,000억 달러를 초과할 것으로 예측됩니다."),
    Document(page_content="생성형 AI는 콘텐츠 생성, 코드 작성, 고객 서비스 분야에서 활발히 활용됩니다."),
]

tech_store = InMemoryVectorStore.from_documents(tech_docs, embeddings)
business_store = InMemoryVectorStore.from_documents(business_docs, embeddings)

tech_retriever = tech_store.as_retriever(search_kwargs={"k": 1})
business_retriever = business_store.as_retriever(search_kwargs={"k": 1})


@tool
def search_tech_docs(query: str) -> str:
    """Python, FastAPI, 프로그래밍 언어 등 기술 문서를 검색합니다.

    Args:
        query: 검색할 기술 관련 키워드 또는 질문
    """
    docs = tech_retriever.invoke(query)
    if not docs:
        return "기술 문서에서 관련 정보를 찾지 못했습니다."
    return "\n\n".join(f"[기술] {doc.page_content}" for doc in docs)


@tool
def search_business_docs(query: str) -> str:
    """AI 시장 동향, 비즈니스 트렌드, 산업 분석 등 비즈니스 문서를 검색합니다.

    Args:
        query: 검색할 비즈니스 관련 키워드 또는 질문
    """
    docs = business_retriever.invoke(query)
    if not docs:
        return "비즈니스 문서에서 관련 정보를 찾지 못했습니다."
    return "\n\n".join(f"[비즈니스] {doc.page_content}" for doc in docs)


# 두 retriever를 도구로 가진 에이전트
agent = create_react_agent(
    model=llm,
    tools=[search_tech_docs, search_business_docs],
    state_modifier=(
        "당신은 기술 및 비즈니스 분야 전문 분석가입니다. "
        "기술 관련 질문은 search_tech_docs를, 비즈니스 관련 질문은 search_business_docs를 사용하세요. "
        "두 분야에 걸친 질문은 두 도구를 모두 활용하세요. "
        "항상 한국어로 답변하세요."
    ),
)

# 기술 + 비즈니스 통합 질문
print("=== 멀티-Retriever Agentic RAG ===")
result = agent.invoke({
    "messages": [(
        "user",
        "Python과 FastAPI를 활용한 AI 서비스 개발 트렌드와 "
        "현재 AI 시장 규모를 함께 분석해줘."
    )]
})
print(result["messages"][-1].content)
```

---

## ✏️ 실습 과제

### 과제 1: 웹 검색 + RAG 하이브리드
`search_knowledge_base` (내부 벡터스토어)와 `web_search` (외부 웹 검색 더미) 두 도구를 가진 에이전트를 만들고, 내부 지식 베이스에 답이 없으면 웹 검색으로 폴백하는 흐름을 구현하세요.

### 과제 2: Part 2 연결
Phase 16 (Advanced RAG)에서 배운 HyDE(Hypothetical Document Embeddings) 기법을 Agentic RAG에 통합해보세요. 에이전트가 검색 전에 가상의 답변을 생성하고, 그 가상 답변으로 검색하는 방식을 구현하세요.
참고: [../02-rag/16-advanced-rag.md](../02-rag/16-advanced-rag.md)

---

## ⚠️ 흔한 함정

**1. Retriever 도구의 docstring이 불명확**
에이전트가 어떤 도구를 언제 사용해야 할지 모릅니다:
```python
# 나쁜 예
@tool
def search(q: str) -> str:
    """검색합니다."""
    ...

# 좋은 예 — 어떤 내용을 검색하는지 명시
@tool
def search_product_catalog(query: str) -> str:
    """제품 카탈로그에서 상품 정보를 검색합니다.
    가격, 재고, 스펙 등 제품 관련 질문에 사용하세요."""
    ...
```

**2. 임베딩 모델 혼용**
인덱싱 시와 검색 시에 다른 임베딩 모델을 사용하면 검색 품질이 크게 저하됩니다.
항상 동일한 모델을 사용하세요.

**3. Corrective RAG에서 재검색 루프**
관련 문서가 계속 없으면 무한 재검색이 발생합니다.
반드시 최대 재시도 횟수를 상태에 추가하고 조건부 라우팅에서 확인하세요.

**4. 너무 많은 문서를 컨텍스트에 포함**
k=10으로 검색하면 컨텍스트가 길어져 LLM의 집중도가 낮아집니다.
k=2~4가 대부분의 경우에 적절합니다.

---

## ✅ 셀프 체크

- [ ] `retriever.as_retriever()`를 `@tool`로 감싸 에이전트에 제공할 수 있다
- [ ] Corrective RAG의 관련성 평가 → 재검색 흐름을 구현할 수 있다
- [ ] Self-RAG에서 에이전트가 검색 필요 여부를 스스로 판단하는 구조를 설명할 수 있다
- [ ] 여러 도메인별 retriever를 별도 도구로 제공하는 멀티-retriever 패턴을 구현할 수 있다
- [ ] 환각 여부를 Pydantic 구조화 출력으로 평가하는 방법을 안다
- [ ] Part 2 RAG 개념(벡터스토어, 임베딩)과 Part 4 에이전트를 결합할 수 있다

---

## 🔗 참고 자료

- [Agentic RAG 개요](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)
- [Corrective RAG (CRAG)](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_crag/)
- [Self-RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_self_rag/)
- [CRAG 논문 (Yan et al., 2024)](https://arxiv.org/abs/2401.15884)
- [Self-RAG 논문 (Asai et al., 2023)](https://arxiv.org/abs/2310.11511)
- 선행 학습: [Phase 15: 기본 RAG](../02-rag/15-retrievers-basic-rag.md), [Phase 16: 고급 RAG](../02-rag/16-advanced-rag.md)

> **API 변동 안내:** `InMemoryVectorStore`는 `langchain-community`에 포함되어 있으며, 버전에 따라 `langchain_core.vectorstores`로 이동할 수 있습니다. 최신 API는 [공식 문서](https://python.langchain.com/docs/integrations/vectorstores/)를 확인하세요.

---

◀ 이전: [Phase 33: 멀티에이전트 — 협업/Swarm](./33-multiagent-collaboration.md)
▶ 다음: [Phase 35: LangSmith 트레이싱](../05-production/35-langsmith-tracing.md)
