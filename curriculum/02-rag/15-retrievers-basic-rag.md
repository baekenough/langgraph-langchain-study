# Phase 15: 리트리버와 기본 RAG (Retrievers & Basic RAG)

| 항목 | 내용 |
|------|------|
| 소요시간 | 약 90분 |
| 난이도 | ★★★★☆ |
| 선행 학습 | [Phase 14: 벡터 스토어](14-vector-stores.md) |

> **프로젝트 흐름**: Phase 14에서 구축한 벡터 스토어를 리트리버로 변환하고, LCEL로 완전한 RAG 파이프라인을 완성합니다. 이 Phase가 RAG 시스템의 핵심입니다.

---

## 🎯 학습 목표

- 리트리버(Retriever)의 역할과 벡터 스토어와의 관계를 이해합니다.
- `as_retriever()`로 벡터 스토어를 리트리버로 변환합니다.
- LCEL을 사용하여 retrieve → prompt → model → parse 파이프라인을 구성합니다.
- 검색 결과에 출처(citation)를 포함하는 RAG를 구현합니다.
- `format_docs()` 헬퍼 함수로 문서 포맷팅을 처리합니다.

---

## 📚 핵심 개념

### RAG(검색 증강 생성) 아키텍처

```
사용자 질문
    │
    ▼
[Retriever] ── 질문을 임베딩하여 유사한 문서 검색
    │
    ▼
[관련 문서 청크들]
    │
    ▼
[Prompt] ── 질문 + 검색된 문서를 하나의 프롬프트로 조합
    │
    ▼
[LLM] ── 프롬프트를 기반으로 답변 생성
    │
    ▼
[Parser] ── LLM 출력을 최종 형식으로 파싱
    │
    ▼
최종 답변 (+ 출처)
```

### 리트리버(Retriever)

리트리버는 `BaseRetriever`를 구현한 인터페이스로, 문자열 쿼리를 입력받아 `List[Document]`를 반환합니다.

```python
# 리트리버의 핵심 인터페이스
retriever.invoke("질문")  # → List[Document]
```

벡터 스토어를 `as_retriever()`로 변환하면 LCEL 체인에서 바로 사용할 수 있습니다.

### LCEL 기반 RAG 체인 구조

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## 💻 코드 예제

### 환경 설정

```bash
pip install langchain langchain-openai langchain-chroma chromadb python-dotenv
```

```python
import os
from dotenv import load_dotenv

load_dotenv()
```

---

### 1. as_retriever() — 벡터 스토어를 리트리버로 변환

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Phase 14에서 구축한 벡터 스토어 로드
vector_store = Chroma(
    collection_name="langchain_docs_qa",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_docs",
)

# 기본 리트리버 생성 (k=4: 상위 4개 문서 반환)
retriever = vector_store.as_retriever(
    search_type="similarity",    # "similarity" | "mmr" | "similarity_score_threshold"
    search_kwargs={"k": 4},
)

# 리트리버 테스트
docs = retriever.invoke("LCEL로 체인을 구성하는 방법")
print(f"검색된 문서 수: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"\n문서 {i + 1}:")
    print(f"  출처: {doc.metadata.get('source', 'unknown')}")
    print(f"  내용: {doc.page_content[:150]}...")
```

```python
# search_type 옵션 비교
# 1. similarity: 기본 코사인 유사도 기반
similarity_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)

# 2. mmr (Maximal Marginal Relevance): 다양성 + 관련성 균형
# 유사한 문서가 중복으로 반환되는 것을 방지합니다
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.7},
    # lambda_mult: 0=최대 다양성, 1=최대 관련성
)

# 3. similarity_score_threshold: 최소 유사도 이상만 반환
threshold_retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 4},
)
```

---

### 2. format_docs() — 문서 포맷팅 헬퍼

```python
from langchain_core.documents import Document


def format_docs(docs: list[Document]) -> str:
    """
    Document 리스트를 프롬프트에 삽입하기 위한 문자열로 변환합니다.

    각 문서를 구분자로 분리하여 LLM이 출처를 구분할 수 있게 합니다.
    """
    return "\n\n".join(doc.page_content for doc in docs)


# 사용 예시
sample_docs = [
    Document(page_content="LCEL은 파이프라인을 | 연산자로 구성합니다.", metadata={"source": "lcel_docs"}),
    Document(page_content="RunnablePassthrough는 입력을 그대로 전달합니다.", metadata={"source": "lcel_docs"}),
]

formatted = format_docs(sample_docs)
print(formatted)
# LCEL은 파이프라인을 | 연산자로 구성합니다.
#
# RunnablePassthrough는 입력을 그대로 전달합니다.
```

---

### 3. RAG 프롬프트 설계

```python
from langchain_core.prompts import ChatPromptTemplate

# RAG에 특화된 프롬프트 — 핵심 원칙:
# 1. 제공된 컨텍스트만 사용하도록 명시
# 2. 모르는 경우 솔직하게 답변하도록 유도
# 3. 간결하고 명확한 답변 요청
RAG_PROMPT_TEMPLATE = """당신은 LangChain 공식 문서를 기반으로 질문에 답변하는 전문 어시스턴트입니다.
아래 제공된 컨텍스트만을 사용하여 질문에 답변해 주세요.
컨텍스트에 관련 정보가 없다면 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 답변하세요.

[컨텍스트]
{context}

[질문]
{question}

[답변]"""

prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

# Hub에서 검증된 RAG 프롬프트 사용 (선택 사항)
# from langchain import hub
# prompt = hub.pull("rlm/rag-prompt")
```

---

### 4. 기본 RAG 체인 구성 (LCEL)

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# LCEL RAG 체인 구성
# RunnablePassthrough: 입력(질문)을 변경 없이 "question" 키로 전달
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 체인 실행
question = "LangChain에서 LCEL을 사용하면 어떤 이점이 있나요?"
answer = rag_chain.invoke(question)

print(f"질문: {question}\n")
print(f"답변:\n{answer}")
```

---

### 5. 출처(Citation) 포함 RAG

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from typing import Any


def format_docs_with_sources(docs: list[Document]) -> str:
    """문서 내용과 출처 번호를 포함하여 포맷팅합니다."""
    formatted_parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "출처 미상")
        formatted_parts.append(f"[문서 {i}] 출처: {source}\n{doc.page_content}")
    return "\n\n".join(formatted_parts)


# 출처 포함 프롬프트
CITATION_PROMPT_TEMPLATE = """당신은 LangChain 공식 문서를 기반으로 답변하는 어시스턴트입니다.
아래 컨텍스트를 사용하여 답변하고, 참조한 문서 번호를 [문서 N] 형식으로 인용하세요.

[컨텍스트]
{context}

[질문]
{question}

[답변] (참조 문서 번호를 포함하세요)"""

citation_prompt = ChatPromptTemplate.from_template(CITATION_PROMPT_TEMPLATE)

# 답변과 소스 문서를 함께 반환하는 체인
rag_chain_with_source = RunnableParallel(
    # 답변 생성
    answer=(
        {"context": retriever | format_docs_with_sources, "question": RunnablePassthrough()}
        | citation_prompt
        | llm
        | StrOutputParser()
    ),
    # 원본 소스 문서 반환
    sources=retriever,
)

result = rag_chain_with_source.invoke("LangChain에서 체인을 어떻게 구성하나요?")

print(f"답변:\n{result['answer']}")
print(f"\n참조된 소스 문서 ({len(result['sources'])}개):")
for i, doc in enumerate(result["sources"], start=1):
    source = doc.metadata.get("source", "출처 미상")
    print(f"  [문서 {i}] {source}")
```

---

### 6. 스트리밍 RAG

```python
# LCEL 체인은 기본적으로 스트리밍을 지원합니다
print(f"질문: LCEL의 핵심 특징은 무엇인가요?\n답변: ", end="", flush=True)

for chunk in rag_chain.stream("LCEL의 핵심 특징은 무엇인가요?"):
    print(chunk, end="", flush=True)
print()  # 줄바꿈
```

---

### 7. 대화형 RAG (히스토리 포함)

```python
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 대화 히스토리를 포함하는 프롬프트
CONVERSATIONAL_RAG_PROMPT = """당신은 LangChain 공식 문서를 기반으로 답변하는 어시스턴트입니다.
이전 대화 내용과 아래 컨텍스트를 참고하여 답변해 주세요.

[컨텍스트]
{context}"""

conv_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONVERSATIONAL_RAG_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

# 대화 히스토리 관리
chat_history: list[HumanMessage | AIMessage] = []


def chat_rag(question: str) -> str:
    """대화 히스토리를 유지하는 RAG 함수."""
    conv_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: chat_history,
        }
        | conv_prompt
        | llm
        | StrOutputParser()
    )

    answer = conv_chain.invoke(question)

    # 히스토리 업데이트
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    return answer


# 다중 턴 대화
print("=== 대화형 RAG 시스템 ===\n")

q1 = "LangChain이 무엇인가요?"
a1 = chat_rag(q1)
print(f"Q: {q1}\nA: {a1}\n")

q2 = "그렇다면 LCEL은 어디에 사용되나요?"  # 이전 대화 컨텍스트 활용
a2 = chat_rag(q2)
print(f"Q: {q2}\nA: {a2}\n")
```

---

### 8. 완전한 QA 시스템 (프로젝트 완성)

```python
"""
프로젝트: Python 라이브러리 공식 문서 QA 시스템
Phase 15: RAG 파이프라인 완성
"""

import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# 벡터 스토어 로드 (또는 새로 구축)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

try:
    vector_store = Chroma(
        collection_name="langchain_docs_qa",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_docs",
    )
    if vector_store._collection.count() == 0:
        raise ValueError("빈 컬렉션")
    print(f"벡터 스토어 로드 완료 ({vector_store._collection.count()}개 문서)")
except Exception:
    print("벡터 스토어 새로 구축 중...")
    URLS = ["https://python.langchain.com/docs/introduction/", "https://python.langchain.com/docs/concepts/"]
    loader = WebBaseLoader(
        web_paths=URLS,
        bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("theme-doc-markdown", "markdown"))},
    )
    raw_docs = loader.load()
    raw_docs = [d for d in raw_docs if len(d.page_content.strip()) > 100]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(raw_docs)
    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name="langchain_docs_qa",
        persist_directory="./chroma_langchain_docs",
    )
    print(f"벡터 스토어 구축 완료 ({vector_store._collection.count()}개 문서)")

# RAG 체인 구성
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

RAG_PROMPT = """당신은 LangChain 공식 문서를 기반으로 답변하는 전문 어시스턴트입니다.
제공된 컨텍스트만을 사용하여 한국어로 답변해 주세요.
컨텍스트에 없는 정보는 "문서에서 찾을 수 없습니다."라고 답변하세요.

[컨텍스트]
{context}

[질문]
{question}"""

prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

rag_chain = RunnableParallel(
    answer=(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    ),
    sources=retriever,
)


def ask(question: str) -> None:
    """질문을 받아 RAG로 답변하고 출처를 표시합니다."""
    result = rag_chain.invoke(question)
    print(f"\n질문: {question}")
    print(f"\n답변:\n{result['answer']}")
    print(f"\n참조 문서:")
    seen_sources = set()
    for doc in result["sources"]:
        source = doc.metadata.get("source", "출처 미상")
        if source not in seen_sources:
            print(f"  - {source}")
            seen_sources.add(source)
    print("-" * 60)


# QA 시스템 사용
ask("LangChain에서 LCEL을 사용하는 이유는 무엇인가요?")
ask("벡터 스토어와 리트리버의 차이는 무엇인가요?")
ask("RAG 시스템을 구축하는 기본 단계를 설명해 주세요.")
```

---

## ✏️ 실습 과제

**과제 1**: `search_type="mmr"`과 `search_type="similarity"`를 비교하여 같은 질문에 대해 반환되는 문서의 다양성 차이를 확인하세요.

**과제 2**: RAG 프롬프트를 수정하여 "반드시 3개 이하의 문장으로 답변"하도록 제약을 추가하고, 답변 품질에 미치는 영향을 평가해 보세요.

**과제 3**: `rag_chain.stream()`을 사용하여 스트리밍 출력을 구현하고, 실시간으로 답변이 생성되는 것을 확인하세요.

**과제 4 (심화)**: `RunnableParallel`로 3개의 서로 다른 k 값(2, 4, 8)을 사용하는 리트리버를 동시에 실행하고 결과를 비교하는 체인을 구성해 보세요.

---

## ⚠️ 흔한 함정

**함정 1: format_docs() 없이 리트리버를 직접 프롬프트에 연결**
```python
# 잘못된 예: Document 객체가 그대로 프롬프트에 삽입됨
bad_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt

# 올바른 예: format_docs로 문자열 변환
good_chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt
```

**함정 2: temperature=0이 아닌 경우 환각 증가**
```python
# RAG에서는 창의성보다 정확성이 중요합니다
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # 결정론적 출력 권장
```

**함정 3: 컨텍스트 없는 질문에 대한 미처리**
- 검색된 문서가 실제로 관련이 없어도 LLM이 억지로 답변할 수 있습니다.
- 프롬프트에 "컨텍스트에 없으면 모른다고 답변하라"는 지시를 명확히 포함하세요.

**함정 4: 리트리버 k 값을 너무 크게 설정**
- k가 클수록 더 많은 컨텍스트 → LLM 비용 증가, 관련성 낮은 정보 포함 가능
- 일반적으로 k=3~6이 적절합니다.

---

## ✅ 셀프 체크

- [ ] `as_retriever()`로 벡터 스토어를 리트리버로 변환하고 `invoke()`를 호출할 수 있다.
- [ ] `similarity`, `mmr`, `similarity_score_threshold` 검색 타입의 차이를 설명할 수 있다.
- [ ] LCEL로 `retriever | format_docs → prompt → llm → parser` 파이프라인을 구성할 수 있다.
- [ ] `RunnableParallel`로 답변과 소스 문서를 동시에 반환할 수 있다.
- [ ] 스트리밍을 사용하여 실시간 답변 출력을 구현할 수 있다.
- [ ] 대화 히스토리를 관리하는 다중 턴 RAG를 구현할 수 있다.

---

## 🔗 참고 자료

- [RAG 개념 가이드](https://python.langchain.com/docs/concepts/rag/)
- [리트리버 개념 가이드](https://python.langchain.com/docs/concepts/retrievers/)
- [LCEL RAG 튜토리얼](https://python.langchain.com/docs/tutorials/rag/)
- [RunnableParallel 가이드](https://python.langchain.com/docs/how_to/parallel/)
- [RAG Q&A with History 튜토리얼](https://python.langchain.com/docs/tutorials/qa_chat_history/)

---

← [Phase 14: 벡터 스토어](14-vector-stores.md) | [Phase 16: 고급 RAG](16-advanced-rag.md) →
