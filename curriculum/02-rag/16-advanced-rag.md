# Phase 16: 고급 RAG (Advanced RAG)

| 항목 | 내용 |
|------|------|
| 소요시간 | 약 90분 |
| 난이도 | ★★★★☆ |
| 선행 학습 | [Phase 15: 리트리버와 기본 RAG](15-retrievers-basic-rag.md) |

> **프로젝트 흐름**: Phase 15의 기본 RAG 파이프라인을 고도화합니다. 검색 품질이 낮을 때 사용하는 4가지 고급 기법을 배우고, "Python 라이브러리 공식 문서 QA 시스템"의 검색 정확도를 개선합니다.

---

## 🎯 학습 목표

- 기본 RAG의 한계를 이해하고 고급 기법이 필요한 상황을 식별합니다.
- `MultiQueryRetriever`로 쿼리 재작성을 통한 검색 범위를 확장합니다.
- `ContextualCompressionRetriever`로 검색된 문서에서 핵심 구절만 추출합니다.
- `EnsembleRetriever`로 BM25 키워드 검색과 벡터 검색을 결합합니다.
- `ParentDocumentRetriever`로 작은 청크로 검색하되 큰 컨텍스트를 반환합니다.

---

## 📚 핵심 개념

### 기본 RAG의 한계

Phase 15의 기본 RAG는 단일 쿼리 임베딩으로 유사 문서를 검색합니다. 이 방식은 세 가지 상황에서 실패합니다.

| 문제 | 원인 | 증상 |
|------|------|------|
| 어휘 불일치 | 사용자 표현 ≠ 문서 표현 | 정답이 있지만 검색 못 함 |
| 청크 노이즈 | 청크에 불필요한 내용 포함 | LLM이 혼란스러운 컨텍스트 수신 |
| 단일 쿼리 편향 | 하나의 임베딩만 사용 | 다각도 관련 문서 누락 |
| 청크 크기 트레이드오프 | 작은 청크=정밀 검색, 큰 청크=풍부한 맥락 | 둘 다 만족하기 어려움 |

### 고급 RAG 기법 개요

```
기본 RAG:       [질문] → [벡터 검색] → [Top-k 청크] → [LLM]

MultiQuery:     [질문] → [LLM이 쿼리 3개 생성] → [3번 검색 + 중복 제거] → [LLM]

Compression:    [질문] → [벡터 검색] → [LLM이 관련 구절만 추출] → [LLM]

Hybrid:         [질문] → [BM25 검색 + 벡터 검색] → [RRF 융합] → [LLM]

Parent-Child:   [질문] → [소형 청크로 검색] → [대형 부모 청크 반환] → [LLM]
```

---

## 💻 코드 예제

### 환경 설정

```bash
pip install langchain langchain-openai langchain-chroma langchain-community \
            chromadb rank-bm25 python-dotenv
```

```python
import os
from dotenv import load_dotenv

load_dotenv()
# OPENROUTER_API_KEY: LLM용
# OPENAI_API_KEY: 임베딩용
```

### 기반 설정 (Phase 15에서 이어받기)

```python
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# LLM: OpenRouter 경유
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# 임베딩: OpenAI 직접 사용 (OpenRouter는 임베딩 미지원)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 문서 로드 및 분할
loader = WebBaseLoader("https://docs.python.org/3/library/itertools.html")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 기본 벡터 스토어
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="python_docs_advanced",
    persist_directory="./chroma_db",
)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

---

### 기법 1: MultiQueryRetriever

단일 쿼리의 어휘 편향을 극복합니다. LLM이 원본 질문을 다양한 관점에서 재표현한 여러 쿼리를 생성하고, 각 쿼리로 검색한 결과를 합집합으로 통합합니다.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
import logging

# 디버깅: 생성된 쿼리 확인
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)

# 테스트
question = "무한 반복자를 만드는 방법은?"
docs_retrieved = multi_query_retriever.invoke(question)
print(f"검색된 문서 수: {len(docs_retrieved)}")

# INFO 로그에서 생성된 쿼리 3개 확인 가능:
# "itertools에서 무한 시퀀스 생성하기"
# "count, cycle, repeat 함수 사용법"
# "Python에서 무한 이터레이터 구현"
```

**언제 사용하나?**
- 사용자가 전문 용어를 모를 때 (예: "itertools" 대신 "반복 도구")
- 같은 개념을 여러 방식으로 표현할 수 있을 때

---

### 기법 2: ContextualCompressionRetriever

검색된 청크 전체가 아니라 질문과 관련된 구절만 LLM이 추출합니다. 노이즈를 줄여 LLM이 더 정확한 답변을 생성할 수 있게 합니다.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Extractor: 각 청크에서 관련 구절만 추출
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

question = "cycle 함수의 동작 방식을 설명해줘"
compressed_docs = compression_retriever.invoke(question)

for i, doc in enumerate(compressed_docs):
    print(f"\n--- 압축된 문서 {i+1} ---")
    print(doc.page_content[:300])
    # 원본 청크(500자) 중 핵심 구절(~100자)만 반환
```

**LLMChainFilter 대안 (더 빠름)**

```python
from langchain.retrievers.document_compressors import LLMChainFilter

# 추출 대신 필터링 (관련 없는 문서 전체 제거)
_filter = LLMChainFilter.from_llm(llm)
filter_retriever = ContextualCompressionRetriever(
    base_compressor=_filter,
    base_retriever=base_retriever,
)
```

> **⚠️ 주의**: 각 청크마다 LLM을 호출하므로 비용과 지연이 증가합니다. 청크 수가 많으면 LLMChainFilter가 더 경제적입니다.

---

### 기법 3: 하이브리드 검색 (EnsembleRetriever)

벡터 검색은 의미 유사도에 강하지만 정확한 키워드 매칭에 약합니다. BM25는 키워드 검색에 강합니다. 두 방식을 결합하면 상호 보완됩니다.

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# BM25 리트리버: 청크 텍스트 기반 키워드 검색
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

# 벡터 리트리버
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 앙상블: 두 결과를 Reciprocal Rank Fusion으로 통합
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],  # BM25 50% + 벡터 50%
)

question = "itertools.chain 함수 예제"
results = ensemble_retriever.invoke(question)
print(f"앙상블 검색 결과: {len(results)}개")
```

**가중치 조정 가이드**

| 상황 | 권장 weights |
|------|-------------|
| 정확한 함수명/코드 검색 | `[0.7, 0.3]` (BM25 우선) |
| 개념/의미 기반 검색 | `[0.3, 0.7]` (벡터 우선) |
| 일반 목적 | `[0.5, 0.5]` (균형) |

---

### 기법 4: ParentDocumentRetriever

작은 청크로 정밀하게 검색하되, LLM에는 더 넓은 컨텍스트(부모 청크)를 제공합니다. "작은 청크가 정확하게 검색됨" + "큰 청크가 더 풍부한 컨텍스트 제공"의 장점을 결합합니다.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 부모 청크 (큰 크기): LLM에 전달될 실제 컨텍스트
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
)

# 자식 청크 (작은 크기): 임베딩 및 검색에 사용
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
)

# 부모 문서를 저장할 인메모리 스토어
docstore = InMemoryStore()

# 자식 청크 임베딩을 저장할 벡터 스토어 (별도 컬렉션)
parent_vectorstore = Chroma(
    collection_name="child_chunks",
    embedding_function=embeddings,
)

parent_retriever = ParentDocumentRetriever(
    vectorstore=parent_vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 문서 추가 (자식 청크는 벡터 스토어에, 부모 청크는 docstore에 저장)
parent_retriever.add_documents(docs)

# 검색: 자식 청크로 검색하지만 반환은 부모 청크
question = "starmap 함수는 어떻게 사용하나요?"
parent_docs = parent_retriever.invoke(question)
print(f"반환된 부모 청크 수: {len(parent_docs)}")
print(f"첫 번째 청크 길이: {len(parent_docs[0].page_content)}자")
# 자식 청크(400자) 검색 → 부모 청크(~2000자) 반환
```

---

### 고급 RAG 파이프라인 완성

4가지 기법 중 MultiQueryRetriever + EnsembleRetriever를 결합한 실용적인 파이프라인입니다.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 하이브리드 리트리버를 기반으로 MultiQuery 적용
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],
)

advanced_retriever = MultiQueryRetriever.from_llm(
    retriever=hybrid_retriever,
    llm=llm,
)

# RAG 프롬프트
rag_prompt = ChatPromptTemplate.from_template("""
당신은 Python 공식 문서 전문가입니다.
아래 검색된 문서를 바탕으로 질문에 정확하게 답변하세요.
문서에 없는 내용은 "문서에서 찾을 수 없습니다"라고 답하세요.

검색된 문서:
{context}

질문: {question}

답변:
""")


def format_docs(docs: list[Document]) -> str:
    """검색된 문서 리스트를 하나의 문자열로 포맷팅합니다."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# 고급 RAG 체인
advanced_rag_chain = (
    {"context": advanced_retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# 실행
answer = advanced_rag_chain.invoke("itertools에서 조합 가능한 함수들을 설명해줘")
print(answer)
```

---

## ✏️ 실습 과제

### 과제 1: 기법 비교 실험

```python
import time

test_questions = [
    "무한 반복자를 만드는 방법은?",
    "두 리스트를 묶어서 순회하는 방법은?",
    "조합(combination)과 순열(permutation)의 차이는?",
]

retrievers = {
    "기본 (Base)": base_retriever,
    "MultiQuery": multi_query_retriever,
    "앙상블 (Hybrid)": ensemble_retriever,
}

for question in test_questions[:1]:  # API 비용 절약을 위해 1개만
    print(f"\n질문: {question}")
    print("=" * 60)
    for name, retriever in retrievers.items():
        start = time.time()
        docs = retriever.invoke(question)
        elapsed = time.time() - start
        print(f"{name}: {len(docs)}개 문서, {elapsed:.2f}초")
```

### 과제 2: 압축 효과 측정

```python
# 압축 전후 토큰 수 비교
def count_tokens_rough(text: str) -> int:
    """대략적인 토큰 수 추정 (4자 = 1토큰)."""
    return len(text) // 4


question = "chain 함수의 사용 예시를 알려줘"

raw_docs = base_retriever.invoke(question)
compressed_docs = compression_retriever.invoke(question)

raw_tokens = sum(count_tokens_rough(d.page_content) for d in raw_docs)
compressed_tokens = sum(count_tokens_rough(d.page_content) for d in compressed_docs)

print(f"압축 전 총 토큰: ~{raw_tokens}")
print(f"압축 후 총 토큰: ~{compressed_tokens}")
print(f"절감률: {(1 - compressed_tokens/raw_tokens)*100:.1f}%")
```

### 과제 3: ParentDocumentRetriever 청크 크기 실험

자식 청크 크기를 200, 400, 600으로 바꾸면서 검색 결과의 정밀도 변화를 관찰하세요. 어떤 크기에서 가장 관련성 높은 부모 청크가 반환되는지 비교합니다.

---

## ⚠️ 흔한 함정

### 1. MultiQueryRetriever가 느릴 때

```python
# 문제: 기본적으로 LLM 호출이 순차적
# 해결: 쿼리 수를 줄이거나 더 빠른 모델 사용

# 쿼리 수 직접 제어 (프롬프트 커스터마이즈)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["question"],
    template="""다음 질문에 대해 검색 관점이 다른 2개의 쿼리를 생성하세요.
각 줄에 하나씩 작성하세요.
원본 질문: {question}
생성된 쿼리:""",
)
# 기본 3개 → 2개로 줄여 속도 개선
```

### 2. BM25Retriever가 한국어에 취약할 때

```python
# 문제: BM25의 기본 토크나이저가 한국어 형태소 분리 미지원
# 해결: 커스텀 전처리 함수 적용 (영어 문서에서는 해당 없음)
# 영어 기반 Python 공식 문서에서는 BM25가 잘 동작합니다.
```

### 3. EnsembleRetriever 결과에 중복 문서가 많을 때

```python
# 문제: 두 리트리버가 동일 문서를 반환
# 원인: 문서 수가 적거나 쿼리와 매우 유사한 문서가 하나뿐
# 해결: k 값을 줄이거나 중복 제거 후처리
from langchain_core.documents import Document


def deduplicate_docs(docs: list[Document]) -> list[Document]:
    """page_content 기준으로 중복 문서를 제거합니다."""
    seen = set()
    unique = []
    for doc in docs:
        content_hash = hash(doc.page_content)
        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(doc)
    return unique
```

### 4. ParentDocumentRetriever에서 docstore 초기화 오류

```python
# 문제: add_documents() 전에 invoke() 호출
# 해결: 반드시 문서 추가 후 검색

# 잘못된 순서:
# parent_retriever.invoke("질문")  # ← docstore 비어있음
# parent_retriever.add_documents(docs)

# 올바른 순서:
parent_retriever.add_documents(docs)  # 먼저 추가
parent_retriever.invoke("질문")       # 그 다음 검색
```

### 5. API 변동 주의

> **📌 참고**: LangChain은 빠르게 발전하는 라이브러리입니다. `MultiQueryRetriever`, `ContextualCompressionRetriever` 등의 API는 버전에 따라 변경될 수 있습니다. 항상 [공식 문서](https://python.langchain.com/docs/how_to/#retrievers)에서 최신 사용법을 확인하세요.

---

## ✅ 셀프 체크

- [ ] `MultiQueryRetriever`가 내부적으로 LLM을 호출하여 쿼리를 생성한다는 것을 이해한다.
- [ ] `ContextualCompressionRetriever`와 `LLMChainExtractor`/`LLMChainFilter`의 차이를 설명할 수 있다.
- [ ] `EnsembleRetriever`의 `weights` 파라미터가 두 리트리버 결과의 Reciprocal Rank Fusion에 영향을 준다는 것을 안다.
- [ ] `ParentDocumentRetriever`에서 자식 청크는 검색에, 부모 청크는 컨텍스트 제공에 사용된다는 구조를 이해한다.
- [ ] BM25와 벡터 검색의 각각의 강점과 약점을 설명할 수 있다.
- [ ] 4가지 고급 기법 중 주어진 상황에 적합한 기법을 선택할 수 있다.

---

## 🔗 참고 자료

- [MultiQueryRetriever 가이드](https://python.langchain.com/docs/how_to/MultiQueryRetriever/)
- [ContextualCompressionRetriever 가이드](https://python.langchain.com/docs/how_to/contextual_compression/)
- [EnsembleRetriever 가이드](https://python.langchain.com/docs/how_to/ensemble_retriever/)
- [ParentDocumentRetriever 가이드](https://python.langchain.com/docs/how_to/parent_document_retriever/)
- [리트리버 목록 전체](https://python.langchain.com/docs/concepts/retrievers/)

---

← [Phase 15: 리트리버와 기본 RAG](15-retrievers-basic-rag.md) | [Phase 17: RAG 평가](17-rag-evaluation.md) →
