# Phase 14: 벡터 스토어 (Vector Stores)

| 항목 | 내용 |
|------|------|
| 소요시간 | 약 90분 |
| 난이도 | ★★★☆☆ |
| 선행 학습 | [Phase 13: 임베딩](13-embeddings.md) |

> **프로젝트 흐름**: Phase 13에서 준비한 임베딩 모델을 사용하여, 분할된 문서들을 벡터 스토어에 저장합니다. Phase 15에서 이 스토어를 검색 엔진으로 활용합니다.

---

## 🎯 학습 목표

- 벡터 스토어의 역할과 동작 원리를 이해합니다.
- `InMemoryVectorStore`로 빠르게 프로토타이핑합니다.
- `Chroma`를 사용하여 데이터를 디스크에 영속화합니다.
- `add_documents()`, `similarity_search()`, `similarity_search_with_score()`를 활용합니다.
- 메타데이터 필터로 검색 범위를 제한합니다.

---

## 📚 핵심 개념

### 벡터 스토어란?

벡터 스토어는 임베딩 벡터를 저장하고 유사도 기반 검색을 수행하는 특수 데이터베이스입니다.

```
┌─────────────────────────────────────────────┐
│              벡터 스토어                      │
│                                             │
│  문서 A → [0.1, 0.8, 0.3, ...]             │
│  문서 B → [0.9, 0.2, 0.7, ...]             │
│  문서 C → [0.2, 0.7, 0.4, ...]  ←── 최근접 │
│  ...                                        │
│                                             │
│  질문 벡터 [0.15, 0.75, 0.35, ...] 입력    │
│  → ANN 알고리즘으로 유사한 문서 반환         │
└─────────────────────────────────────────────┘
```

### 벡터 스토어 선택 가이드

| 스토어 | 영속성 | 특징 | 권장 상황 |
|--------|--------|------|-----------|
| `InMemoryVectorStore` | 없음 (메모리) | 설치 불필요, 즉시 사용 | 학습, 프로토타이핑 |
| `Chroma` | 있음 (로컬 파일) | 경량, 설치 간편 | 개발, 소규모 프로덕션 |
| `FAISS` | 있음 (파일) | 빠른 유사도 검색, Meta 개발 | 대규모 데이터 |
| `pgvector` | 있음 (PostgreSQL) | SQL + 벡터 검색, 트랜잭션 | 기존 PostgreSQL 사용 중 |
| `Pinecone` | 있음 (클라우드) | 완전 관리형 서비스 | 프로덕션, 팀 협업 |

> **API 변동 주의**: 각 벡터 스토어의 설치 방법과 API는 버전에 따라 변경될 수 있습니다. 최신 정보는 [공식 문서](https://python.langchain.com/docs/integrations/vectorstores/)를 확인하세요.

---

## 💻 코드 예제

### 환경 설정

```bash
pip install langchain-core langchain-openai langchain-chroma chromadb python-dotenv
```

```python
import os
from dotenv import load_dotenv

load_dotenv()
```

---

### 1. InMemoryVectorStore — 메모리 기반 빠른 시작

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# InMemoryVectorStore 생성
vector_store = InMemoryVectorStore(embedding=embeddings)

# 문서 추가
docs = [
    Document(
        page_content="WebBaseLoader는 웹 페이지에서 텍스트를 추출하는 로더입니다.",
        metadata={"source": "langchain_docs", "topic": "loaders", "page": 1},
    ),
    Document(
        page_content="PyPDFLoader는 PDF 파일을 페이지 단위로 로드합니다.",
        metadata={"source": "langchain_docs", "topic": "loaders", "page": 2},
    ),
    Document(
        page_content="RecursiveCharacterTextSplitter는 텍스트를 재귀적으로 분할합니다.",
        metadata={"source": "langchain_docs", "topic": "splitters", "page": 3},
    ),
    Document(
        page_content="OpenAIEmbeddings는 OpenAI API를 통해 텍스트를 벡터로 변환합니다.",
        metadata={"source": "langchain_docs", "topic": "embeddings", "page": 4},
    ),
    Document(
        page_content="Chroma는 오픈소스 벡터 데이터베이스로 로컬 파일에 저장합니다.",
        metadata={"source": "langchain_docs", "topic": "vector_stores", "page": 5},
    ),
]

# 문서 추가 (임베딩 자동 수행)
ids = vector_store.add_documents(docs)
print(f"추가된 문서 수: {len(ids)}")
print(f"문서 ID 예시: {ids[0]}")
```

---

### 2. 유사도 검색

```python
# similarity_search(): 가장 유사한 k개 문서 반환
query = "PDF 파일을 불러오는 방법"
results = vector_store.similarity_search(query, k=2)

print(f"검색어: {query}\n")
for i, doc in enumerate(results):
    print(f"결과 {i + 1}:")
    print(f"  내용: {doc.page_content}")
    print(f"  메타데이터: {doc.metadata}")
    print()
```

```python
# similarity_search_with_score(): 유사도 점수 포함 반환
results_with_score = vector_store.similarity_search_with_score(query, k=3)

print(f"검색어: {query}\n")
for doc, score in results_with_score:
    # 점수는 벡터 스토어 구현에 따라 다릅니다
    # InMemoryVectorStore: 코사인 유사도 (높을수록 유사)
    # Chroma: L2 거리 (낮을수록 유사)
    print(f"점수: {score:.4f} | 내용: {doc.page_content[:60]}...")
```

---

### 3. Chroma — 영속화 벡터 스토어

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 새로운 Chroma 컬렉션 생성 (로컬 디스크에 저장)
vector_store = Chroma(
    collection_name="langchain_docs_qa",
    embedding_function=embeddings,
    persist_directory="./chroma_db",  # 데이터 저장 위치
)

# 문서 추가
docs = [
    Document(
        page_content="LCEL은 LangChain Expression Language의 약자로, | 연산자로 체인을 구성합니다.",
        metadata={"source": "langchain_docs", "topic": "lcel", "version": "0.2"},
    ),
    Document(
        page_content="RunnablePassthrough는 입력을 변경 없이 다음 단계로 전달합니다.",
        metadata={"source": "langchain_docs", "topic": "lcel", "version": "0.2"},
    ),
    Document(
        page_content="StrOutputParser는 LLM 출력을 문자열로 파싱합니다.",
        metadata={"source": "langchain_docs", "topic": "output_parsers", "version": "0.2"},
    ),
]

ids = vector_store.add_documents(docs)
print(f"Chroma에 {len(ids)}개 문서 추가 완료")
print(f"저장 위치: ./chroma_db/")
```

```python
# 프로그램 재시작 후 기존 Chroma 컬렉션 로드
vector_store_loaded = Chroma(
    collection_name="langchain_docs_qa",
    embedding_function=embeddings,
    persist_directory="./chroma_db",  # 이전과 동일한 경로
)

# 기존 데이터가 그대로 남아 있습니다
results = vector_store_loaded.similarity_search("LCEL 파이프라인 구성", k=2)
print(f"재로드 후 검색 결과: {len(results)}개")
```

---

### 4. 문서 관리 (추가/삭제/업데이트)

```python
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 새 문서 추가
new_doc = Document(
    page_content="ChatOpenAI는 OpenAI의 챗 모델을 LangChain에서 사용하는 클래스입니다.",
    metadata={"source": "langchain_docs", "topic": "models"},
)
new_ids = vector_store.add_documents([new_doc])
print(f"추가된 ID: {new_ids[0]}")

# 특정 ID로 문서 삭제
vector_store.delete(ids=[new_ids[0]])
print("문서 삭제 완료")

# 전체 컬렉션 문서 수 확인
collection = vector_store._collection
print(f"현재 컬렉션 문서 수: {collection.count()}")
```

---

### 5. 메타데이터 필터

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(
    collection_name="filtered_search_demo",
    embedding_function=embeddings,
)

# 다양한 토픽의 문서 추가
demo_docs = [
    Document(page_content="WebBaseLoader 사용법", metadata={"topic": "loaders", "level": "beginner"}),
    Document(page_content="PyPDFLoader 심화 가이드", metadata={"topic": "loaders", "level": "advanced"}),
    Document(page_content="Chroma 설정 방법", metadata={"topic": "vector_stores", "level": "beginner"}),
    Document(page_content="FAISS 인덱스 최적화", metadata={"topic": "vector_stores", "level": "advanced"}),
    Document(page_content="OpenAI 임베딩 기초", metadata={"topic": "embeddings", "level": "beginner"}),
]
vector_store.add_documents(demo_docs)

# 메타데이터 필터 적용 검색
# Chroma 필터 문법: {"field": value} 또는 논리 연산자 사용
results = vector_store.similarity_search(
    query="데이터 로드 방법",
    k=3,
    filter={"topic": "loaders"},  # topic이 "loaders"인 문서만 검색
)

print(f"'loaders' 토픽 필터 검색 결과:")
for doc in results:
    print(f"  [{doc.metadata['level']}] {doc.page_content}")
```

```python
# 복합 필터 (Chroma에서 지원하는 연산자 사용)
results = vector_store.similarity_search(
    query="고급 설정",
    k=3,
    filter={
        "$and": [
            {"level": {"$eq": "advanced"}},
            {"topic": {"$in": ["loaders", "vector_stores"]}},
        ]
    },
)
print(f"\n고급 레벨 loaders/vector_stores 필터 결과:")
for doc in results:
    print(f"  {doc.page_content} (topic={doc.metadata['topic']})")
```

---

### 6. from_documents() 클래스 메서드로 일괄 생성

```python
from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

docs = [
    Document(page_content=f"문서 {i}: LangChain 개념 설명", metadata={"id": i})
    for i in range(10)
]

# 문서 리스트에서 벡터 스토어를 한 번에 생성 (편의 메서드)
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="batch_demo",
    persist_directory="./chroma_db_demo",
)

print(f"일괄 생성 완료: {vector_store._collection.count()}개 문서")
```

---

### 7. 다른 벡터 스토어 간략 소개

```python
# FAISS (대규모 데이터, 로컬)
# pip install faiss-cpu langchain-community
from langchain_community.vectorstores import FAISS

faiss_store = FAISS.from_documents(docs, embeddings)
faiss_store.save_local("./faiss_index")              # 저장
faiss_store_loaded = FAISS.load_local(               # 로드
    "./faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

# pgvector (PostgreSQL 기반, 기업 환경)
# pip install langchain-postgres psycopg2-binary
# from langchain_postgres import PGVector
# connection = "postgresql://user:pass@localhost:5432/mydb"
# pg_store = PGVector(connection=connection, embeddings=embeddings, collection_name="docs")
```

---

### 8. QA 시스템용 벡터 스토어 구축 (프로젝트 계속)

```python
"""
프로젝트: Python 라이브러리 공식 문서 QA 시스템
Phase 14: 벡터 스토어 구축 단계
"""

import bs4
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Phase 11~12 재현: 문서 로드 및 분할
LANGCHAIN_DOCS_URLS = [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
    "https://python.langchain.com/docs/concepts/lcel/",
]

print("[1/3] 문서 로딩 중...")
loader = WebBaseLoader(
    web_paths=LANGCHAIN_DOCS_URLS,
    bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("theme-doc-markdown", "markdown"))},
)
raw_docs = loader.load()
raw_docs = [doc for doc in raw_docs if len(doc.page_content.strip()) > 100]

print("[2/3] 문서 분할 중...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(raw_docs)
print(f"  총 {len(split_docs)}개 청크 생성")

# Phase 13: 임베딩 모델
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("[3/3] 벡터 스토어 구축 중 (임베딩 + 저장)...")
vector_store = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    collection_name="langchain_docs_qa",
    persist_directory="./chroma_langchain_docs",
)

print(f"\n벡터 스토어 구축 완료!")
print(f"  저장 위치: ./chroma_langchain_docs/")
print(f"  총 문서 수: {vector_store._collection.count()}")

# 검색 테스트
test_results = vector_store.similarity_search("LCEL 파이프라인 구성 방법", k=2)
print(f"\n테스트 검색 결과 ({len(test_results)}개):")
for doc in test_results:
    print(f"  출처: {doc.metadata.get('source', 'unknown')}")
    print(f"  내용: {doc.page_content[:100]}...")

print("\nPhase 15에서 이 벡터 스토어로 RAG 체인을 완성합니다.")
```

---

## ✏️ 실습 과제

**과제 1**: `InMemoryVectorStore`에 문서 20개를 추가하고 `k=5`로 검색 후, 결과의 유사도 점수를 내림차순으로 출력해 보세요.

**과제 2**: Chroma를 사용하여 벡터 스토어를 생성하고 저장한 뒤, 파이썬을 재시작하고 동일한 경로에서 로드하여 검색이 정상적으로 작동하는지 확인하세요.

**과제 3**: `filter` 파라미터를 사용하여 특정 `source`를 가진 문서만 검색하는 예제를 작성해 보세요. 필터 없는 검색과 결과를 비교하세요.

**과제 4 (심화)**: `similarity_search_with_score()`의 점수 값을 분석하여 "유사도 임계값(threshold)"을 설정하고, 임계값 이하의 결과는 무시하는 필터 함수를 구현해 보세요.

---

## ⚠️ 흔한 함정

**함정 1: Chroma persist_directory 경로 중복**
```python
# 같은 persist_directory로 다른 collection_name을 사용하면 섞일 수 있음
store1 = Chroma(collection_name="docs_v1", persist_directory="./chroma_db")
store2 = Chroma(collection_name="docs_v2", persist_directory="./chroma_db")
# 경로는 같아도 collection_name이 다르면 독립적으로 관리됩니다
```

**함정 2: similarity_search_with_score의 점수 방향**
```python
# 구현마다 점수의 의미가 다릅니다!
# InMemoryVectorStore: 코사인 유사도 → 높을수록 유사
# Chroma:             L2 거리        → 낮을수록 유사
# FAISS:              내적 또는 L2   → 설정에 따라 다름

# 항상 사용하는 벡터 스토어의 문서를 확인하세요
```

**함정 3: 문서 업데이트 미지원**
```python
# 벡터 스토어는 일반적으로 "업데이트"를 직접 지원하지 않습니다
# 삭제 후 재추가가 올바른 방법입니다
vector_store.delete(ids=[doc_id])
vector_store.add_documents([updated_doc])
```

**함정 4: 대량 문서 한 번에 추가**
```python
# 수천 개 문서를 한 번에 추가하면 API 타임아웃 가능
# from_documents()를 사용하거나, 배치로 나누어 추가하세요
batch_size = 100
for i in range(0, len(split_docs), batch_size):
    batch = split_docs[i : i + batch_size]
    vector_store.add_documents(batch)
```

---

## ✅ 셀프 체크

- [ ] 벡터 스토어의 역할과 ANN 검색 원리를 설명할 수 있다.
- [ ] `InMemoryVectorStore`와 `Chroma`의 차이점과 각각의 적합한 사용 시나리오를 설명할 수 있다.
- [ ] `add_documents()`, `similarity_search()`, `similarity_search_with_score()`를 사용할 수 있다.
- [ ] `Chroma`로 데이터를 디스크에 저장하고 다시 로드할 수 있다.
- [ ] 메타데이터 필터를 사용하여 검색 범위를 제한할 수 있다.
- [ ] 점수 방향이 벡터 스토어마다 다를 수 있음을 이해하고 확인하는 방법을 안다.

---

## 🔗 참고 자료

- [벡터 스토어 개념 가이드](https://python.langchain.com/docs/concepts/vectorstores/)
- [벡터 스토어 통합 목록](https://python.langchain.com/docs/integrations/vectorstores/)
- [Chroma 공식 문서](https://docs.trychroma.com/)
- [InMemoryVectorStore API](https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.in_memory.InMemoryVectorStore.html)
- [FAISS 설치 및 사용](https://python.langchain.com/docs/integrations/vectorstores/faiss/)

---

← [Phase 13: 임베딩](13-embeddings.md) | [Phase 15: 리트리버와 기본 RAG](15-retrievers-basic-rag.md) →
