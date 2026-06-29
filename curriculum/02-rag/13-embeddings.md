# Phase 13: 임베딩 (Embeddings)

| 항목 | 내용 |
|------|------|
| 소요시간 | 약 75분 |
| 난이도 | ★★★☆☆ |
| 선행 학습 | [Phase 12: 텍스트 분할](12-text-splitting.md) |

> **프로젝트 흐름**: Phase 12에서 분할한 청크를 이 Phase에서 수치 벡터로 변환합니다. 이 벡터는 Phase 14에서 벡터 스토어에 저장됩니다.

---

## 🎯 학습 목표

- 임베딩이 텍스트를 수치 벡터로 변환하는 과정을 이해합니다.
- `OpenAIEmbeddings`를 사용하여 실제 임베딩을 생성합니다.
- `embed_query()`와 `embed_documents()`의 차이와 용도를 구분합니다.
- 코사인 유사도를 직접 계산하여 의미적 유사성을 측정합니다.
- 임베딩 모델과 차원 수 선택 기준을 이해합니다.

---

## 📚 핵심 개념

### 임베딩(Embedding)이란?

임베딩은 텍스트를 의미를 보존하는 고차원 숫자 벡터로 변환하는 기술입니다.

```
"LangChain은 LLM 프레임워크입니다"
         ↓ 임베딩 모델
[0.023, -0.145, 0.892, ..., 0.031]  ← 1536차원 벡터
```

핵심 특성: **의미가 비슷한 텍스트는 벡터 공간에서도 가깝습니다.**

```
"강아지"       → [0.8, 0.1, ...]
"개"           → [0.79, 0.12, ...]  ← 비슷한 벡터
"자동차"       → [-0.3, 0.9, ...]  ← 먼 벡터
```

### 유사도 측정: 코사인 유사도

두 벡터 사이의 각도를 측정하여 유사성을 계산합니다.

```
cos(θ) = (A · B) / (||A|| × ||B||)

범위: -1.0 (정반대) ~ 0.0 (무관) ~ 1.0 (동일)
임베딩 모델 출력은 보통 정규화되어 있어 0~1 범위
```

### RAG에서 임베딩의 역할

```
[문서 청크] → [임베딩] → [벡터 스토어 저장]
[사용자 질문] → [임베딩] → [벡터 스토어 검색] → [유사한 청크 반환]
```

### OpenAI 임베딩 모델 비교

| 모델 | 차원 | 최대 토큰 | 특징 |
|------|------|-----------|------|
| `text-embedding-3-small` | 1536 | 8191 | 빠르고 저렴 (추천 시작점) |
| `text-embedding-3-large` | 3072 | 8191 | 높은 정확도, 더 비쌈 |
| `text-embedding-ada-002` | 1536 | 8191 | 구버전 (하위 호환성용) |

> **API 변동 주의**: 모델 이름과 가격은 변경될 수 있습니다. 최신 정보는 [OpenAI 임베딩 문서](https://platform.openai.com/docs/models/embeddings)를 확인하세요.

---

## 💻 코드 예제

### 환경 설정

```python
import os
from dotenv import load_dotenv

load_dotenv()
# OPENAI_API_KEY가 .env 파일에 설정되어 있어야 합니다
```

```bash
pip install langchain-openai python-dotenv numpy
```

---

### 1. OpenAIEmbeddings 기본 사용

```python
from langchain_openai import OpenAIEmbeddings

# 기본 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 권장 기본 모델
    # api_key는 환경변수 OPENAI_API_KEY에서 자동 로드
)

# 단일 텍스트 임베딩
text = "LangChain은 LLM 애플리케이션 개발 프레임워크입니다."
vector = embeddings.embed_query(text)

print(f"벡터 차원: {len(vector)}")          # 1536
print(f"벡터 타입: {type(vector)}")         # list
print(f"처음 5개 값: {vector[:5]}")
# [0.0123, -0.0456, 0.0789, -0.0234, 0.0567]
```

---

### 2. embed_query() vs embed_documents()

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# embed_query(): 단일 검색 쿼리 임베딩 (사용자 질문)
query = "LangChain에서 문서를 어떻게 로드하나요?"
query_vector = embeddings.embed_query(query)
print(f"쿼리 벡터 차원: {len(query_vector)}")

# embed_documents(): 여러 문서 임베딩 (한 번의 API 호출로 일괄 처리)
documents = [
    "WebBaseLoader는 웹 페이지를 로드하는 로더입니다.",
    "PyPDFLoader는 PDF 파일을 로드하는 로더입니다.",
    "TextLoader는 텍스트 파일을 로드하는 로더입니다.",
]
doc_vectors = embeddings.embed_documents(documents)

print(f"문서 벡터 수: {len(doc_vectors)}")           # 3
print(f"각 벡터 차원: {len(doc_vectors[0])}")        # 1536
```

```
핵심 차이점:
- embed_query()    : 검색 시 사용자 질문을 임베딩
- embed_documents(): 인덱싱 시 문서들을 일괄 임베딩 (배치 최적화)

내부적으로 일부 모델은 쿼리와 문서에 다른 최적화를 적용하므로
용도에 맞는 메서드를 사용해야 합니다.
```

---

### 3. 코사인 유사도 직접 계산

```python
import numpy as np
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """두 벡터 간의 코사인 유사도를 계산합니다."""
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# 의미적으로 유사한 텍스트들 비교
texts = {
    "query": "파이썬에서 리스트를 정렬하는 방법",
    "similar": "Python list를 sort하는 방법",
    "related": "파이썬 자료구조 튜토리얼",
    "unrelated": "오늘 점심 메뉴는 무엇인가요",
}

# 임베딩 생성
vectors = {key: embeddings.embed_query(text) for key, text in texts.items()}

# 쿼리와 각 텍스트 간의 유사도 계산
query_vec = vectors["query"]
print(f"기준 텍스트: {texts['query']}\n")

for key, vec in vectors.items():
    if key == "query":
        continue
    similarity = cosine_similarity(query_vec, vec)
    print(f"유사도 ({key:10s}): {similarity:.4f} | {texts[key]}")
```

```
예상 출력:
기준 텍스트: 파이썬에서 리스트를 정렬하는 방법

유사도 (similar   ): 0.9123 | Python list를 sort하는 방법
유사도 (related   ): 0.7856 | 파이썬 자료구조 튜토리얼
유사도 (unrelated ): 0.3421 | 오늘 점심 메뉴는 무엇인가요
```

---

### 4. 차원 축소 옵션 (text-embedding-3 시리즈)

```python
from langchain_openai import OpenAIEmbeddings

# text-embedding-3 모델은 차원 축소를 지원합니다
# 더 작은 차원 = 더 빠른 검색 + 낮은 저장 비용 (정확도 소폭 감소)
embeddings_small_dim = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512,  # 기본 1536에서 512로 축소
)

vector = embeddings_small_dim.embed_query("테스트 텍스트")
print(f"축소된 차원: {len(vector)}")  # 512
```

---

### 5. 배치 처리와 비용 최적화

```python
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def embed_documents_in_batches(
    docs: list[Document],
    batch_size: int = 100,
) -> list[list[float]]:
    """
    대량의 문서를 배치로 나누어 임베딩합니다.

    Args:
        docs: 임베딩할 Document 리스트
        batch_size: 한 번에 처리할 문서 수

    Returns:
        각 문서의 임베딩 벡터 리스트
    """
    all_vectors = []
    texts = [doc.page_content for doc in docs]

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_vectors = embeddings.embed_documents(batch)
        all_vectors.extend(batch_vectors)

        progress = min(i + batch_size, len(texts))
        print(f"진행: {progress}/{len(texts)} 문서 처리 완료")

    return all_vectors


# 사용 예시 (분할된 문서들 임베딩)
# vectors = embed_documents_in_batches(split_docs, batch_size=50)
```

---

### 6. 다른 임베딩 모델 옵션

```python
# 옵션 1: HuggingFace (무료, 로컬 실행)
# pip install langchain-huggingface sentence-transformers
from langchain_huggingface import HuggingFaceEmbeddings

hf_embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",  # 한국어 포함 다국어 지원 모델
    model_kwargs={"device": "cpu"},
)

# 옵션 2: Ollama (로컬 LLM 서버 사용)
# pip install langchain-ollama
from langchain_ollama import OllamaEmbeddings

ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")
```

```python
# 임베딩 모델 선택 가이드
"""
비용 없이 시작: HuggingFaceEmbeddings (BAAI/bge-m3 — 다국어)
                OllamaEmbeddings (로컬 서버 필요)

프로덕션: OpenAIEmbeddings (text-embedding-3-small — 속도/비용 균형)
          OpenAIEmbeddings (text-embedding-3-large — 최고 정확도)

한국어 특화: BAAI/bge-m3, jhgan/ko-sroberta-multitask
"""
```

---

### 7. 임베딩 캐싱

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings

# 동일한 텍스트 재임베딩 방지 → API 비용 절감
underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
store = LocalFileStore("./cache/embeddings/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=underlying_embeddings,
    document_embedding_cache=store,
    namespace=underlying_embeddings.model,  # 모델별로 캐시 분리
)

# 첫 호출: API 요청 발생
vector1 = cached_embedder.embed_query("테스트 텍스트")
# 두 번째 호출: 캐시에서 로드 (API 요청 없음)
vector2 = cached_embedder.embed_query("테스트 텍스트")

print(f"캐시 동작 확인: {vector1 == vector2}")  # True
```

---

### 8. QA 시스템용 임베딩 준비 (프로젝트 계속)

```python
"""
프로젝트: Python 라이브러리 공식 문서 QA 시스템
Phase 13: 임베딩 단계
"""

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Phase 11~12 재현
LANGCHAIN_DOCS_URLS = [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
]
loader = WebBaseLoader(
    web_paths=LANGCHAIN_DOCS_URLS,
    bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("theme-doc-markdown", "markdown"))},
)
raw_docs = loader.load()
raw_docs = [doc for doc in raw_docs if len(doc.page_content.strip()) > 100]

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(raw_docs)

# 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 간단한 유사도 테스트
test_query = "LangChain에서 체인을 어떻게 구성하나요?"
query_vec = embeddings.embed_query(test_query)

# 첫 5개 청크와의 유사도 측정
sample_texts = [doc.page_content for doc in split_docs[:5]]
sample_vecs = embeddings.embed_documents(sample_texts)


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


print(f"질문: {test_query}\n")
for i, (text, vec) in enumerate(zip(sample_texts, sample_vecs)):
    sim = cosine_similarity(query_vec, vec)
    print(f"청크 {i + 1} 유사도: {sim:.4f}")
    print(f"  내용: {text[:100]}...")

print(f"\n임베딩 준비 완료. 총 {len(split_docs)}개 청크를 Phase 14에서 저장합니다.")
```

---

## ✏️ 실습 과제

**과제 1**: "파이썬"과 "Python"을 임베딩하고 코사인 유사도를 계산해 보세요. 같은 언어의 다른 표기법이 얼마나 유사하게 임베딩되는지 확인하세요.

**과제 2**: 문장 10개를 임베딩하고 서로 간의 유사도 행렬을 만들어 보세요. 의미적으로 가까운 문장들이 높은 유사도를 보이는지 확인하세요.

**과제 3**: `text-embedding-3-small`의 기본 차원(1536)과 축소 차원(256)으로 같은 텍스트를 임베딩하고, 검색 결과 품질을 비교해 보세요.

**과제 4 (심화)**: `CacheBackedEmbeddings`를 사용하여 동일한 텍스트를 두 번 임베딩할 때의 소요 시간을 `time` 모듈로 측정하고 캐싱 효과를 검증하세요.

---

## ⚠️ 흔한 함정

**함정 1: embed_query()와 embed_documents()를 바꿔 사용**
```python
# 검색 시 문서 임베딩 함수를 사용 → 성능 저하 가능
bad_query_vec = embeddings.embed_documents(["질문 텍스트"])[0]

# 올바른 사용
good_query_vec = embeddings.embed_query("질문 텍스트")
```

**함정 2: 토큰 한도 초과**
- `text-embedding-3-small`의 최대 입력은 **8191 토큰**입니다.
- Phase 12에서 `chunk_size`를 적절히 설정하지 않으면 임베딩 오류 발생합니다.
- 일반적으로 `chunk_size=1000`(문자)은 안전합니다.

**함정 3: 다차원 비교 시 일관성**
```python
# 잘못된 예: 다른 차원 벡터끼리 유사도 계산
vec_1536 = OpenAIEmbeddings(model="text-embedding-3-small").embed_query("test")
vec_512  = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=512).embed_query("test")

# cosine_similarity(vec_1536, vec_512)  # 차원이 달라 오류!
# ← 항상 동일한 모델/차원으로 임베딩해야 합니다
```

**함정 4: API 키 미설정**
```python
# KeyError 또는 AuthenticationError 방지
import os
assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY 환경변수가 설정되어야 합니다"
```

---

## ✅ 셀프 체크

- [ ] 임베딩이 텍스트를 벡터로 변환하고 의미 유사도를 보존하는 방식을 설명할 수 있다.
- [ ] `embed_query()`와 `embed_documents()`의 용도 차이를 설명할 수 있다.
- [ ] 코사인 유사도 공식을 이해하고 NumPy로 직접 계산할 수 있다.
- [ ] `text-embedding-3-small`과 `text-embedding-3-large`의 차이를 비교할 수 있다.
- [ ] `CacheBackedEmbeddings`를 사용하여 API 비용을 절감할 수 있다.
- [ ] 차원 수를 조절하는 방법과 트레이드오프를 설명할 수 있다.

---

## 🔗 참고 자료

- [LangChain 임베딩 개념 가이드](https://python.langchain.com/docs/concepts/embedding_models/)
- [OpenAIEmbeddings API 레퍼런스](https://python.langchain.com/api_reference/openai/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html)
- [CacheBackedEmbeddings 가이드](https://python.langchain.com/docs/how_to/caching_embeddings/)
- [OpenAI 임베딩 모델](https://platform.openai.com/docs/models/embeddings)

---

← [Phase 12: 텍스트 분할](12-text-splitting.md) | [Phase 14: 벡터 스토어](14-vector-stores.md) →
