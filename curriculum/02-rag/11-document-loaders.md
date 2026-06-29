# Phase 11: 문서 로더 (Document Loaders)

| 항목 | 내용 |
|------|------|
| 소요시간 | 약 60분 |
| 난이도 | ★★☆☆☆ |
| 선행 학습 | [Phase 10: 구조화된 출력](../01-langchain-core/10-structured-output.md) |

> **프로젝트 흐름**: 이 Phase부터 Phase 17까지는 **"Python 라이브러리 공식 문서 QA 시스템"**을 단계적으로 완성합니다. 각 Phase는 이전 Phase의 결과물을 이어받습니다.

---

## 🎯 학습 목표

- LangChain `Document` 객체의 구조(`page_content`, `metadata`)를 이해합니다.
- `WebBaseLoader`, `PyPDFLoader`, `DirectoryLoader`, `TextLoader`를 실제로 사용합니다.
- `lazy_load()`로 메모리 효율적으로 문서를 처리합니다.
- 로딩 후 데이터를 점검하여 품질을 확인하는 방법을 익힙니다.

---

## 📚 핵심 개념

### Document 객체

LangChain에서 모든 문서는 `Document` 객체로 표현됩니다. 두 가지 핵심 필드를 가집니다.

| 필드 | 타입 | 설명 |
|------|------|------|
| `page_content` | `str` | 문서의 실제 텍스트 내용 |
| `metadata` | `dict` | 출처, 페이지 번호, 작성자 등 부가 정보 |

`metadata`는 RAG 시스템에서 검색 결과의 출처를 추적하거나 필터링할 때 매우 중요합니다.

### 로더(Loader) 종류

| 로더 | 용도 | 패키지 |
|------|------|--------|
| `TextLoader` | 로컬 텍스트 파일 | `langchain_community` |
| `PyPDFLoader` | PDF 파일 | `langchain_community` |
| `WebBaseLoader` | 웹 페이지 URL | `langchain_community` |
| `DirectoryLoader` | 디렉토리 내 여러 파일 | `langchain_community` |

### load() vs lazy_load()

- `load()`: 모든 문서를 메모리에 한 번에 로드합니다. 소규모 데이터에 적합합니다.
- `lazy_load()`: 제너레이터를 반환하여 필요할 때마다 한 문서씩 읽습니다. 대용량 처리에 적합합니다.

---

## 💻 코드 예제

### 환경 설정

```python
# .env 파일에 OPENAI_API_KEY=sk-... 저장 후 사용
from dotenv import load_dotenv

load_dotenv()
```

```bash
# 필요 패키지 설치
pip install langchain-community langchain-openai python-dotenv
pip install pypdf beautifulsoup4 requests
```

> **API 변동 주의**: 로더 위치는 LangChain 버전에 따라 변경될 수 있습니다. 최신 정보는 [공식 문서](https://python.langchain.com/docs/integrations/document_loaders/)를 확인하세요.

---

### 1. Document 객체 직접 생성

```python
from langchain_core.documents import Document

# 직접 Document 객체 생성 (테스트/프로토타이핑용)
doc = Document(
    page_content="LangChain은 LLM 애플리케이션 개발을 위한 프레임워크입니다.",
    metadata={
        "source": "manual",
        "author": "developer",
        "category": "introduction",
    },
)

print(f"내용: {doc.page_content}")
print(f"메타데이터: {doc.metadata}")
# 내용: LangChain은 LLM 애플리케이션 개발을 위한 프레임워크입니다.
# 메타데이터: {'source': 'manual', 'author': 'developer', 'category': 'introduction'}
```

---

### 2. TextLoader — 로컬 텍스트 파일 로드

```python
from langchain_community.document_loaders import TextLoader

# UTF-8 인코딩 명시 권장 (특히 한국어 파일)
loader = TextLoader("data/langchain_intro.txt", encoding="utf-8")
docs = loader.load()

print(f"문서 수: {len(docs)}")
print(f"내용 미리보기: {docs[0].page_content[:200]}")
print(f"메타데이터: {docs[0].metadata}")
# 메타데이터: {'source': 'data/langchain_intro.txt'}
```

---

### 3. WebBaseLoader — 웹 페이지 로드

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

# LangChain 공식 문서 페이지 로드
loader = WebBaseLoader(
    web_paths=["https://python.langchain.com/docs/introduction/"],
    # bs4_kwargs로 파싱 범위 제한 (선택 사항)
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=("theme-doc-markdown", "markdown")
        )
    },
)

docs = loader.load()

print(f"로드된 문서 수: {len(docs)}")
print(f"내용 길이: {len(docs[0].page_content)} 문자")
print(f"메타데이터: {docs[0].metadata}")
# 메타데이터: {'source': 'https://python.langchain.com/docs/introduction/', 'title': '...'}
```

```python
# 여러 URL 동시 로드
urls = [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
]

loader = WebBaseLoader(web_paths=urls)
docs = loader.load()

print(f"총 {len(docs)}개 페이지 로드 완료")
for doc in docs:
    print(f"  - {doc.metadata['source']}: {len(doc.page_content)} 문자")
```

---

### 4. PyPDFLoader — PDF 파일 로드

```python
from langchain_community.document_loaders import PyPDFLoader

# PDF는 페이지 단위로 Document 객체 생성
loader = PyPDFLoader("data/python_tutorial.pdf")
docs = loader.load()

print(f"총 페이지 수: {len(docs)}")
for i, doc in enumerate(docs[:3]):
    print(f"\n--- 페이지 {i + 1} ---")
    print(f"내용 미리보기: {doc.page_content[:150]}")
    print(f"메타데이터: {doc.metadata}")
    # 메타데이터: {'source': 'data/python_tutorial.pdf', 'page': 0}
```

---

### 5. DirectoryLoader — 디렉토리 전체 로드

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# 특정 디렉토리의 모든 .txt 파일 로드
loader = DirectoryLoader(
    path="data/docs/",
    glob="**/*.txt",           # 재귀적으로 .txt 파일 검색
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    show_progress=True,        # 진행 상태 표시
    use_multithreading=True,   # 멀티스레딩으로 빠른 로드
)

docs = loader.load()
print(f"총 {len(docs)}개 문서 로드 완료")
```

```python
# PDF 파일 디렉토리 로드
from langchain_community.document_loaders import PyPDFLoader

pdf_loader = DirectoryLoader(
    path="data/pdfs/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
)
pdf_docs = pdf_loader.load()
```

---

### 6. lazy_load() — 메모리 효율적 로딩

```python
from langchain_community.document_loaders import WebBaseLoader

urls = [f"https://python.langchain.com/docs/concepts/" for _ in range(5)]
loader = WebBaseLoader(web_paths=urls)

# lazy_load()는 제너레이터를 반환합니다
total_chars = 0
doc_count = 0

for doc in loader.lazy_load():
    # 각 문서를 즉시 처리하고 메모리 해제
    total_chars += len(doc.page_content)
    doc_count += 1
    print(f"처리 중: {doc.metadata.get('source', 'unknown')} ({len(doc.page_content)} 문자)")

print(f"\n총 {doc_count}개 문서, {total_chars:,} 문자 처리 완료")
```

---

### 7. 로딩 후 데이터 점검

```python
def inspect_documents(docs: list[Document]) -> None:
    """로드된 문서의 품질을 점검하는 유틸리티 함수."""
    print(f"=== 문서 점검 보고서 ===")
    print(f"총 문서 수: {len(docs)}")

    # 기본 통계
    lengths = [len(doc.page_content) for doc in docs]
    print(f"평균 문서 길이: {sum(lengths) / len(lengths):.0f} 문자")
    print(f"최단 문서: {min(lengths)} 문자")
    print(f"최장 문서: {max(lengths)} 문자")

    # 빈 문서 감지
    empty_docs = [i for i, doc in enumerate(docs) if not doc.page_content.strip()]
    if empty_docs:
        print(f"⚠️  빈 문서 발견: 인덱스 {empty_docs}")

    # 메타데이터 키 확인
    all_keys = set()
    for doc in docs:
        all_keys.update(doc.metadata.keys())
    print(f"메타데이터 키: {sorted(all_keys)}")

    # 출처별 문서 수
    sources = {}
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        sources[source] = sources.get(source, 0) + 1
    print(f"\n출처별 문서 수:")
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count}개")


# 사용 예시
loader = WebBaseLoader(web_paths=["https://python.langchain.com/docs/introduction/"])
docs = loader.load()
inspect_documents(docs)
```

---

### 8. QA 시스템용 문서 로드 (프로젝트 시작)

```python
"""
프로젝트: Python 라이브러리 공식 문서 QA 시스템
Phase 11: 문서 로드 단계

이후 Phase에서 이 docs 리스트를 청킹 → 임베딩 → 저장 → 검색에 활용합니다.
"""

import bs4
from langchain_community.document_loaders import WebBaseLoader

# LangChain 공식 문서 여러 페이지 로드
LANGCHAIN_DOCS_URLS = [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
    "https://python.langchain.com/docs/concepts/lcel/",
]

loader = WebBaseLoader(
    web_paths=LANGCHAIN_DOCS_URLS,
    bs_kwargs={
        "parse_only": bs4.SoupStrainer(
            class_=("theme-doc-markdown", "markdown", "article")
        )
    },
)

raw_docs = loader.load()

# 품질 점검
inspect_documents(raw_docs)

# 빈 문서 필터링
docs = [doc for doc in raw_docs if len(doc.page_content.strip()) > 100]
print(f"\n필터링 후 문서 수: {len(docs)}")
```

---

## ✏️ 실습 과제

**과제 1**: `TextLoader`로 로컬에 저장한 텍스트 파일을 로드하고 `inspect_documents()` 함수로 점검해 보세요.

**과제 2**: `PyPDFLoader`로 PDF 파일을 로드하여 페이지별 내용과 메타데이터를 출력해 보세요. `page` 키의 값을 확인하세요.

**과제 3**: `DirectoryLoader`를 사용하여 `**/*.md` 패턴으로 마크다운 파일을 로드하고, `TextLoader`를 `loader_cls`로 지정해 보세요.

**과제 4 (심화)**: `lazy_load()`를 사용하여 10개 이상의 URL에서 문서를 로드하되, 각 문서의 `page_content` 길이가 500자 미만인 경우 건너뛰는 로직을 구현해 보세요.

---

## ⚠️ 흔한 함정

**함정 1: 인코딩 오류**
```python
# 잘못된 예: 한국어 파일에서 UnicodeDecodeError 발생 가능
loader = TextLoader("korean_doc.txt")

# 올바른 예: 인코딩 명시
loader = TextLoader("korean_doc.txt", encoding="utf-8")
```

**함정 2: 웹 로더의 불필요한 HTML 태그**
- `WebBaseLoader`는 기본적으로 전체 HTML을 파싱하므로 네비게이션 바, 푸터 등 노이즈가 포함됩니다.
- `bs4_kwargs`의 `parse_only`로 원하는 영역만 파싱하도록 제한하세요.

**함정 3: 빈 문서 미처리**
- PDF의 표지/목차 페이지, 웹 페이지의 일부 섹션은 비어 있을 수 있습니다.
- `load()` 후 빈 `page_content`를 가진 `Document`를 필터링하세요.

**함정 4: DirectoryLoader 경로 오류**
```python
# 잘못된 예: glob 패턴 없이 전체 디렉토리
loader = DirectoryLoader("data/")  # 바이너리 파일도 시도하여 오류 발생 가능

# 올바른 예: 명확한 glob 패턴
loader = DirectoryLoader("data/", glob="**/*.txt")
```

---

## ✅ 셀프 체크

- [ ] `Document` 객체의 `page_content`와 `metadata` 필드 역할을 설명할 수 있다.
- [ ] `TextLoader`, `WebBaseLoader`, `PyPDFLoader`, `DirectoryLoader` 각각의 사용 시나리오를 구분할 수 있다.
- [ ] `load()`와 `lazy_load()`의 차이점과 각각의 적합한 상황을 설명할 수 있다.
- [ ] 로드 후 빈 문서를 필터링하는 코드를 작성할 수 있다.
- [ ] `WebBaseLoader`에서 `bs4_kwargs`로 파싱 범위를 제한할 수 있다.

---

## 🔗 참고 자료

- [LangChain 문서 로더 통합 목록](https://python.langchain.com/docs/integrations/document_loaders/)
- [Document 클래스 API 레퍼런스](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html)
- [WebBaseLoader 가이드](https://python.langchain.com/docs/integrations/document_loaders/web_base/)
- [PyPDFLoader 가이드](https://python.langchain.com/docs/integrations/document_loaders/pypdfloader/)

---

← [Phase 10: 구조화된 출력](../01-langchain-core/10-structured-output.md) | [Phase 12: 텍스트 분할](12-text-splitting.md) →
