# Phase 12: 텍스트 분할 (Text Splitting)

| 항목 | 내용 |
|------|------|
| 소요시간 | 약 75분 |
| 난이도 | ★★★☆☆ |
| 선행 학습 | [Phase 11: 문서 로더](11-document-loaders.md) |

> **프로젝트 흐름**: Phase 11에서 로드한 LangChain 공식 문서를 이 Phase에서 적절한 크기로 분할합니다. 분할된 청크는 Phase 13에서 임베딩됩니다.

---

## 🎯 학습 목표

- LLM 컨텍스트 한계 때문에 문서 분할이 필요한 이유를 이해합니다.
- `RecursiveCharacterTextSplitter`의 작동 원리와 파라미터를 설명합니다.
- `chunk_size`와 `chunk_overlap`을 사용 목적에 맞게 설정합니다.
- 토큰 기반 분할과 특수 문서(마크다운, 코드) 분할을 적용합니다.
- 시맨틱 청킹의 개념을 이해합니다.

---

## 📚 핵심 개념

### 왜 청킹(Chunking)이 필요한가?

LLM은 한 번에 처리할 수 있는 텍스트의 양(컨텍스트 윈도우)에 한계가 있습니다. 또한 임베딩 모델도 최대 입력 토큰이 제한되어 있습니다.

| 문제 | 설명 |
|------|------|
| 컨텍스트 초과 | LLM이 한 번에 읽을 수 없는 분량의 문서 |
| 관련성 저하 | 너무 큰 덩어리는 특정 질문과의 관련성이 낮아짐 |
| 검색 정밀도 | 작은 청크일수록 관련 내용만 정확히 검색 가능 |

### chunk_size와 chunk_overlap

```
원본 텍스트: [···················A···················]
                    ↓ chunk_size=100, chunk_overlap=20
청크 1:     [·······A·······]
청크 2:           [·······A·······]  ← overlap 부분이 겹침
청크 3:                 [·······A·······]
```

- **chunk_size**: 각 청크의 최대 길이. 문자 수 또는 토큰 수 기준.
- **chunk_overlap**: 연속된 청크 간에 중복되는 길이. 문맥 단절 방지.

### 분할기 비교

| 분할기 | 특징 | 사용 시나리오 |
|--------|------|--------------|
| `RecursiveCharacterTextSplitter` | 여러 구분자를 재귀적으로 적용 | 일반 텍스트 (기본 선택) |
| `CharacterTextSplitter` | 단일 구분자로 단순 분할 | 명확한 구분자가 있는 텍스트 |
| `TokenTextSplitter` | 토큰 기반 분할 | 토큰 한도가 엄격한 경우 |
| `MarkdownHeaderTextSplitter` | 마크다운 헤더 기준 분할 | 마크다운 문서 |
| `PythonCodeTextSplitter` | 파이썬 구문 단위 분할 | 파이썬 소스 코드 |
| `SemanticChunker` | 의미 유사도 기반 분할 | 의미적 일관성이 중요한 경우 |

---

## 💻 코드 예제

### 환경 설정

```bash
pip install langchain-text-splitters tiktoken langchain-experimental langchain-openai
```

---

### 1. RecursiveCharacterTextSplitter 기본 사용

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 가장 자주 사용하는 범용 분할기
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 청크 최대 길이 (문자 수)
    chunk_overlap=200,    # 청크 간 중복 길이
    length_function=len,  # 길이 계산 함수
    is_separator_regex=False,
)

# 분할기의 기본 구분자 확인
print(splitter._separators)
# ['\n\n', '\n', ' ', '']
# 단락 → 줄 → 공백 → 문자 순으로 시도합니다
```

```python
# 텍스트 직접 분할
sample_text = """
LangChain은 언어 모델(LLM)을 사용한 애플리케이션 개발을 위한 프레임워크입니다.

주요 기능으로는 다음이 있습니다:
1. 다양한 LLM 통합 지원
2. 문서 로딩 및 처리
3. 벡터 스토어 연동
4. 에이전트 구축

LangChain Expression Language(LCEL)를 통해 파이프라인을 선언적으로 구성할 수 있습니다.
LCEL은 병렬 실행, 스트리밍, 비동기 처리를 기본 지원합니다.
""" * 5  # 반복하여 분할 테스트

chunks = splitter.split_text(sample_text)
print(f"원본 길이: {len(sample_text)} 문자")
print(f"청크 수: {len(chunks)}")
for i, chunk in enumerate(chunks[:3]):
    print(f"\n청크 {i + 1} ({len(chunk)} 문자):\n{chunk[:100]}...")
```

---

### 2. Document 객체와 함께 사용

```python
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Phase 11에서 로드한 문서 사용
loader = WebBaseLoader(web_paths=["https://python.langchain.com/docs/introduction/"])
raw_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# split_documents()는 메타데이터를 유지하면서 Document 리스트를 반환합니다
split_docs = splitter.split_documents(raw_docs)

print(f"원본 문서 수: {len(raw_docs)}")
print(f"분할 후 청크 수: {len(split_docs)}")
print(f"\n첫 번째 청크:")
print(f"  내용: {split_docs[0].page_content[:200]}")
print(f"  메타데이터: {split_docs[0].metadata}")
# 메타데이터는 원본 문서에서 그대로 상속됩니다
```

---

### 3. chunk_size와 chunk_overlap 선택 가이드

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def analyze_chunks(docs, chunk_size, chunk_overlap):
    """청크 설정의 영향을 분석하는 함수."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    lengths = [len(c.page_content) for c in chunks]
    print(f"설정: size={chunk_size}, overlap={chunk_overlap}")
    print(f"  청크 수: {len(chunks)}")
    print(f"  평균 길이: {sum(lengths) / len(lengths):.0f}")
    print(f"  최단/최장: {min(lengths)}/{max(lengths)}")
    return chunks


# 다양한 설정 비교
# 일반 QA: 중간 크기 청크
qa_chunks = analyze_chunks(raw_docs, chunk_size=1000, chunk_overlap=200)

# 요약 작업: 큰 청크 (더 많은 문맥)
summary_chunks = analyze_chunks(raw_docs, chunk_size=3000, chunk_overlap=300)

# 세밀한 검색: 작은 청크 (높은 정밀도)
detail_chunks = analyze_chunks(raw_docs, chunk_size=300, chunk_overlap=50)
```

```
경험적 가이드:
- chunk_size: 500~2000 (embedding 모델 한도 내에서)
- chunk_overlap: chunk_size의 10~20%
- 질문이 짧고 구체적 → 작은 chunk_size
- 문맥이 중요한 요약/분석 → 큰 chunk_size
```

---

### 4. 토큰 기반 분할

```python
from langchain_text_splitters import TokenTextSplitter

# tiktoken 기반 토큰 카운터 사용
# 임베딩 모델이나 LLM의 토큰 한도를 정확히 제어할 때 사용합니다
token_splitter = TokenTextSplitter(
    encoding_name="cl100k_base",  # GPT-4, text-embedding-3 모델이 사용하는 인코딩
    chunk_size=512,    # 토큰 기준 (문자 수가 아님)
    chunk_overlap=64,
)

chunks = token_splitter.split_documents(raw_docs)
print(f"토큰 기반 청크 수: {len(chunks)}")
```

```python
# 직접 토큰 수 확인
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

for chunk in chunks[:3]:
    token_count = len(encoding.encode(chunk.page_content))
    char_count = len(chunk.page_content)
    print(f"토큰: {token_count}, 문자: {char_count}, 비율: {char_count/token_count:.1f}")
```

---

### 5. 마크다운 문서 분할

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

# 마크다운 헤더를 메타데이터로 추출하며 분할
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ],
    strip_headers=False,  # 헤더를 청크 내용에 포함
)

markdown_text = """
# LangChain 소개

LangChain은 LLM 애플리케이션 프레임워크입니다.

## 핵심 컴포넌트

### 문서 로더

다양한 소스에서 문서를 로드합니다.

### 텍스트 분할기

긴 문서를 청크로 나눕니다.

## 설치 방법

pip으로 설치할 수 있습니다.

```bash
pip install langchain
```
"""

md_chunks = markdown_splitter.split_text(markdown_text)
for chunk in md_chunks:
    print(f"내용: {chunk.page_content[:80]}")
    print(f"메타데이터: {chunk.metadata}")
    print()
# 메타데이터에 헤더 계층이 자동으로 추가됩니다
# {'h1': 'LangChain 소개', 'h2': '핵심 컴포넌트', 'h3': '문서 로더'}
```

```python
# 마크다운 분할 후 추가 크기 제한
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1단계: 헤더 기준 분할
md_chunks = markdown_splitter.split_text(markdown_text)

# 2단계: 여전히 너무 큰 청크는 추가 분할
char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
final_chunks = char_splitter.split_documents(md_chunks)
```

---

### 6. 파이썬 코드 분할

```python
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# 파이썬 코드에 특화된 구분자 세트 사용
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=100,
)

# 지원하는 언어 확인
print([lang.value for lang in Language])
# ['cpp', 'go', 'java', 'kotlin', 'js', 'ts', 'python', 'rst', 'ruby', ...]

python_code = """
class DocumentProcessor:
    \"\"\"문서 처리 파이프라인 클래스.\"\"\"

    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size
        )

    def process(self, docs):
        \"\"\"문서를 분할하여 반환합니다.\"\"\"
        return self.splitter.split_documents(docs)


def load_and_split(url: str, chunk_size: int = 1000) -> list:
    \"\"\"URL에서 문서를 로드하고 분할하는 편의 함수.\"\"\"
    loader = WebBaseLoader(web_paths=[url])
    docs = loader.load()
    processor = DocumentProcessor(chunk_size=chunk_size)
    return processor.process(docs)
"""

code_chunks = python_splitter.split_text(python_code)
for i, chunk in enumerate(code_chunks):
    print(f"청크 {i + 1}:\n{chunk}\n{'='*40}")
```

---

### 7. 시맨틱 청킹 소개

```python
# SemanticChunker는 의미적 유사성을 기준으로 분할합니다
# 기존 방식: 고정 크기로 분할 → 문장이 부자연스럽게 끊길 수 있음
# 시맨틱 방식: 의미가 바뀌는 지점에서 분할 → 더 자연스러운 청크

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # "percentile" | "standard_deviation" | "interquartile"
    breakpoint_threshold_amount=95,          # 상위 5%의 유사도 차이에서 분할
)

# 시맨틱 청킹은 임베딩 API 호출이 필요하므로 비용과 시간이 더 걸립니다
# 일반적으로 RecursiveCharacterTextSplitter로 시작하고
# 검색 품질이 부족할 때 시맨틱 청킹을 고려하세요
```

---

### 8. QA 시스템용 문서 분할 (프로젝트 계속)

```python
"""
프로젝트: Python 라이브러리 공식 문서 QA 시스템
Phase 12: 텍스트 분할 단계
"""

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Phase 11에서 로드한 문서 (동일한 로드 코드)
LANGCHAIN_DOCS_URLS = [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
    "https://python.langchain.com/docs/concepts/lcel/",
]

loader = WebBaseLoader(
    web_paths=LANGCHAIN_DOCS_URLS,
    bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("theme-doc-markdown", "markdown"))},
)
raw_docs = loader.load()
raw_docs = [doc for doc in raw_docs if len(doc.page_content.strip()) > 100]

# 텍스트 분할
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
split_docs = splitter.split_documents(raw_docs)

print(f"원본 문서: {len(raw_docs)}개")
print(f"분할 후 청크: {len(split_docs)}개")
print(f"\n첫 번째 청크 미리보기:")
print(f"  내용: {split_docs[0].page_content[:200]}")
print(f"  메타데이터: {split_docs[0].metadata}")

# 다음 Phase에서 split_docs를 임베딩합니다
```

---

## ✏️ 실습 과제

**과제 1**: `chunk_size`를 200, 500, 1000, 2000으로 변경하면서 청크 수와 평균 길이 변화를 관찰해 보세요.

**과제 2**: `chunk_overlap`을 0(겹침 없음)과 chunk_size의 30%로 설정했을 때 청크 경계 부분의 텍스트를 비교해 보세요.

**과제 3**: 마크다운 파일(이 커리큘럼 파일도 좋습니다)에 `MarkdownHeaderTextSplitter`를 적용하고, 헤더 메타데이터가 어떻게 구성되는지 확인하세요.

**과제 4 (심화)**: `RecursiveCharacterTextSplitter`와 `SemanticChunker`로 동일한 문서를 분할하고 결과를 비교해 보세요. 어떤 경우에 시맨틱 청킹이 더 나은 결과를 보이나요?

---

## ⚠️ 흔한 함정

**함정 1: chunk_size를 너무 작게 설정**
```python
# 너무 작은 청크 → 문맥 부족으로 검색 품질 저하
# 질문에 답하기 위한 최소 문맥이 담겨야 합니다
splitter = RecursiveCharacterTextSplitter(chunk_size=50)  # 위험
```

**함정 2: split_text() vs split_documents() 혼용**
```python
raw_text = "긴 텍스트..."

# split_text(): str 입력 → str 리스트 반환
text_chunks = splitter.split_text(raw_text)  # List[str]

# split_documents(): Document 입력 → Document 리스트 반환
doc_chunks = splitter.split_documents(docs)  # List[Document]
# ← 메타데이터 유지에는 반드시 split_documents() 사용!
```

**함정 3: 토큰 수와 문자 수 혼동**
- `text-embedding-3-small`의 최대 입력은 **8191 토큰**입니다.
- 영어 기준 1 토큰 ≈ 4 문자, 한국어는 1 토큰 ≈ 1~2 문자.
- `chunk_size=1000`(문자)는 한국어 기준 약 500~1000 토큰에 해당합니다.

**함정 4: overlap이 chunk_size보다 크거나 같음**
```python
# 오류 발생
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=150,  # chunk_size보다 크면 오류!
)
```

---

## ✅ 셀프 체크

- [ ] 문서를 분할해야 하는 이유를 LLM 컨텍스트 한계와 연결하여 설명할 수 있다.
- [ ] `RecursiveCharacterTextSplitter`의 기본 구분자 순서(`\n\n` → `\n` → ` ` → `""`)를 설명할 수 있다.
- [ ] `chunk_size`와 `chunk_overlap`의 트레이드오프를 설명할 수 있다.
- [ ] `split_text()`와 `split_documents()`의 차이를 알고 메타데이터 보존 방법을 설명할 수 있다.
- [ ] 마크다운 문서에 `MarkdownHeaderTextSplitter`를 적용할 수 있다.
- [ ] 시맨틱 청킹의 개념과 일반 청킹과의 차이점을 설명할 수 있다.

---

## 🔗 참고 자료

- [텍스트 분할기 개념 가이드](https://python.langchain.com/docs/concepts/text_splitters/)
- [RecursiveCharacterTextSplitter API](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)
- [SemanticChunker 가이드](https://python.langchain.com/docs/how_to/semantic-chunker/)
- [tiktoken (OpenAI 토크나이저)](https://github.com/openai/tiktoken)

---

← [Phase 11: 문서 로더](11-document-loaders.md) | [Phase 13: 임베딩](13-embeddings.md) →
