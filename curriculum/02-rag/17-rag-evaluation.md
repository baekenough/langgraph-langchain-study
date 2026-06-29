# Phase 17: RAG 평가 (RAG Evaluation)

| 항목 | 내용 |
|------|------|
| 소요시간 | 약 75분 |
| 난이도 | ★★★★☆ |
| 선행 학습 | [Phase 16: 고급 RAG](16-advanced-rag.md) |

> **프로젝트 흐름**: Part 2의 마지막 Phase입니다. Phase 11~16에서 구축한 "Python 라이브러리 공식 문서 QA 시스템"이 실제로 잘 동작하는지 정량적으로 측정합니다. 이후 Part 3(LangGraph)에서 에이전트로 확장하고, Phase 36(LangSmith)에서 프로덕션급 평가 파이프라인을 구축합니다.

---

## 🎯 학습 목표

- RAG 평가의 3가지 핵심 지표(faithfulness, answer relevance, context relevance)를 이해합니다.
- LLM-as-judge 패턴으로 자동화된 평가기를 구현합니다.
- `ragas` 라이브러리의 개념과 활용 방향을 파악합니다.
- 간단한 평가 루프를 직접 구현하여 RAG 시스템의 품질을 측정합니다.
- Phase 36에서 다룰 LangSmith 기반 평가와의 연관성을 이해합니다.

---

## 📚 핵심 개념

### RAG 평가가 어려운 이유

RAG 시스템을 "잘 동작한다"고 판단하는 기준이 주관적입니다. 기존 분류/회귀 모델과 달리, 자유 형식 텍스트 생성에는 정답이 하나가 아닙니다.

```
질문: "itertools.chain은 무엇을 하나요?"

좋은 답변 A: "chain은 여러 이터러블을 하나로 연결합니다. chain([1,2], [3,4])는 1,2,3,4를 반환합니다."
좋은 답변 B: "itertools.chain() 함수는 복수의 이터러블을 순서대로 이어 하나의 이터레이터로 만듭니다."

두 답변 모두 정답 — 어떻게 자동으로 평가하나?
```

### 3가지 핵심 평가 지표

RAG 파이프라인은 **검색**과 **생성** 두 단계로 구성됩니다. 각 단계와 전체를 평가하는 지표가 있습니다.

```
[질문] → [검색] → [컨텍스트] → [생성] → [답변]
           │                         │
    context_relevance        faithfulness
    (검색 품질)               (생성 품질)
                                       │
                               answer_relevance
                               (답변 품질)
```

| 지표 | 측정 대상 | 핵심 질문 |
|------|-----------|-----------|
| **Context Relevance** | 검색된 문서 품질 | 검색된 청크가 질문과 얼마나 관련 있는가? |
| **Faithfulness** | 답변의 근거 충실도 | 답변이 검색된 문서에만 근거하는가? (환각 없음) |
| **Answer Relevance** | 답변의 질문 적합성 | 답변이 질문에 직접적으로 답하는가? |

### LLM-as-Judge 패턴

평가 자체를 LLM이 수행합니다. 인간 평가자의 판단력을 모방하되 자동화합니다.

```python
# 개념적 흐름
평가_입력 = {
    "question": "사용자의 질문",
    "context": "검색된 문서들",
    "answer": "RAG가 생성한 답변",
}

평가_프롬프트 = "위 입력을 바탕으로 faithfulness를 0-5점으로 채점하고 이유를 설명하세요."

점수 = 평가_LLM(평가_프롬프트 + 평가_입력)
```

**장점**: 자동화, 일관성, 확장성
**단점**: 평가 LLM 자체의 편향, 추가 API 비용

---

## 💻 코드 예제

### 환경 설정

```bash
pip install langchain langchain-openai python-dotenv
# ragas 선택적 설치:
pip install ragas
```

```python
import os
from dotenv import load_dotenv

load_dotenv()
# OPENROUTER_API_KEY: LLM용
# OPENAI_API_KEY: 임베딩용
```

### 평가 데이터셋 준비

좋은 평가는 대표적인 질문-정답 쌍(골든 데이터셋)에서 시작합니다.

```python
from dataclasses import dataclass

@dataclass
class EvalSample:
    """단일 평가 샘플을 나타냅니다."""
    question: str
    ground_truth: str  # 사람이 작성한 기준 답변


# Python itertools 문서 QA 시스템 평가 데이터셋
EVAL_DATASET: list[EvalSample] = [
    EvalSample(
        question="itertools.chain은 무엇을 하나요?",
        ground_truth="chain은 여러 이터러블을 연결하여 하나의 이터레이터로 만듭니다.",
    ),
    EvalSample(
        question="itertools.count의 step 파라미터는 무엇인가요?",
        ground_truth="step은 각 반복에서 더할 값으로 기본값은 1입니다.",
    ),
    EvalSample(
        question="combinations와 permutations의 차이는 무엇인가요?",
        ground_truth="combinations는 순서를 고려하지 않고, permutations는 순서를 고려합니다.",
    ),
    EvalSample(
        question="itertools.groupby는 어떻게 작동하나요?",
        ground_truth="groupby는 연속된 동일한 키를 가진 요소를 그룹으로 묶습니다. 사전에 정렬이 필요합니다.",
    ),
    EvalSample(
        question="islice로 무한 이터레이터를 제한하는 방법은?",
        ground_truth="islice(iterator, stop) 또는 islice(iterator, start, stop)으로 슬라이싱할 수 있습니다.",
    ),
]
```

### RAG 시스템 준비 (Phase 15~16 결과물)

```python
from pydantic import SecretStr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# LLM: OpenRouter 경유
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# 임베딩: OpenAI 직접 사용 (OpenRouter는 임베딩 미지원)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 문서 로드
loader = WebBaseLoader("https://docs.python.org/3/library/itertools.html")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="itertools_eval",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def format_docs(docs: list[Document]) -> str:
    """검색된 문서 리스트를 하나의 문자열로 포맷팅합니다."""
    return "\n\n".join(doc.page_content for doc in docs)


# 평가 대상 RAG 체인
rag_prompt = ChatPromptTemplate.from_template("""
다음 문서를 바탕으로 질문에 답변하세요. 문서에 없는 내용은 모른다고 하세요.

문서:
{context}

질문: {question}

답변:
""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)
```

---

### 평가기 구현: Faithfulness (환각 감지)

```python
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


class FaithfulnessScore(BaseModel):
    """Faithfulness 평가 결과를 나타냅니다."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="0.0(완전 환각) ~ 1.0(완전히 근거 있음)",
    )
    reasoning: str = Field(description="점수 산정 이유")
    hallucinated_claims: list[str] = Field(
        default_factory=list,
        description="문서에 근거 없는 주장 목록",
    )


faithfulness_prompt = ChatPromptTemplate.from_template("""
당신은 RAG 시스템의 답변 품질을 평가하는 전문가입니다.

[검색된 문서]
{context}

[생성된 답변]
{answer}

작업: 답변의 각 주장이 검색된 문서에 근거가 있는지 확인하세요.
- 문서에 있는 내용만을 바탕으로 한 주장: 근거 있음
- 문서에 없거나 문서와 모순되는 주장: 환각(hallucination)

faithfulness 점수(0.0~1.0)와 근거 없는 주장 목록을 JSON으로 반환하세요.
""")

faithfulness_evaluator = faithfulness_prompt | llm.with_structured_output(FaithfulnessScore)


def evaluate_faithfulness(context: str, answer: str) -> FaithfulnessScore:
    """답변이 검색된 문서에 얼마나 충실한지 평가합니다."""
    return faithfulness_evaluator.invoke({
        "context": context,
        "answer": answer,
    })
```

### 평가기 구현: Answer Relevance (답변 적합성)

```python
class AnswerRelevanceScore(BaseModel):
    """Answer Relevance 평가 결과를 나타냅니다."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="0.0(무관함) ~ 1.0(완전히 적합)",
    )
    reasoning: str = Field(description="점수 산정 이유")


answer_relevance_prompt = ChatPromptTemplate.from_template("""
당신은 RAG 시스템의 답변 적합성을 평가하는 전문가입니다.

[사용자 질문]
{question}

[생성된 답변]
{answer}

작업: 답변이 질문에 직접적으로 답하는지 평가하세요.
- 질문의 핵심을 다루는가?
- 불필요한 내용 없이 간결한가?
- 질문이 요구하는 형식으로 답하는가?

answer_relevance 점수(0.0~1.0)와 이유를 JSON으로 반환하세요.
""")

answer_relevance_evaluator = answer_relevance_prompt | llm.with_structured_output(AnswerRelevanceScore)


def evaluate_answer_relevance(question: str, answer: str) -> AnswerRelevanceScore:
    """답변이 질문에 얼마나 적합한지 평가합니다."""
    return answer_relevance_evaluator.invoke({
        "question": question,
        "answer": answer,
    })
```

### 평가기 구현: Context Relevance (컨텍스트 관련성)

```python
class ContextRelevanceScore(BaseModel):
    """Context Relevance 평가 결과를 나타냅니다."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="0.0(무관한 청크) ~ 1.0(모두 관련)",
    )
    relevant_chunk_count: int = Field(description="관련 있는 청크 수")
    total_chunk_count: int = Field(description="전체 청크 수")
    reasoning: str = Field(description="점수 산정 이유")


context_relevance_prompt = ChatPromptTemplate.from_template("""
당신은 RAG 시스템의 검색 품질을 평가하는 전문가입니다.

[사용자 질문]
{question}

[검색된 문서들 (각 --- 로 구분)]
{context}

작업: 각 문서 청크가 질문에 관련 있는지 평가하세요.
- 질문 답변에 도움이 되는 청크: 관련 있음
- 질문과 무관한 내용을 담은 청크: 관련 없음

context_relevance 점수(관련 청크 수 / 전체 청크 수)와 이유를 JSON으로 반환하세요.
""")

context_relevance_evaluator = context_relevance_prompt | llm.with_structured_output(ContextRelevanceScore)


def evaluate_context_relevance(question: str, context: str) -> ContextRelevanceScore:
    """검색된 컨텍스트가 질문에 얼마나 관련 있는지 평가합니다."""
    return context_relevance_evaluator.invoke({
        "question": question,
        "context": context,
    })
```

### 평가 루프 실행

```python
import json
from dataclasses import dataclass, asdict


@dataclass
class EvalResult:
    """단일 샘플의 전체 평가 결과를 담습니다."""

    question: str
    answer: str
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    hallucinated_claims: list[str]


def run_evaluation(
    rag_chain,
    retriever,
    dataset: list[EvalSample],
) -> list[EvalResult]:
    """전체 평가 데이터셋에 대해 RAG 평가를 실행합니다."""
    results = []

    for i, sample in enumerate(dataset):
        print(f"[{i+1}/{len(dataset)}] 평가 중: {sample.question[:40]}...")

        # 1. RAG 파이프라인 실행
        retrieved_docs = retriever.invoke(sample.question)
        context = format_docs(retrieved_docs)
        answer = rag_chain.invoke(sample.question)

        # 2. 3가지 지표 평가
        faithfulness_result = evaluate_faithfulness(context, answer)
        answer_rel_result = evaluate_answer_relevance(sample.question, answer)
        context_rel_result = evaluate_context_relevance(sample.question, context)

        result = EvalResult(
            question=sample.question,
            answer=answer,
            faithfulness=faithfulness_result.score,
            answer_relevance=answer_rel_result.score,
            context_relevance=context_rel_result.score,
            hallucinated_claims=faithfulness_result.hallucinated_claims,
        )
        results.append(result)

    return results


# 실제 실행 (API 비용 주의: 샘플 2개만)
eval_results = run_evaluation(rag_chain, retriever, EVAL_DATASET[:2])

# 결과 출력
print("\n" + "=" * 60)
print("평가 결과 요약")
print("=" * 60)

for result in eval_results:
    print(f"\n질문: {result.question}")
    print(f"  Faithfulness:      {result.faithfulness:.2f}")
    print(f"  Answer Relevance:  {result.answer_relevance:.2f}")
    print(f"  Context Relevance: {result.context_relevance:.2f}")
    if result.hallucinated_claims:
        print(f"  환각 주장: {result.hallucinated_claims}")

# 평균 점수
avg_faithfulness = sum(r.faithfulness for r in eval_results) / len(eval_results)
avg_answer_rel = sum(r.answer_relevance for r in eval_results) / len(eval_results)
avg_context_rel = sum(r.context_relevance for r in eval_results) / len(eval_results)

print(f"\n{'='*60}")
print(f"전체 평균 점수 ({len(eval_results)}개 샘플)")
print(f"  Faithfulness:      {avg_faithfulness:.3f}")
print(f"  Answer Relevance:  {avg_answer_rel:.3f}")
print(f"  Context Relevance: {avg_context_rel:.3f}")
print(f"  종합 점수:         {(avg_faithfulness + avg_answer_rel + avg_context_rel)/3:.3f}")
```

---

### ragas 라이브러리 소개

`ragas`는 RAG 평가를 위한 오픈소스 라이브러리로, 위에서 직접 구현한 평가 로직을 표준화하고 자동화합니다.

```python
# pip install ragas

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# ragas가 기대하는 데이터 형식
eval_data = {
    "question": ["itertools.chain은 무엇인가요?"],
    "answer": ["chain은 여러 이터러블을 하나로 연결합니다."],
    "contexts": [["itertools.chain(*iterables)... 여러 이터러블을 연결..."]],  # 각 질문에 대한 청크 리스트
    "ground_truth": ["chain은 여러 이터러블을 연결하는 함수입니다."],
}

dataset = Dataset.from_dict(eval_data)

# 평가 실행 (실제 LLM 호출 발생)
# results = evaluate(
#     dataset=dataset,
#     metrics=[faithfulness, answer_relevancy, context_precision],
# )
# print(results)
```

**ragas vs 직접 구현 비교**

| 항목 | ragas | 직접 구현 |
|------|-------|-----------|
| 표준화 | 논문 기반 표준 지표 | 커스터마이즈 가능 |
| 편의성 | `evaluate()` 한 번 호출 | 각 지표 직접 구현 |
| 투명성 | 내부 동작 추상화 | 평가 로직 완전 이해 |
| LLM 의존 | 기본적으로 OpenAI 사용 | 원하는 LLM 사용 가능 |
| 학습 가치 | 낮음 (블랙박스) | 높음 (원리 이해) |

> **추천**: 학습 목적으로는 직접 구현 → 운영 목적으로는 ragas + LangSmith 조합

---

### LangSmith 연계 (미리보기)

Phase 36에서 본격적으로 다루지만, 개념을 미리 파악해 두세요.

```python
# Phase 36에서 배울 LangSmith 평가 패턴
# (지금은 실행하지 않음 — LANGCHAIN_API_KEY 필요)

from langsmith import Client
from langsmith.evaluation import evaluate as ls_evaluate

# 평가 데이터셋을 LangSmith에 저장
client = Client()
# dataset = client.create_dataset("itertools-rag-eval")

# LangSmith 기반 평가:
# - 평가 결과가 웹 대시보드에 자동 기록
# - 여러 버전의 RAG 시스템 비교
# - 시간에 따른 성능 추적
# - 인간 피드백(thumbs up/down) 수집
```

LangSmith는 다음을 추가로 제공합니다:
- 모든 LLM 호출의 추적(trace) 자동 기록
- 평가 결과 시각화 대시보드
- 회귀 테스트 자동화
- 팀 협업 기능

자세한 내용은 [Phase 36: LangSmith 평가](../05-production/36-langsmith-evaluation.md)에서 다룹니다.

---

## ✏️ 실습 과제

### 과제 1: 기본 vs 고급 RAG 비교 평가

Phase 15의 기본 RAG와 Phase 16의 고급 RAG(MultiQuery 또는 Hybrid)를 동일한 데이터셋으로 평가하고 지표를 비교하세요.

```python
# 힌트: run_evaluation()에 두 체인을 각각 전달
basic_results = run_evaluation(rag_chain, base_retriever, EVAL_DATASET[:2])
# advanced_results = run_evaluation(advanced_rag_chain, advanced_retriever, EVAL_DATASET[:2])
# 두 결과의 평균 점수를 비교하세요
```

### 과제 2: 청크 크기와 Faithfulness 관계

chunk_size를 200, 500, 1000으로 바꾸면서 Faithfulness 점수가 어떻게 변하는지 실험하세요. 어떤 청크 크기에서 환각이 가장 적게 발생하는지 분석합니다.

### 과제 3: 새 질문 추가

EVAL_DATASET에 5개의 질문을 직접 추가하고 평가를 실행하세요. `ground_truth`를 먼저 직접 작성한 후, RAG 답변과 비교하세요.

### 과제 4: 평가 결과 CSV 저장

```python
import csv

def save_results_to_csv(results: list[EvalResult], filepath: str) -> None:
    """평가 결과를 CSV 파일로 저장합니다."""
    fieldnames = ["question", "answer", "faithfulness", "answer_relevance", "context_relevance"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "question": r.question,
                "answer": r.answer[:100],  # 너무 길면 잘라서 저장
                "faithfulness": r.faithfulness,
                "answer_relevance": r.answer_relevance,
                "context_relevance": r.context_relevance,
            })


save_results_to_csv(eval_results, "rag_eval_results.csv")
```

---

## ⚠️ 흔한 함정

### 1. 평가 LLM과 생성 LLM이 같을 때 편향

```python
# 문제: 생성 LLM = GPT-4o-mini, 평가 LLM = GPT-4o-mini
# 같은 모델이 자신의 출력을 평가 → 과대 평가 경향

# 해결: 평가 LLM을 더 강력한 모델로 교체
from pydantic import SecretStr

eval_llm = ChatOpenAI(
    model="openai/gpt-4o",  # 더 강한 모델로 평가
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)
```

### 2. 골든 데이터셋 없이 평가

```python
# 문제: ground_truth 없이 faithfulness만 측정
# → "답변이 문서와 일치하는가"는 측정하지만 "정답인가"는 모름

# 해결: 최소 20개 이상의 ground_truth 포함 데이터셋 준비
# 데이터셋 없으면 → 상대 비교만 가능 (A안 vs B안)
```

### 3. 평가 비용 과다

```python
# 문제: 100개 샘플 × 3개 지표 × LLM 호출 = 300번의 API 호출

# 해결 1: 샘플 수 줄이기 (대표성 있는 20개만)
# 해결 2: 빠른 모델로 평가 (gpt-4o-mini)
# 해결 3: 배치 처리
# 해결 4: 캐싱 (동일 질문 재평가 방지)

import hashlib
import json

_eval_cache: dict[str, EvalResult] = {}


def cached_evaluate(question: str, answer: str, context: str) -> dict:
    """동일한 입력에 대한 평가 결과를 캐싱합니다."""
    cache_key = hashlib.md5(f"{question}{answer}{context}".encode()).hexdigest()
    if cache_key in _eval_cache:
        return _eval_cache[cache_key]
    # 실제 평가 수행 후 캐시에 저장
    # ...
```

### 4. Context Relevance vs Context Precision/Recall 혼동

```python
# ragas에서는 더 세분화된 지표 사용:
# - context_precision: 관련 청크가 앞에 랭크되는가? (순서 중요)
# - context_recall: ground_truth를 설명하는 데 필요한 청크가 있는가?

# 우리가 구현한 context_relevance는 precision에 가깝습니다.
# recall 측정은 ground_truth가 반드시 필요합니다.
```

### 5. API 변동 주의

> **📌 참고**: `ragas`와 LangChain 평가 API는 빠르게 발전합니다. 특히 `ragas`는 버전에 따라 평가 방식과 API가 크게 변경됩니다. 항상 [ragas 공식 문서](https://docs.ragas.io/)와 [LangSmith 평가 가이드](https://docs.smith.langchain.com/evaluation)에서 최신 사용법을 확인하세요.

---

## ✅ 셀프 체크

- [ ] Faithfulness, Answer Relevance, Context Relevance의 차이를 설명할 수 있다.
- [ ] 각 지표가 RAG 파이프라인의 어느 단계를 평가하는지 매핑할 수 있다.
- [ ] LLM-as-judge 패턴의 장점(자동화)과 단점(편향, 비용)을 설명할 수 있다.
- [ ] `with_structured_output`으로 평가 결과를 Pydantic 모델로 파싱할 수 있다.
- [ ] 평가 루프를 실행하여 RAG 시스템의 종합 점수를 계산할 수 있다.
- [ ] ragas 라이브러리의 역할을 이해하고 직접 구현과의 차이를 설명할 수 있다.
- [ ] Phase 36에서 LangSmith로 평가를 확장하는 방향을 이해한다.

---

## 🔗 참고 자료

- [LangSmith 평가 개념](https://docs.smith.langchain.com/evaluation)
- [ragas 공식 문서](https://docs.ragas.io/)
- [ragas 지표 설명](https://docs.ragas.io/en/latest/concepts/metrics/index.html)
- [LangChain 평가 가이드](https://python.langchain.com/docs/guides/productionization/evaluation/)
- [RAG 평가 논문 (RAGAS)](https://arxiv.org/abs/2309.15217)

---

## Part 2 완료

축하합니다. **Part 2: RAG**를 완료했습니다.

Phase 11~17을 통해 "Python 라이브러리 공식 문서 QA 시스템"을 단계적으로 구축했습니다:

| Phase | 내용 | 결과물 |
|-------|------|--------|
| 11 | 문서 로더 | `Document` 객체 리스트 |
| 12 | 텍스트 분할 | 청크 리스트 |
| 13 | 임베딩 | 벡터 표현 |
| 14 | 벡터 스토어 | Chroma DB |
| 15 | 기본 RAG | LCEL RAG 체인 |
| 16 | 고급 RAG | MultiQuery + Hybrid 체인 |
| 17 | 평가 | 정량적 품질 지표 |

다음은 **Part 3: LangGraph**에서 RAG를 에이전트로 확장합니다.

---

← [Phase 16: 고급 RAG](16-advanced-rag.md) | [Phase 18: LangGraph 소개](../03-langgraph-core/18-langgraph-intro-stategraph.md) →
