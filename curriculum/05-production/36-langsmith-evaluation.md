# Phase 36: LangSmith 평가

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 105분 |
| 난이도 | ★★★★☆ |
| 선행 학습 | Phase 35 (LangSmith 트레이싱), Phase 17 (RAG 평가) |

---

## 🎯 학습 목표

- LangSmith에서 평가용 데이터셋을 생성하고 관리할 수 있습니다.
- 규칙 기반 evaluator와 LLM-as-judge evaluator의 차이를 이해합니다.
- `client.evaluate()`로 실험을 실행하고 결과를 해석합니다.
- 두 가지 이상의 실험을 LangSmith UI에서 나란히 비교합니다.
- 성능 회귀(regression)를 방지하는 평가 워크플로를 설계합니다.
- Phase 17의 RAG 평가를 LangSmith 데이터셋으로 확장하는 방법을 이해합니다.

---

## 📚 핵심 개념

### 1. 평가의 구성 요소

LLM 애플리케이션 평가는 세 가지 요소로 구성됩니다:

```
데이터셋 (Dataset)
├── 예제 1: {input: "질문", output: "기대 답변"}
├── 예제 2: {input: "질문", output: "기대 답변"}
└── ...

        ↓  evaluate()

평가 대상 함수 (Target Function)
└── LLM 체인, 에이전트, RAG 파이프라인

        ↓  각 예제마다 실행

Evaluator (채점자)
├── 규칙 기반: 정확도, 문자열 매칭, 길이 검사
└── LLM-as-judge: GPT-4가 응답 품질을 0~1로 평가
```

### 2. 데이터셋 유형

| 유형 | 설명 | 언제 사용 |
|------|------|----------|
| 수동 생성 | 개발자가 직접 작성 | 핵심 시나리오 보장 |
| 트레이스에서 추출 | 실제 사용자 입력 기반 | 프로덕션 분포 반영 |
| LLM으로 생성 | 합성 데이터 | 빠른 초기 커버리지 확보 |

### 3. Evaluator 유형

**규칙 기반 Evaluator**: 결정론적이고 빠릅니다.

```python
def exact_match(run, example):
    """예측값과 기대값의 완전 일치를 검사합니다."""
    prediction = run.outputs.get("output", "")
    expected = example.outputs.get("output", "")
    return {"key": "exact_match", "score": int(prediction == expected)}
```

**LLM-as-judge Evaluator**: 의미론적 품질을 평가합니다.

```python
# LangSmith 내장 LLM 평가자 사용
from langsmith.evaluation import LangChainStringEvaluator

evaluator = LangChainStringEvaluator("qa")  # 정확성 평가
```

### 4. Phase 17 RAG 평가와의 연결

Phase 17에서는 RAGAS 라이브러리로 개별 RAG 실행을 평가했습니다.
이번 Phase에서는 LangSmith 데이터셋을 활용해 **체계적이고 반복 가능한** 평가 파이프라인을 구축합니다:

```
Phase 17: RAGAS → 단일 실행 평가
Phase 36: LangSmith → 데이터셋 기반 체계적 비교 실험
```

---

## 💻 코드 예제

### 예제 1: 데이터셋 생성

```python
# create_dataset.py
import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()

# 데이터셋 생성
dataset_name = "파이썬-QA-기본"

# 이미 존재하는 경우 재사용
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="파이썬 기본 개념 QA 평가용 데이터셋",
    )
    print(f"데이터셋 생성: {dataset.name}")
else:
    dataset = client.read_dataset(dataset_name=dataset_name)
    print(f"기존 데이터셋 사용: {dataset.name}")

# 예제 추가
examples = [
    {
        "inputs": {"question": "파이썬에서 리스트와 튜플의 차이는?"},
        "outputs": {"answer": "리스트는 수정 가능(mutable), 튜플은 수정 불가능(immutable)합니다."},
    },
    {
        "inputs": {"question": "파이썬의 GIL이란 무엇인가요?"},
        "outputs": {"answer": "Global Interpreter Lock의 약자로, 한 번에 하나의 스레드만 파이썬 바이트코드를 실행하도록 하는 뮤텍스입니다."},
    },
    {
        "inputs": {"question": "데코레이터(@)는 어떤 역할을 하나요?"},
        "outputs": {"answer": "함수나 클래스를 수정하지 않고 기능을 추가하는 디자인 패턴입니다. 고차 함수를 간결하게 표현합니다."},
    },
    {
        "inputs": {"question": "파이썬에서 __init__과 __new__의 차이는?"},
        "outputs": {"answer": "__new__는 객체를 생성하고 __init__은 생성된 객체를 초기화합니다."},
    },
    {
        "inputs": {"question": "제너레이터와 이터레이터의 차이는?"},
        "outputs": {"answer": "이터레이터는 __iter__와 __next__를 구현한 객체이고, 제너레이터는 yield를 사용해 이터레이터를 간편하게 만드는 함수입니다."},
    },
]

client.create_examples(
    inputs=[e["inputs"] for e in examples],
    outputs=[e["outputs"] for e in examples],
    dataset_id=dataset.id,
)

print(f"예제 {len(examples)}개 추가 완료")
```

### 예제 2: 규칙 기반 evaluator로 실험 실행

```python
# evaluate_rule_based.py
import os
from dotenv import load_dotenv
from langsmith import Client, evaluate
from langsmith.schemas import Run, Example
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

load_dotenv()

client = Client()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# 평가 대상 함수 (target function)
def qa_chain(inputs: dict) -> dict:
    """질문에 답변하는 체인입니다."""
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "당신은 파이썬 전문가입니다. 간결하고 정확하게 답변하세요."),
            ("human", "{question}"),
        ])
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke({"question": inputs["question"]})
    return {"answer": answer}


# 규칙 기반 Evaluator들
def contains_key_concept(run: Run, example: Example) -> dict:
    """기대 답변의 핵심 키워드가 실제 답변에 포함되는지 확인합니다."""
    prediction = run.outputs.get("answer", "").lower()
    expected = example.outputs.get("answer", "").lower()

    # 기대 답변에서 첫 번째 단어들(핵심 개념)을 추출
    key_words = [w for w in expected.split() if len(w) > 3][:5]
    matched = sum(1 for kw in key_words if kw in prediction)
    score = matched / len(key_words) if key_words else 0

    return {"key": "keyword_coverage", "score": score}


def answer_length_check(run: Run, example: Example) -> dict:
    """답변이 너무 짧거나 길지 않은지 확인합니다."""
    answer = run.outputs.get("answer", "")
    word_count = len(answer.split())

    # 20~200 단어가 적절
    is_good_length = 20 <= word_count <= 200
    return {
        "key": "length_appropriate",
        "score": 1.0 if is_good_length else 0.0,
        "comment": f"단어 수: {word_count}",
    }


# 실험 실행
results = evaluate(
    qa_chain,
    data="파이썬-QA-기본",  # 데이터셋 이름
    evaluators=[contains_key_concept, answer_length_check],
    experiment_prefix="gpt4o-mini-baseline",
    description="GPT-4o-mini 기본 프롬프트 성능 평가",
    max_concurrency=2,  # 병렬 실행 수
)

print("평가 완료!")
print(f"실험 URL: {results.experiment_name}")

# 결과 요약
import pandas as pd
df = results.to_pandas()
print("\n평가 결과 요약:")
print(df[["inputs.question", "outputs.answer",
          "feedback.keyword_coverage", "feedback.length_appropriate"]].to_string())
```

### 예제 3: LLM-as-judge Evaluator

```python
# evaluate_llm_judge.py
import os
from dotenv import load_dotenv
from langsmith import Client, evaluate
from langsmith.schemas import Run, Example
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

load_dotenv()

client = Client()

# 평가 대상 LLM
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# Judge LLM (같은 모델을 사용해도 되지만, 더 강력한 모델이 더 신뢰할 수 있습니다)
judge_llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


def qa_chain(inputs: dict) -> dict:
    """평가 대상 QA 체인입니다."""
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "당신은 파이썬 전문가입니다."),
            ("human", "{question}"),
        ])
        | llm
        | StrOutputParser()
    )
    return {"answer": chain.invoke({"question": inputs["question"]})}


def llm_judge_correctness(run: Run, example: Example) -> dict:
    """LLM을 사용해 답변의 정확성을 평가합니다 (0~1 점수)."""
    question = example.inputs.get("question", "")
    expected = example.outputs.get("answer", "")
    prediction = run.outputs.get("answer", "")

    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 파이썬 교육 전문가입니다.
아래 질문에 대한 모범 답안과 학생 답안을 비교하여 정확성을 평가하세요.

평가 기준:
- 1.0: 완전히 정확하고 모범 답안의 핵심 내용을 모두 포함
- 0.7: 대체로 정확하지만 일부 세부사항 누락
- 0.4: 부분적으로 정확하나 중요한 오류나 누락 있음
- 0.0: 부정확하거나 관련 없는 답변

반드시 0.0~1.0 사이의 숫자만 답하세요."""),
        ("human", """질문: {question}

모범 답안: {expected}

학생 답안: {prediction}

점수 (0.0~1.0):"""),
    ])

    judge_chain = judge_prompt | judge_llm | StrOutputParser()
    raw_score = judge_chain.invoke({
        "question": question,
        "expected": expected,
        "prediction": prediction,
    })

    try:
        score = float(raw_score.strip())
        score = max(0.0, min(1.0, score))  # 범위 클리핑
    except ValueError:
        score = 0.5  # 파싱 실패 시 중간값

    return {
        "key": "llm_correctness",
        "score": score,
        "comment": f"judge 원본 응답: {raw_score.strip()}",
    }


# LLM-as-judge로 실험 실행
results = evaluate(
    qa_chain,
    data="파이썬-QA-기본",
    evaluators=[llm_judge_correctness],
    experiment_prefix="gpt4o-mini-llm-judge",
    description="LLM-as-judge 정확성 평가",
    max_concurrency=1,  # judge 호출이 많아져 속도 낮춤
)

print("LLM-as-judge 평가 완료!")
df = results.to_pandas()
print(f"\n평균 정확성 점수: {df['feedback.llm_correctness'].mean():.3f}")
```

### 예제 4: 두 실험 결과 비교 및 회귀 방지

```python
# compare_experiments.py
import os
from dotenv import load_dotenv
from langsmith import Client, evaluate
from langsmith.schemas import Run, Example
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

load_dotenv()

client = Client()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


def keyword_coverage(run: Run, example: Example) -> dict:
    """키워드 커버리지 평가기입니다."""
    prediction = run.outputs.get("answer", "").lower()
    expected = example.outputs.get("answer", "").lower()
    key_words = [w for w in expected.split() if len(w) > 3][:5]
    matched = sum(1 for kw in key_words if kw in prediction)
    score = matched / len(key_words) if key_words else 0
    return {"key": "keyword_coverage", "score": score}


# 버전 A: 기본 프롬프트
def chain_v1(inputs: dict) -> dict:
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", "당신은 파이썬 전문가입니다."),
            ("human", "{question}"),
        ])
        | llm
        | StrOutputParser()
    )
    return {"answer": chain.invoke({"question": inputs["question"]})}


# 버전 B: 개선된 프롬프트 (초보자 친화)
def chain_v2(inputs: dict) -> dict:
    chain = (
        ChatPromptTemplate.from_messages([
            ("system", """당신은 파이썬 전문가입니다.
초보자도 이해할 수 있도록 다음 형식으로 답변하세요:
1. 핵심 정의 (한 문장)
2. 간단한 예시나 비유
3. 실용적인 사용 팁"""),
            ("human", "{question}"),
        ])
        | llm
        | StrOutputParser()
    )
    return {"answer": chain.invoke({"question": inputs["question"]})}


# 두 버전 실험 실행
print("버전 A 평가 중...")
results_v1 = evaluate(
    chain_v1,
    data="파이썬-QA-기본",
    evaluators=[keyword_coverage],
    experiment_prefix="v1-basic-prompt",
)

print("버전 B 평가 중...")
results_v2 = evaluate(
    chain_v2,
    data="파이썬-QA-기본",
    evaluators=[keyword_coverage],
    experiment_prefix="v2-structured-prompt",
)

# 결과 비교
df_v1 = results_v1.to_pandas()
df_v2 = results_v2.to_pandas()

score_v1 = df_v1["feedback.keyword_coverage"].mean()
score_v2 = df_v2["feedback.keyword_coverage"].mean()

print(f"\n=== 실험 결과 비교 ===")
print(f"버전 A (기본 프롬프트):    {score_v1:.3f}")
print(f"버전 B (구조화 프롬프트):  {score_v2:.3f}")
improvement = (score_v2 - score_v1) / score_v1 * 100
print(f"개선율: {improvement:+.1f}%")

# 회귀 방지 임계값 설정 예시
REGRESSION_THRESHOLD = 0.7

if score_v2 < REGRESSION_THRESHOLD:
    print(f"\n경고: 성능이 임계값({REGRESSION_THRESHOLD}) 미만입니다!")
    print("배포 전 프롬프트를 검토하세요.")
else:
    print(f"\n성능 양호: 임계값({REGRESSION_THRESHOLD}) 이상입니다.")
```

### 예제 5: 트레이스에서 데이터셋 구축

```python
# dataset_from_traces.py
import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()

# 프로젝트의 최근 run에서 좋은 예제를 골라 데이터셋에 추가합니다
project_name = os.environ.get("LANGSMITH_PROJECT", "default")

# 최근 50개 run 조회
runs = list(client.list_runs(
    project_name=project_name,
    run_type="chain",
    limit=50,
))

print(f"조회된 run 수: {len(runs)}")

# 성공한 run만 필터링해서 데이터셋으로 변환
dataset_name = "프로덕션-샘플-데이터셋"

good_examples = []
for run in runs:
    # 오류 없이 완료된 run만 선택
    if run.error is None and run.outputs:
        example = {
            "inputs": run.inputs,
            "outputs": run.outputs,
        }
        good_examples.append(example)

print(f"사용 가능한 예제: {len(good_examples)}개")

if good_examples and not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="프로덕션 트레이스에서 추출한 평가 데이터셋",
    )

    # 최대 20개만 추가
    selected = good_examples[:20]
    client.create_examples(
        inputs=[e["inputs"] for e in selected],
        outputs=[e["outputs"] for e in selected],
        dataset_id=dataset.id,
    )
    print(f"데이터셋 '{dataset_name}' 생성 완료: {len(selected)}개 예제")
```

---

## ✏️ 실습 과제

### 과제 1: QA 데이터셋 구축 (필수)

본인이 관심 있는 주제(파이썬, LangChain, 머신러닝 등)의 Q&A 데이터셋을 최소 10개 예제로 구축합니다. `client.create_dataset()`과 `client.create_examples()`를 사용하세요.

### 과제 2: 커스텀 Evaluator 작성 (중급)

다음 두 가지 evaluator를 작성합니다:
1. **한국어 비율 확인**: 답변에서 한글 문자의 비율이 50% 이상인지 확인
2. **마크다운 구조 사용**: `#`, `-`, `*`, `1.` 등의 구조적 요소를 포함하는지 확인

### 과제 3: A/B 실험 설계 (심화)

Phase 17에서 만든 RAG 파이프라인의 retriever 설정을 비교하는 실험을 설계합니다:
- 실험 A: `k=2` (상위 2개 문서 검색)
- 실험 B: `k=5` (상위 5개 문서 검색)

동일한 데이터셋에 두 실험을 실행하고 LangSmith UI에서 결과를 비교합니다.

---

## ⚠️ 흔한 함정

### 1. 데이터셋 중복 생성

```python
# 잘못된 예: 실행할 때마다 새 데이터셋 생성
dataset = client.create_dataset(dataset_name="my-dataset")  # 이미 존재하면 오류

# 올바른 예: 존재 여부 확인 후 생성
if client.has_dataset(dataset_name="my-dataset"):
    dataset = client.read_dataset(dataset_name="my-dataset")
else:
    dataset = client.create_dataset(dataset_name="my-dataset")
```

### 2. target function 반환 형식

```python
# 잘못된 예: 문자열을 직접 반환
def my_chain(inputs: dict) -> str:  # 잘못됨
    return "답변"

# 올바른 예: 딕셔너리로 반환
def my_chain(inputs: dict) -> dict:  # 올바름
    return {"answer": "답변"}
```

### 3. max_concurrency 설정

```python
# LLM-as-judge를 사용할 때는 concurrency를 낮게 설정하세요
# (judge 호출로 인해 API 레이트 리밋 초과 위험)
results = evaluate(
    my_chain,
    data="my-dataset",
    evaluators=[llm_judge],
    max_concurrency=1,  # judge 사용 시 낮게 설정
)
```

### 4. 평가 점수 해석 주의

LLM-as-judge는 절대적 기준이 아닙니다. 같은 답변도 judge 모델에 따라 점수가 다를 수 있습니다.
항상 여러 evaluator를 조합하고, 수동 검토를 병행하세요.

---

## ✅ 셀프 체크

- [ ] `client.create_dataset()`으로 평가 데이터셋을 생성했습니다.
- [ ] `client.create_examples()`로 10개 이상의 예제를 추가했습니다.
- [ ] 규칙 기반 evaluator를 직접 작성해 `evaluate()`에 전달했습니다.
- [ ] LLM-as-judge evaluator를 구현하고 점수 파싱 오류를 처리했습니다.
- [ ] `results.to_pandas()`로 결과를 DataFrame으로 변환해 분석했습니다.
- [ ] 두 가지 실험을 실행하고 LangSmith UI에서 비교했습니다.
- [ ] 회귀 방지를 위한 임계값 기반 경고 로직을 구현했습니다.
- [ ] Phase 17 RAG 평가와의 차이점을 설명할 수 있습니다.

---

## 🔗 참고 자료

- [LangSmith 평가 가이드](https://docs.smith.langchain.com/evaluation)
- [evaluate() API 레퍼런스](https://docs.smith.langchain.com/reference/python/evaluation)
- [LLM-as-judge 패턴](https://docs.smith.langchain.com/evaluation/faq/llm-as-judge)
- [Phase 17: RAG 평가](../02-rag/17-rag-evaluation.md)

> **API 변동 안내**: `evaluate()` 함수의 파라미터와 반환 형식은 LangSmith 버전에 따라 달라질 수 있습니다. 실행 전 [공식 문서](https://docs.smith.langchain.com/evaluation)를 확인하세요.

---

← [Phase 35: LangSmith 트레이싱](35-langsmith-tracing.md) | [Phase 37: 프롬프트 관리](37-prompt-management.md) →
