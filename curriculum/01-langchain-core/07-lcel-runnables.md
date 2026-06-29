# Phase 07: LCEL과 Runnable

> 예상 소요시간: 90분 | 난이도: ★★★☆☆ | 선행 페이즈: [06-output-parsing](06-output-parsing.md)

---

## 🎯 학습 목표

- LCEL(LangChain Expression Language)의 핵심 개념을 이해합니다.
- `|` 연산자로 `prompt | model | parser` 체인을 구성할 수 있습니다.
- `RunnablePassthrough`, `RunnableParallel`, `RunnableLambda`의 역할을 설명할 수 있습니다.
- 체인에서 `invoke`, `stream`, `batch`가 동일하게 동작함을 이해합니다.
- LCEL을 사용해야 하는 이유를 설명할 수 있습니다.

---

## 📚 핵심 개념

### 1. LCEL이란

LCEL(LangChain Expression Language)은 LangChain 컴포넌트를 **파이프 연산자(`|`)**로 연결하는 방법입니다.
Python의 Unix 파이프(쉘의 `|`)에서 영감을 받았습니다.

```python
chain = prompt | llm | parser
result = chain.invoke({"question": "..."})
```

단순해 보이지만 내부에서는 다음을 자동 처리합니다.

- 각 단계의 출력 타입을 다음 단계의 입력 타입으로 자동 변환
- 스트리밍 지원 — 체인 전체가 토큰 단위로 스트리밍 가능
- 배치 처리 지원 — 체인 전체가 병렬 배치 가능
- 비동기 지원 — `ainvoke`, `astream` 자동 지원

### 2. Runnable 인터페이스

LCEL로 연결할 수 있는 모든 컴포넌트는 **Runnable** 인터페이스를 구현합니다.

| 메서드 | 설명 |
|--------|------|
| `invoke(input)` | 단건 동기 실행 |
| `stream(input)` | 청크 단위 스트리밍 |
| `batch(inputs)` | 병렬 배치 실행 |
| `ainvoke(input)` | 단건 비동기 실행 |
| `astream(input)` | 비동기 스트리밍 |
| `abatch(inputs)` | 비동기 배치 실행 |

### 3. 핵심 Runnable 유틸리티

| 유틸리티 | 역할 |
|---------|------|
| `RunnablePassthrough` | 입력을 그대로 다음 단계로 전달 |
| `RunnableParallel` | 여러 체인을 병렬 실행하고 결과를 딕셔너리로 합침 |
| `RunnableLambda` | 일반 Python 함수를 Runnable로 래핑 |

### 4. LCEL을 쓰는 이유

- **일관성**: 모든 체인이 동일한 `invoke/stream/batch` API
- **구성 가능성**: 체인을 쉽게 조합하고 확장
- **스트리밍 자동 지원**: 구현 없이 체인 전체 스트리밍
- **LangSmith 연동**: 디버깅과 추적 자동 지원
- **병렬화**: `RunnableParallel`로 병렬 실행 간단히 구현

---

## 💻 코드 예제

### 예제 1: 기본 체인 — prompt | llm | parser

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# Phase 04~06에서 배운 컴포넌트를 LCEL로 조립
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 Python 전문가입니다. 간결하게 답변합니다."),
    ("human", "{question}"),
])
parser = StrOutputParser()

# LCEL 체인 구성
chain = prompt | llm | parser

# 세 가지 호출 방식이 모두 동작
result = chain.invoke({"question": "클로저란 무엇인가요?"})
print(result)

# 스트리밍
for chunk in chain.stream({"question": "제너레이터를 설명해주세요."}):
    print(chunk, end="", flush=True)
print()

# 배치
results = chain.batch([
    {"question": "데코레이터란?"},
    {"question": "컨텍스트 매니저란?"},
])
for r in results:
    print(r[:80])
```

### 예제 2: 체인의 중간 단계 검사

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "짧게 답변합니다."),
    ("human", "{question}"),
])

# 단계별로 결과 확인
step1 = prompt.invoke({"question": "파이썬이란?"})
print("Step 1 (ChatPromptValue):", type(step1))

step2 = llm.invoke(step1)
print("Step 2 (AIMessage):", type(step2), "-", step2.content[:50])

step3 = StrOutputParser().invoke(step2)
print("Step 3 (str):", type(step3), "-", step3[:50])

# 위 세 단계를 체인으로 표현하면
chain = prompt | llm | StrOutputParser()
print("\n체인 결과:", chain.invoke({"question": "파이썬이란?"}))
```

### 예제 3: RunnablePassthrough — 입력 보존

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "다음 컨텍스트를 참고하여 답변하세요.\n\n컨텍스트: {context}"),
    ("human", "{question}"),
])

# RAG 스타일 체인 (Phase 11에서 상세 학습)
# RunnablePassthrough는 입력 딕셔너리를 그대로 전달
def retrieve_context(question: str) -> str:
    """간단한 검색 시뮬레이션."""
    return f"{question}에 관한 검색된 문서 내용입니다."

chain = (
    {
        "context": (lambda x: x["question"]) | RunnableLambda(retrieve_context),
        "question": RunnablePassthrough(),  # question을 그대로 통과
    }
    | prompt
    | llm
    | StrOutputParser()
)

# RunnableLambda 임포트를 위해
from langchain_core.runnables import RunnableLambda

chain = (
    {
        "context": RunnableLambda(lambda x: retrieve_context(x["question"])),
        "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
    }
    | prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke({"question": "파이썬의 장점은?"})
print(result)
```

### 예제 4: RunnableParallel — 병렬 실행

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# 두 가지 관점으로 동시에 분석
pros_prompt = ChatPromptTemplate.from_messages([
    ("system", "장점만 3가지 나열합니다."),
    ("human", "{topic}의 장점은?"),
])

cons_prompt = ChatPromptTemplate.from_messages([
    ("system", "단점만 3가지 나열합니다."),
    ("human", "{topic}의 단점은?"),
])

pros_chain = pros_prompt | llm | StrOutputParser()
cons_chain = cons_prompt | llm | StrOutputParser()

# 두 체인을 병렬로 실행 — 순차 실행보다 약 2배 빠름
parallel_chain = RunnableParallel(
    pros=pros_chain,
    cons=cons_chain,
)

result = parallel_chain.invoke({"topic": "Python의 GIL"})
print("장점:", result["pros"])
print("\n단점:", result["cons"])
```

### 예제 5: RunnableLambda — Python 함수를 Runnable로

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

def preprocess(text: str) -> dict:
    """입력 전처리: 앞뒤 공백 제거 후 딕셔너리로 변환."""
    return {"question": text.strip()}

def postprocess(text: str) -> str:
    """출력 후처리: 마침표가 없으면 추가."""
    return text if text.endswith(".") else text + "."

prompt = ChatPromptTemplate.from_messages([
    ("system", "Python 개념을 한 문장으로 설명합니다."),
    ("human", "{question}"),
])

# Python 함수를 체인에 통합
chain = (
    RunnableLambda(preprocess)
    | prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(postprocess)
)

result = chain.invoke("  제너레이터란?  ")  # 앞뒤 공백 자동 처리
print(result)
```

### 예제 6: 체인 조합 — 체인을 다른 체인의 부품으로

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# 요약 체인
summarize_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "주어진 텍스트를 한 문장으로 요약합니다."),
        ("human", "{text}"),
    ])
    | llm
    | StrOutputParser()
)

# 번역 체인 — 이전 체인의 출력을 입력으로 받음
translate_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "주어진 텍스트를 {target_lang}으로 번역합니다."),
        ("human", "{text}"),
    ])
    | llm
    | StrOutputParser()
)

# 두 체인을 합쳐 "요약 후 번역" 파이프라인 구성
from langchain_core.runnables import RunnableLambda

def build_translate_input(data: dict) -> dict:
    """요약 결과와 목표 언어를 번역 체인 입력으로 변환합니다."""
    return {"text": data["summary"], "target_lang": data["target_lang"]}

# 전체 파이프라인
long_text = """
파이썬은 1991년 귀도 반 로섬이 만든 고수준 프로그래밍 언어입니다.
가독성을 최우선으로 설계되었으며, 들여쓰기로 블록을 구분합니다.
데이터 과학, 웹 개발, 자동화 등 다양한 분야에서 사용됩니다.
"""

summary = summarize_chain.invoke({"text": long_text})
print("요약:", summary)

translation = translate_chain.invoke({"text": summary, "target_lang": "영어"})
print("번역:", translation)
```

### 예제 7: 체인 디버깅 — `.with_config()`

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

chain = (
    ChatPromptTemplate.from_messages([("human", "{question}")])
    | llm
    | StrOutputParser()
)

# 실행 시 태그를 붙여 LangSmith 추적에서 식별 가능
result = chain.with_config({"tags": ["my-chain", "debug"]}).invoke(
    {"question": "Python의 type hints란?"}
)
print(result)
```

---

## ✏️ 실습 과제

### 과제 1: 3단계 파이프라인

코드 스니펫을 입력받아 (1) 코드 설명 → (2) 개선 제안 → (3) 개선된 코드 생성의 3단계 체인을 구현하세요. 각 단계를 별도 체인으로 만들고 조합하세요.

### 과제 2: 병렬 모델 비교

`RunnableParallel`을 사용하여 두 가지 다른 OpenRouter 모델(예: `openai/gpt-4o-mini`, `anthropic/claude-3.5-haiku`)이 동일한 질문에 대해 동시에 응답하도록 구현하세요.

### 과제 3: 전처리/후처리 파이프라인

`RunnableLambda`로 (1) 사용자 입력에서 코드 블록만 추출 → (2) LLM 코드 리뷰 → (3) 결과를 마크다운 형식으로 포맷 하는 체인을 구현하세요.

### 과제 4: 체인 성능 측정

`time` 모듈로 `RunnableParallel`(병렬)과 순차 실행의 처리 시간을 비교하세요.

---

## ⚠️ 흔한 함정

**1. LLMChain, ConversationChain 사용 금지**

```python
# 금지 — deprecated
from langchain.chains import LLMChain
chain = LLMChain(llm=llm, prompt=prompt)

# 올바른 LCEL 방식
chain = prompt | llm | StrOutputParser()
```

**2. `|` 연산자 우선순위**

```python
# 잘못됨 — RunnableParallel 딕셔너리 리터럴이 먼저 평가됨
chain = {"a": chain_a} | prompt | llm  # KeyError 위험

# 올바름 — RunnableParallel 명시적 사용
chain = RunnableParallel({"a": chain_a}) | prompt | llm
```

**3. `invoke` 입력 타입 주의**

```python
# 단일 변수 프롬프트라도 딕셔너리로 전달
chain.invoke("질문")         # 가능하지만 권장하지 않음
chain.invoke({"question": "질문"})  # 명시적이고 안전함
```

**4. API 변동 주의**

> LangChain은 빠르게 발전합니다. LCEL 관련 최신 사양은 [공식 문서](https://python.langchain.com/docs/concepts/lcel/)를 확인하세요.

---

## ✅ 셀프 체크

- [ ] `prompt | llm | parser` 패턴으로 체인을 구성할 수 있다.
- [ ] LCEL 체인에서 `invoke`, `stream`, `batch`가 동일하게 동작함을 확인했다.
- [ ] `RunnablePassthrough`가 입력을 그대로 통과시키는 상황을 설명할 수 있다.
- [ ] `RunnableParallel`로 두 체인을 병렬 실행할 수 있다.
- [ ] `RunnableLambda`로 일반 Python 함수를 체인에 통합할 수 있다.

---

## 🔗 참고 자료

- [LangChain LCEL 개요](https://python.langchain.com/docs/concepts/lcel/)
- [Runnable 인터페이스](https://python.langchain.com/docs/concepts/runnables/)
- [RunnableParallel](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableParallel.html)
- [RunnableLambda](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableLambda.html)

---

← [Phase 06: 출력 파싱](06-output-parsing.md) | [Phase 08: 스트리밍과 비동기](08-streaming-async.md) →
