# Phase 10: 구조화된 출력

> 예상 소요시간: 60분 | 난이도: ★★★☆☆ | 선행 페이즈: [09-tool-calling](09-tool-calling.md)

---

## 🎯 학습 목표

- `with_structured_output()`으로 LLM이 Pydantic 모델 형식으로 응답하게 할 수 있습니다.
- Pydantic 스키마로 응답 구조를 정의하고 자동 검증할 수 있습니다.
- 정보 추출(extraction) 사용례를 구현할 수 있습니다.
- `JsonOutputParser`와 `with_structured_output()`의 차이를 설명할 수 있습니다.

---

## 📚 핵심 개념

### 1. with_structured_output()이란

Phase 06의 `JsonOutputParser`는 LLM에게 "JSON 형식으로 답해줘"라고 프롬프트로 요청합니다.
`with_structured_output()`은 **모델의 Function Calling/Tool Calling 기능을 활용**하여 더 안정적으로 구조화된 출력을 강제합니다.

| 방식 | 메커니즘 | 안정성 |
|------|---------|--------|
| `JsonOutputParser` | 프롬프트 기반 | 낮음 (모델 무시 가능) |
| `with_structured_output()` | Function Calling 기반 | 높음 (스키마 강제) |

### 2. Pydantic 스키마

Pydantic의 `BaseModel`로 출력 구조를 정의합니다.
**Field의 description**이 모델에게 각 필드의 의미를 알려주므로 반드시 명시해야 합니다.

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="사람의 이름")
    age: int = Field(description="사람의 나이")
    occupation: str = Field(description="직업")
```

### 3. 정보 추출(Extraction)

비정형 텍스트에서 구조화된 데이터를 추출하는 것은 `with_structured_output()`의 대표적 사용례입니다.

```
"김민수(32세)는 서울 소재 스타트업에서 백엔드 개발자로 일합니다."
    ↓ with_structured_output(Person)
Person(name="김민수", age=32, occupation="백엔드 개발자")
```

---

## 💻 코드 예제

### 예제 1: 기본 with_structured_output()

```python
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# 출력 스키마 정의
class ProgrammingLanguage(BaseModel):
    """프로그래밍 언어 정보."""
    name: str = Field(description="언어 이름")
    year: int = Field(description="최초 출시 연도")
    creator: str = Field(description="창시자 이름")
    paradigm: list[str] = Field(description="지원하는 프로그래밍 패러다임 목록")
    use_cases: list[str] = Field(description="주요 사용 분야 목록")

# 구조화된 출력 LLM 생성
structured_llm = llm.with_structured_output(ProgrammingLanguage)

result = structured_llm.invoke("Python 프로그래밍 언어에 대해 알려주세요.")

print(type(result))          # <class 'ProgrammingLanguage'>
print(result.name)           # Python
print(result.year)           # 1991
print(result.paradigm)       # ['객체지향', '함수형', '절차적']
print(result.model_dump())   # 딕셔너리 변환
```

### 예제 2: 텍스트에서 정보 추출

```python
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

class Person(BaseModel):
    """텍스트에서 추출한 인물 정보."""
    name: str = Field(description="인물의 이름")
    age: int | None = Field(default=None, description="인물의 나이. 언급되지 않으면 None")
    occupation: str | None = Field(default=None, description="직업. 언급되지 않으면 None")
    location: str | None = Field(default=None, description="거주 또는 근무 지역. 없으면 None")

structured_llm = llm.with_structured_output(Person)

prompt = ChatPromptTemplate.from_messages([
    ("system", "텍스트에서 인물 정보를 추출하세요. 언급되지 않은 정보는 None으로 설정합니다."),
    ("human", "{text}"),
])

chain = prompt | structured_llm

texts = [
    "이지수(29)는 판교의 테크 스타트업에서 프론트엔드 개발자로 근무하고 있습니다.",
    "박철호 교수는 연세대학교에서 컴퓨터 과학을 가르치고 있습니다.",
    "홍길동은 오늘 발표에서 훌륭한 성과를 보여주었습니다.",
]

for text in texts:
    result = chain.invoke({"text": text})
    print(f"입력: {text}")
    print(f"추출: {result.model_dump()}")
    print()
```

### 예제 3: 중첩 스키마

```python
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

class Ingredient(BaseModel):
    """레시피 재료."""
    name: str = Field(description="재료 이름")
    amount: str = Field(description="분량 (예: '200g', '2컵')")

class Recipe(BaseModel):
    """요리 레시피."""
    dish_name: str = Field(description="요리 이름")
    difficulty: str = Field(description="난이도: 쉬움/보통/어려움")
    cooking_time_minutes: int = Field(description="조리 시간 (분)")
    ingredients: list[Ingredient] = Field(description="재료 목록")
    steps: list[str] = Field(description="조리 단계 목록")

structured_llm = llm.with_structured_output(Recipe)

prompt = ChatPromptTemplate.from_messages([
    ("system", "요리 레시피를 구조화된 형식으로 제공합니다."),
    ("human", "{dish}의 간단한 레시피를 알려주세요."),
])

chain = prompt | structured_llm

result = chain.invoke({"dish": "김치볶음밥"})
print(f"요리: {result.dish_name}")
print(f"난이도: {result.difficulty}")
print(f"조리시간: {result.cooking_time_minutes}분")
print("재료:")
for ing in result.ingredients:
    print(f"  - {ing.name}: {ing.amount}")
print("조리 단계:")
for i, step in enumerate(result.steps, 1):
    print(f"  {i}. {step}")
```

### 예제 4: 선택적 필드와 검증

```python
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, SecretStr
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

class SentimentAnalysis(BaseModel):
    """감정 분석 결과."""
    sentiment: str = Field(description="감정: positive, negative, neutral 중 하나")
    confidence: float = Field(description="확신도 (0.0~1.0)")
    keywords: list[str] = Field(description="감정을 나타내는 핵심 키워드 목록")
    summary: str = Field(description="한 문장 요약")

    @field_validator("sentiment")
    @classmethod
    def validate_sentiment(cls, v: str) -> str:
        """감정 값이 유효한지 검증합니다."""
        allowed = {"positive", "negative", "neutral"}
        if v.lower() not in allowed:
            msg = f"sentiment는 {allowed} 중 하나여야 합니다."
            raise ValueError(msg)
        return v.lower()

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """확신도가 0과 1 사이인지 검증합니다."""
        if not 0.0 <= v <= 1.0:
            msg = "confidence는 0.0에서 1.0 사이여야 합니다."
            raise ValueError(msg)
        return v

structured_llm = llm.with_structured_output(SentimentAnalysis)

reviews = [
    "이 제품 정말 최고예요! 배송도 빠르고 품질도 훌륭합니다.",
    "기대 이하였어요. 사진과 실물이 너무 달라서 실망했습니다.",
    "평범한 제품입니다. 특별히 좋지도 나쁘지도 않네요.",
]

for review in reviews:
    result = structured_llm.invoke(review)
    print(f"리뷰: {review[:40]}...")
    print(f"감정: {result.sentiment} (확신도: {result.confidence:.1%})")
    print(f"키워드: {result.keywords}")
    print(f"요약: {result.summary}")
    print()
```

### 예제 5: batch로 대량 추출

```python
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

class JobPosting(BaseModel):
    """채용 공고에서 추출한 정보."""
    company: str = Field(description="회사명")
    position: str = Field(description="채용 포지션")
    required_skills: list[str] = Field(description="필수 기술 스택 목록")
    experience_years: int | None = Field(default=None, description="요구 경력 연수. 신입이면 0")

structured_llm = llm.with_structured_output(JobPosting)

prompt = ChatPromptTemplate.from_messages([
    ("system", "채용 공고에서 구조화된 정보를 추출합니다."),
    ("human", "{posting}"),
])

chain = prompt | structured_llm

postings = [
    "ABC 테크에서 5년 이상 경력의 Python 백엔드 개발자를 채용합니다. Django, FastAPI, PostgreSQL 경험 필수.",
    "스타트업 DEF에서 신입 프론트엔드 개발자를 모십니다. React, TypeScript, Next.js를 다룰 수 있는 분.",
    "GHI 금융에서 데이터 분석가 채용. SQL, Python, Tableau 능숙자 우대. 3년 이상 경력자.",
]

inputs = [{"posting": p} for p in postings]
results = chain.batch(inputs)

for posting, result in zip(postings, results):
    print(f"공고: {posting[:50]}...")
    print(f"회사: {result.company}")
    print(f"포지션: {result.position}")
    print(f"기술: {result.required_skills}")
    print(f"경력: {result.experience_years}년")
    print()
```

### 예제 6: JsonOutputParser vs with_structured_output 비교

```python
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

class BookInfo(BaseModel):
    title: str = Field(description="책 제목")
    author: str = Field(description="저자")
    genre: str = Field(description="장르")

# 방법 1: JsonOutputParser — 프롬프트 기반, dict 반환
json_prompt = ChatPromptTemplate.from_messages([
    ("system", '다음 JSON만 반환하세요: {{"title": "...", "author": "...", "genre": "..."}}'),
    ("human", "{book}을 소개해주세요."),
])
json_chain = json_prompt | llm | JsonOutputParser()

# 방법 2: with_structured_output — Function Calling 기반, Pydantic 모델 반환
structured_llm = llm.with_structured_output(BookInfo)
struct_prompt = ChatPromptTemplate.from_messages([
    ("human", "{book}에 대해 알려주세요."),
])
struct_chain = struct_prompt | structured_llm

book = "파이썬 클린 코드"

json_result = json_chain.invoke({"book": book})
struct_result = struct_chain.invoke({"book": book})

print("JsonOutputParser 결과:")
print(f"  타입: {type(json_result)}")  # dict
print(f"  데이터: {json_result}")

print("\nwith_structured_output 결과:")
print(f"  타입: {type(struct_result)}")  # BookInfo
print(f"  데이터: {struct_result.model_dump()}")
print(f"  검증 완료: {isinstance(struct_result, BookInfo)}")
```

---

## ✏️ 실습 과제

### 과제 1: 뉴스 기사 파싱기

뉴스 텍스트에서 `{"headline", "date", "author", "tags": [], "summary"}` 형태로 정보를 추출하는 체인을 구현하세요.

### 과제 2: 이력서 파서

이력서 텍스트에서 이름, 이메일, 경력 목록(회사명, 기간, 역할), 기술 스택을 추출하는 중첩 Pydantic 스키마를 설계하세요.

### 과제 3: 데이터 검증 파이프라인

`@field_validator`를 사용하여 추출된 이메일 형식(`@` 포함), 연도(1900~2100), 점수(0~100) 등을 검증하는 스키마를 구현하세요.

### 과제 4: 대량 분류기

100개 문장의 카테고리를 `batch()`로 한 번에 분류하고 카테고리별 분포를 집계하세요.

---

## ⚠️ 흔한 함정

**1. Field description 누락**

```python
# 나쁜 예 — 모델이 필드 의미를 모름
class Bad(BaseModel):
    n: str  # description 없음

# 좋은 예
class Good(BaseModel):
    name: str = Field(description="사람의 전체 이름 (성+이름)")
```

**2. 너무 엄격한 스키마**

LLM이 정보를 찾지 못하면 에러 대신 잘못된 값을 채울 수 있습니다.
불확실한 필드는 `Optional` + `None` 기본값을 사용하세요.

```python
age: int | None = Field(default=None, description="나이. 언급되지 않으면 None")
```

**3. with_structured_output은 스트리밍이 제한됨**

구조화된 출력은 완성될 때까지 전체를 기다려야 합니다.
토큰 스트리밍이 필요하면 `JsonOutputParser`를 사용하세요.

**4. API 변동 주의**

> LangChain은 빠르게 발전합니다. 구조화 출력 관련 최신 사양은 [공식 문서](https://python.langchain.com/docs/concepts/structured_outputs/)를 확인하세요.

---

## ✅ 셀프 체크

- [ ] Pydantic `BaseModel`로 출력 스키마를 정의할 수 있다.
- [ ] `Field(description=...)`로 각 필드의 의미를 모델에게 전달할 수 있다.
- [ ] `with_structured_output()`으로 LLM이 Pydantic 모델을 반환하게 할 수 있다.
- [ ] 비정형 텍스트에서 구조화된 정보를 추출하는 체인을 구현할 수 있다.
- [ ] `JsonOutputParser`와 `with_structured_output()`의 차이를 설명할 수 있다.

---

## 🔗 참고 자료

- [LangChain Structured Outputs 개요](https://python.langchain.com/docs/concepts/structured_outputs/)
- [with_structured_output() 가이드](https://python.langchain.com/docs/how_to/structured_output/)
- [Pydantic 공식 문서](https://docs.pydantic.dev/latest/)
- [정보 추출 튜토리얼](https://python.langchain.com/docs/tutorials/extraction/)

---

← [Phase 09: 툴 호출 기초](09-tool-calling.md) | [Phase 11: 문서 로더](../02-rag/11-document-loaders.md) →
