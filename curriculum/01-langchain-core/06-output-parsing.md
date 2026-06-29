# Phase 06: 출력 파싱

> 예상 소요시간: 60분 | 난이도: ★★☆☆☆ | 선행 페이즈: [05-prompt-templates](05-prompt-templates.md)

---

## 🎯 학습 목표

- `StrOutputParser`로 `AIMessage`에서 텍스트를 추출할 수 있습니다.
- `JsonOutputParser`로 LLM 응답을 파이썬 딕셔너리로 변환할 수 있습니다.
- 파싱 실패 시 `OutputParserException`을 처리하는 패턴을 이해합니다.
- LCEL 체인에 파서를 연결하는 방법을 익힙니다.

---

## 📚 핵심 개념

### 1. 출력 파서의 역할

LLM은 항상 `AIMessage` 객체를 반환합니다. 애플리케이션에서는 보통 순수 텍스트나 구조화된 데이터가 필요합니다. **출력 파서(Output Parser)**가 이 변환을 담당합니다.

```
AIMessage(content="...") → StrOutputParser → "순수 문자열"
AIMessage(content='{"key": "val"}') → JsonOutputParser → {"key": "val"}
```

### 2. 주요 파서 종류

| 파서 | 입력 | 출력 | 사용 시점 |
|------|------|------|----------|
| `StrOutputParser` | `AIMessage` | `str` | 텍스트 응답이 필요할 때 |
| `JsonOutputParser` | `AIMessage` | `dict` | JSON 응답이 필요할 때 |
| `PydanticOutputParser` | `AIMessage` | Pydantic 모델 | 스키마 검증이 필요할 때 (간단 버전) |

> 더 강력한 구조화 출력은 Phase 10의 `with_structured_output()`에서 다룹니다.

### 3. 파서 위치

LCEL 체인의 마지막에 파서를 배치합니다.

```python
chain = prompt | llm | parser
```

---

## 💻 코드 예제

### 예제 1: StrOutputParser — 텍스트 추출

```python
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "간결한 한국어로 답변합니다."),
    ("human", "{question}"),
])

parser = StrOutputParser()
chain = prompt | llm | parser

# 파서 없이 — AIMessage 객체 반환
response_obj = (prompt | llm).invoke({"question": "Python 탄생 연도는?"})
print(type(response_obj))    # <class 'langchain_core.messages.ai.AIMessage'>
print(response_obj.content)  # "Python은 1991년에..."

# 파서 포함 — 순수 문자열 반환
result = chain.invoke({"question": "Python 탄생 연도는?"})
print(type(result))   # <class 'str'>
print(result)
```

### 예제 2: JsonOutputParser — 구조화된 딕셔너리 반환

```python
import os
from dotenv import load_dotenv
from pydantic import SecretStr
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

prompt = ChatPromptTemplate.from_messages([
    ("system", """다음 형식의 JSON만 반환하세요. 다른 텍스트는 포함하지 마세요.
{{
  "name": "언어 이름",
  "year": 출시 연도(정수),
  "creator": "창시자 이름",
  "paradigm": ["패러다임 목록"]
}}"""),
    ("human", "{language}에 대한 정보를 JSON으로 제공해주세요."),
])

parser = JsonOutputParser()
chain = prompt | llm | parser

result = chain.invoke({"language": "Python"})
print(type(result))         # dict
print(result["name"])       # Python
print(result["year"])       # 1991
print(result["paradigm"])   # ['객체지향', '함수형', '절차적']
```

### 예제 3: 스트리밍과 JsonOutputParser

```python
import os
from dotenv import load_dotenv
from pydantic import SecretStr
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

prompt = ChatPromptTemplate.from_messages([
    ("system", '다음 형식의 JSON만 반환하세요: {{"items": ["항목1", "항목2", "항목3"]}}'),
    ("human", "{topic}의 핵심 특징 3가지를 JSON으로 알려주세요."),
])

chain = prompt | llm | JsonOutputParser()

# JsonOutputParser는 스트리밍 중에도 부분 JSON을 점진적으로 파싱
for partial in chain.stream({"topic": "FastAPI"}):
    print(partial)  # 점진적으로 완성되는 dict 출력
```

### 예제 4: 파싱 실패 처리

```python
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

parser = JsonOutputParser()

# 파서를 직접 호출하여 실패 케이스 테스트
try:
    result = parser.invoke("안녕하세요, 저는 JSON이 아닙니다.")
except OutputParserException as e:
    print(f"파싱 실패: {e}")
    print("대처: LLM에게 JSON만 반환하도록 지시를 더 명확히 하거나,")
    print("     with_structured_output()을 사용하세요 (Phase 10).")

# 실용적인 fallback 패턴
def safe_parse_json(text: str) -> dict:
    """JSON 파싱 실패 시 빈 딕셔너리를 반환합니다."""
    try:
        return parser.invoke(text)
    except OutputParserException:
        return {}

print(safe_parse_json('{"key": "value"}'))  # {'key': 'value'}
print(safe_parse_json("JSON이 아닌 텍스트"))  # {}
```

### 예제 5: 여러 파서 비교

```python
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# 텍스트 응답용 체인
text_prompt = ChatPromptTemplate.from_messages([
    ("system", "Python 개념을 설명합니다."),
    ("human", "{concept}를 설명해주세요."),
])
text_chain = text_prompt | llm | StrOutputParser()

# JSON 응답용 체인
json_prompt = ChatPromptTemplate.from_messages([
    ("system", '다음 JSON 형식으로만 답변하세요: {{"concept": "...", "example": "...", "difficulty": "초급/중급/고급"}}'),
    ("human", "{concept}에 대한 정보를 JSON으로 주세요."),
])
json_chain = json_prompt | llm | JsonOutputParser()

# 텍스트 체인 실행
text_result = text_chain.invoke({"concept": "클로저"})
print("텍스트 응답:", text_result[:100])

# JSON 체인 실행
json_result = json_chain.invoke({"concept": "클로저"})
print("JSON 응답:", json_result)
print("난이도:", json_result.get("difficulty"))
```

### 예제 6: 파서를 독립적으로 사용

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import AIMessage

# 파서를 LLM 없이 독립적으로 사용 (테스트, 디버깅용)
str_parser = StrOutputParser()
json_parser = JsonOutputParser()

# AIMessage에서 텍스트 추출
msg = AIMessage(content="안녕하세요!")
print(str_parser.invoke(msg))  # "안녕하세요!"

# JSON 문자열 파싱
json_msg = AIMessage(content='{"name": "Alice", "age": 30}')
print(json_parser.invoke(json_msg))  # {'name': 'Alice', 'age': 30}
```

---

## ✏️ 실습 과제

### 과제 1: 프로그래밍 언어 정보 추출기

사용자가 언어 이름을 입력하면 `{"name", "year", "creator", "use_cases": []}` 형태의 딕셔너리를 반환하는 체인을 구현하세요.

### 과제 2: 안전한 JSON 파이프라인

`JsonOutputParser` 파싱 실패 시 LLM에게 재시도를 요청하는 로직(최대 2회)을 구현하세요.

### 과제 3: 여러 항목 배치 파싱

`batch()`로 5개 언어를 동시에 JSON 형식으로 파싱하고 결과를 리스트로 수집하세요.

### 과제 4: 스트리밍 JSON 진행 상황 표시

`JsonOutputParser`의 스트리밍 동작을 활용하여 JSON이 점진적으로 완성되는 과정을 시각화하세요.

---

## ⚠️ 흔한 함정

**1. LLM이 코드 블록으로 JSON을 감쌀 때**

````
```json
{"key": "value"}
```
````

`JsonOutputParser`는 마크다운 코드 블록을 자동으로 제거하지만 불완전한 경우 파싱 오류가 납니다.
프롬프트에 "코드 블록 없이 순수 JSON만 반환하세요"를 명시하세요.

**2. StrOutputParser와 직접 `.content` 접근의 차이**

```python
# 둘 다 같은 결과지만
response.content             # AIMessage에서 직접 접근
str_parser.invoke(response)  # 파서를 통한 접근

# 체인에서는 파서만 사용 가능 — .content는 체인 중간에 쓸 수 없음
chain = prompt | llm | StrOutputParser()       # 올바름
chain = prompt | llm | (lambda x: x.content)  # 가능하지만 비권장
```

**3. JSON 스키마를 프롬프트에 포함하지 않을 때**

LLM은 지시하지 않으면 임의 형식으로 응답합니다. 원하는 JSON 구조를 시스템 메시지에 명시하세요.

**4. API 변동 주의**

> LangChain은 빠르게 발전합니다. 파서 관련 최신 사양은 [공식 문서](https://python.langchain.com/docs/concepts/output_parsers/)를 확인하세요.

---

## ✅ 셀프 체크

- [ ] `StrOutputParser`로 `AIMessage`에서 텍스트를 추출할 수 있다.
- [ ] `JsonOutputParser`로 LLM 응답을 딕셔너리로 변환할 수 있다.
- [ ] `chain = prompt | llm | parser` 패턴으로 체인을 구성할 수 있다.
- [ ] `OutputParserException`을 처리하는 패턴을 작성할 수 있다.
- [ ] 스트리밍 환경에서 `JsonOutputParser`가 어떻게 동작하는지 설명할 수 있다.

---

## 🔗 참고 자료

- [LangChain Output Parsers 개요](https://python.langchain.com/docs/concepts/output_parsers/)
- [StrOutputParser API 레퍼런스](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html)
- [JsonOutputParser API 레퍼런스](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.json.JsonOutputParser.html)

---

← [Phase 05: 프롬프트 템플릿](05-prompt-templates.md) | [Phase 07: LCEL과 Runnable](07-lcel-runnables.md) →
