# Phase 09: 툴 호출 기초

> 예상 소요시간: 75분 | 난이도: ★★★☆☆ | 선행 페이즈: [08-streaming-async](08-streaming-async.md)

---

## 🎯 학습 목표

- `@tool` 데코레이터로 LangChain 도구를 정의할 수 있습니다.
- `bind_tools()`로 LLM에 도구를 등록할 수 있습니다.
- `tool_calls` 구조를 파싱하여 실제 Python 함수를 호출할 수 있습니다.
- `ToolMessage`로 실행 결과를 모델에 전달할 수 있습니다.
- 수동 도구 호출 루프를 구현할 수 있습니다.

---

## 📚 핵심 개념

### 1. 툴 호출(Tool Calling)이란

LLM에게 **외부 함수(도구)를 호출할 권한**을 부여하는 기능입니다.
모델은 어떤 도구를 어떤 인자로 호출할지 결정하지만, **실제 실행은 우리 코드**가 담당합니다.

```
사용자: "서울 날씨 알려줘"
  ↓
LLM: get_weather(city="서울") 를 호출해야겠다 → tool_calls 반환
  ↓
우리 코드: get_weather("서울") 실행 → "맑음, 22도"
  ↓
LLM: ToolMessage 받아 최종 응답 생성 → "서울은 현재 맑고 22도입니다."
```

### 2. @tool 데코레이터

`@tool`은 일반 Python 함수를 LangChain 도구로 변환합니다.
**docstring이 도구 설명**이 되므로 반드시 명확하게 작성해야 합니다.

```python
from langchain_core.tools import tool

@tool
def add_numbers(a: int, b: int) -> int:
    """두 정수를 더합니다. 덧셈 계산이 필요할 때 사용합니다."""
    return a + b
```

### 3. tool_calls 구조

LLM이 도구를 호출하기로 결정하면 `AIMessage.tool_calls` 리스트를 반환합니다.

```python
[{
    "name": "get_weather",           # 도구 이름
    "args": {"city": "서울"},        # 인자 딕셔너리
    "id": "call_abc123",             # 호출 고유 ID
    "type": "tool_call",
}]
```

### 4. ToolMessage

도구 실행 결과를 모델에게 전달할 때 사용합니다.
`tool_call_id`로 어느 호출의 결과인지 연결합니다.

```python
from langchain_core.messages import ToolMessage

ToolMessage(
    content="맑음, 22도",
    tool_call_id="call_abc123",
)
```

### 5. 에이전트와의 관계

이 페이즈에서는 **수동으로** 도구 호출 루프를 구현합니다.
실전에서는 LangGraph 에이전트가 이 루프를 자동 처리합니다 (Part 3에서 학습).

---

## 💻 코드 예제

### 예제 1: @tool로 도구 정의

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """지정된 도시의 현재 날씨를 반환합니다.

    Args:
        city: 날씨를 조회할 도시 이름 (예: "서울", "부산")

    Returns:
        날씨 정보 문자열
    """
    # 실제 구현에서는 Weather API 호출
    weather_db = {
        "서울": "맑음, 22°C",
        "부산": "흐림, 18°C",
        "제주": "비, 16°C",
    }
    return weather_db.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다.")

@tool
def calculate(expression: str) -> str:
    """안전하게 수학 표현식을 계산합니다.

    Args:
        expression: 계산할 수학 표현식 (예: "2 + 3 * 4")
    """
    # 실제 구현에서는 더 안전한 파서 사용
    try:
        allowed_names = {"__builtins__": {}}
        result = eval(expression, allowed_names)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"계산 오류: {e}"

# 도구 메타데이터 확인
print("도구 이름:", get_weather.name)
print("도구 설명:", get_weather.description)
print("도구 스키마:", get_weather.args_schema.model_json_schema())
```

### 예제 2: bind_tools — LLM에 도구 등록

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

@tool
def get_weather(city: str) -> str:
    """지정된 도시의 현재 날씨를 반환합니다."""
    weather_db = {"서울": "맑음, 22°C", "부산": "흐림, 18°C"}
    return weather_db.get(city, "정보 없음")

@tool
def search_web(query: str) -> str:
    """웹에서 정보를 검색합니다."""
    return f"'{query}' 검색 결과: [검색 결과 시뮬레이션]"

# 도구를 LLM에 바인딩
llm_with_tools = llm.bind_tools([get_weather, search_web])

# 도구가 필요한 질문
response = llm_with_tools.invoke("서울 날씨 알려줘")

print("content:", response.content)
print("tool_calls:", response.tool_calls)
# tool_calls가 있으면 모델이 도구 호출을 요청한 것
```

### 예제 3: tool_calls 파싱 및 실행

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

@tool
def get_weather(city: str) -> str:
    """지정된 도시의 현재 날씨를 반환합니다."""
    return {"서울": "맑음, 22°C", "부산": "흐림, 18°C"}.get(city, "정보 없음")

# 도구를 이름으로 조회할 수 있도록 딕셔너리 구성
tools = [get_weather]
tool_map = {t.name: t for t in tools}

llm_with_tools = llm.bind_tools(tools)

# 1단계: 모델이 도구 호출 결정
messages = [HumanMessage(content="서울이랑 부산 날씨 모두 알려줘")]
response = llm_with_tools.invoke(messages)
messages.append(response)  # AIMessage (tool_calls 포함)

print("1단계 - 도구 호출 요청:")
for tc in response.tool_calls:
    print(f"  {tc['name']}({tc['args']})")

# 2단계: 도구 실행 및 ToolMessage 생성
for tool_call in response.tool_calls:
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    # 실제 도구 함수 실행
    tool_func = tool_map[tool_name]
    tool_result = tool_func.invoke(tool_args)

    # ToolMessage로 결과 전달
    messages.append(
        ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"],
        )
    )

print("\n2단계 - 도구 실행 결과:")
for msg in messages[2:]:  # ToolMessage들
    print(f"  {msg.content}")

# 3단계: 최종 응답 생성
final_response = llm_with_tools.invoke(messages)
print("\n3단계 - 최종 응답:")
print(final_response.content)
```

### 예제 4: 수동 도구 호출 루프

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

@tool
def get_weather(city: str) -> str:
    """지정된 도시의 현재 날씨를 반환합니다."""
    return {"서울": "맑음, 22°C", "대전": "구름, 20°C", "부산": "흐림, 18°C"}.get(city, "정보 없음")

@tool
def get_population(city: str) -> str:
    """도시의 인구를 반환합니다."""
    pop = {"서울": "약 950만 명", "대전": "약 150만 명", "부산": "약 340만 명"}
    return pop.get(city, "정보 없음")

tools = [get_weather, get_population]
tool_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

def run_agent_loop(user_input: str, max_iterations: int = 5) -> str:
    """도구 호출이 없을 때까지 반복하는 수동 에이전트 루프."""
    messages = [HumanMessage(content=user_input)]

    for iteration in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # 도구 호출이 없으면 최종 응답
        if not response.tool_calls:
            return response.content

        # 도구 호출 처리
        for tool_call in response.tool_calls:
            tool_result = tool_map[tool_call["name"]].invoke(tool_call["args"])
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )

    return "최대 반복 횟수 도달"

result = run_agent_loop("서울과 부산의 날씨와 인구를 비교해줘")
print(result)
```

### 예제 5: 복잡한 도구 — Pydantic 스키마

```python
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

class FlightSearchInput(BaseModel):
    """항공권 검색 입력 스키마."""
    origin: str = Field(description="출발지 도시 이름")
    destination: str = Field(description="목적지 도시 이름")
    date: str = Field(description="출발 날짜 (YYYY-MM-DD 형식)")
    passengers: int = Field(default=1, description="탑승자 수")

@tool(args_schema=FlightSearchInput)
def search_flights(origin: str, destination: str, date: str, passengers: int = 1) -> str:
    """항공권을 검색합니다. 출발지, 목적지, 날짜가 필요합니다."""
    return f"{origin} → {destination} ({date}, {passengers}명): 항공편 3개 발견, 최저가 150,000원"

llm_with_tools = llm.bind_tools([search_flights])

response = llm_with_tools.invoke("다음 주 금요일에 서울에서 제주까지 2명 항공권 검색해줘")
print("tool_calls:", response.tool_calls)
```

### 예제 6: 도구 호출 여부 강제

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

@tool
def get_current_time() -> str:
    """현재 시각을 반환합니다."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# tool_choice="any" — 반드시 도구를 사용하도록 강제
llm_forced = llm.bind_tools([get_current_time], tool_choice="any")

response = llm_forced.invoke("안녕하세요!")
print("강제 도구 호출:", response.tool_calls)
# 모델이 "안녕" 같은 질문에도 get_current_time을 호출함
```

---

## ✏️ 실습 과제

### 과제 1: 계산기 + 단위 변환 도구

`@tool`로 사칙연산 계산기와 온도 변환기(섭씨↔화씨)를 만들고, 수동 루프로 "32°F는 몇 도이고, 거기에 10을 더하면?"을 처리하세요.

### 과제 2: 멀티 도구 에이전트

날씨, 인구, 관광지 3개 도구를 만들어 "서울 여행 정보를 종합해줘"라는 요청을 처리하는 루프를 구현하세요.

### 과제 3: 도구 실행 로깅

`run_agent_loop`에 각 도구 호출의 이름, 인자, 결과를 로그로 출력하는 기능을 추가하세요.

### 과제 4: 에러 처리

도구 실행 중 예외가 발생할 때 `ToolMessage`에 에러 메시지를 담아 모델이 우아하게 처리하도록 구현하세요.

---

## ⚠️ 흔한 함정

**1. docstring 없는 도구**

```python
@tool
def bad_tool(x: int) -> int:  # docstring 없음
    return x * 2

# 모델이 언제 이 도구를 써야 하는지 모름
# 항상 명확한 docstring 작성 필수
```

**2. tool_calls가 있는데 content도 있을 때**

```python
response = llm_with_tools.invoke(messages)

if response.tool_calls:
    # 도구 호출 처리 — content는 보통 비어 있음
    pass
else:
    # 최종 텍스트 응답
    print(response.content)
```

**3. ToolMessage의 tool_call_id 불일치**

`tool_call_id`는 반드시 `response.tool_calls[i]["id"]`와 정확히 일치해야 합니다.

**4. 완전 자동화된 에이전트는 LangGraph 사용**

> 이 페이즈의 수동 루프는 학습 목적입니다. 실전에서는 LangGraph의 `create_react_agent`를 사용하세요 (Part 3, Phase 18~26).
>
> LangChain은 빠르게 발전합니다. 최신 Tool Calling API는 [공식 문서](https://python.langchain.com/docs/concepts/tool_calling/)를 확인하세요.

---

## ✅ 셀프 체크

- [ ] `@tool` 데코레이터로 도구를 정의하고 docstring을 작성할 수 있다.
- [ ] `bind_tools()`로 LLM에 도구를 등록할 수 있다.
- [ ] `response.tool_calls`를 파싱하여 실제 함수를 호출할 수 있다.
- [ ] `ToolMessage`로 실행 결과를 모델에 전달할 수 있다.
- [ ] 도구 호출 루프(LLM → 도구 실행 → LLM)를 수동으로 구현할 수 있다.

---

## 🔗 참고 자료

- [LangChain Tool Calling 개요](https://python.langchain.com/docs/concepts/tool_calling/)
- [@tool 데코레이터 레퍼런스](https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html)
- [ToolMessage API 레퍼런스](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.tool.ToolMessage.html)
- [How to use tools](https://python.langchain.com/docs/how_to/tool_calling/)

---

← [Phase 08: 스트리밍과 비동기](08-streaming-async.md) | [Phase 10: 구조화된 출력](10-structured-output.md) →
