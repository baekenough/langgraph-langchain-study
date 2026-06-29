# Phase 27: ReAct 에이전트

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 90분 |
| 난이도 | ★★★☆☆ |
| 선행 학습 | Phase 26 (장기 메모리 스토어), Phase 09 (도구 호출) |

---

## 🎯 학습 목표

- ReAct(Reasoning + Acting) 패턴의 작동 원리를 설명할 수 있습니다.
- `create_react_agent`로 도구를 사용하는 에이전트를 10줄 이내로 만들 수 있습니다.
- `state_modifier`로 시스템 프롬프트를 주입할 수 있습니다.
- 스트리밍으로 에이전트의 사고 과정을 실시간으로 확인할 수 있습니다.
- 체크포인터를 연결해 대화 히스토리를 유지할 수 있습니다.

---

## 📚 핵심 개념

### ReAct 패턴이란?

ReAct는 **Re**asoning(추론)과 **Act**ing(행동)을 교차하는 에이전트 패턴입니다.
2022년 Yao et al. 논문에서 제안되었으며, LLM이 다음 사이클을 반복합니다:

```
Thought → Action → Observation → Thought → Action → Observation → ... → Final Answer
(추론)     (도구 호출)  (결과 확인)    (추론)     (도구 호출)  (결과 확인)
```

실제 LangGraph 내부에서는 이 사이클이 그래프 노드 간 루프로 구현됩니다:

```
START
  │
  ▼
[agent 노드] ──── 도구 호출 없음 ───► END
     ▲                 │
     │          도구 호출 있음
     │                 ▼
     └──────── [tools 노드]
```

### create_react_agent 개요

`langgraph.prebuilt.create_react_agent`는 위 그래프를 자동으로 생성해주는 팩토리 함수입니다.
직접 `StateGraph`를 구성하지 않아도 바로 동작하는 ReAct 에이전트를 얻을 수 있습니다.

주요 파라미터:

| 파라미터 | 설명 |
|---------|------|
| `model` | 도구 호출을 지원하는 LLM |
| `tools` | 에이전트가 사용할 도구 리스트 |
| `state_modifier` | 시스템 프롬프트 또는 상태 변환 함수 |
| `checkpointer` | 대화 지속성을 위한 체크포인터 |
| `interrupt_before` | 특정 노드 전에 사람 확인 |
| `interrupt_after` | 특정 노드 후에 사람 확인 |

---

## 💻 코드 예제

### 예제 1: 기본 ReAct 에이전트

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

# 모델 초기화 (OpenRouter 경유)
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


# 도구 정의
@tool
def add(a: float, b: float) -> float:
    """두 수를 더합니다."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """두 수를 곱합니다."""
    return a * b


@tool
def web_search(query: str) -> str:
    """웹에서 정보를 검색합니다. (데모용 더미 구현)"""
    # 실제 구현에서는 Tavily, SerpAPI 등을 사용합니다.
    return f"'{query}'에 대한 검색 결과: 2024년 기준 관련 데이터가 발견되었습니다."


# 에이전트 생성 — 단 한 줄!
agent = create_react_agent(
    model=llm,
    tools=[add, multiply, web_search],
)

# 실행
result = agent.invoke(
    {"messages": [("user", "3과 4를 더한 다음 그 결과에 5를 곱해줘")]}
)

# 마지막 AI 메시지 출력
print(result["messages"][-1].content)
```

### 예제 2: 시스템 프롬프트 주입 (state_modifier)

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


@tool
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 조회합니다. (데모용 더미 구현)"""
    weather_data = {
        "서울": "맑음, 22°C",
        "부산": "흐림, 18°C",
        "제주": "비, 20°C",
    }
    return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다.")


# state_modifier: 문자열로 시스템 프롬프트 주입
system_prompt = "당신은 친절한 날씨 도우미입니다. 항상 한국어로 답변하고, 날씨 정보를 알려줄 때 적절한 옷차림 조언도 함께 제공하세요."

agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    state_modifier=system_prompt,  # 문자열 또는 SystemMessage 또는 함수
)

result = agent.invoke(
    {"messages": [("user", "서울이랑 부산 날씨 알려줘")]}
)

for message in result["messages"]:
    print(f"[{message.__class__.__name__}]: {message.content[:100]}")
    print("---")
```

### 예제 3: 스트리밍으로 실시간 추론 확인

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


@tool
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """체중(kg)과 키(m)로 BMI를 계산합니다."""
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "저체중"
    elif bmi < 25:
        category = "정상"
    elif bmi < 30:
        category = "과체중"
    else:
        category = "비만"
    return f"BMI: {bmi:.1f} ({category})"


agent = create_react_agent(
    model=llm,
    tools=[calculate_bmi],
)

# stream_mode="values": 각 스텝의 전체 상태 스트리밍
print("=== values 모드 스트리밍 ===")
for state in agent.stream(
    {"messages": [("user", "키 175cm, 몸무게 70kg인 사람의 BMI를 계산해줘")]},
    stream_mode="values",
):
    last_message = state["messages"][-1]
    print(f"[{last_message.__class__.__name__}] {last_message.content[:80]}")

print("\n=== updates 모드 스트리밍 (노드 단위) ===")
for node_name, update in agent.stream(
    {"messages": [("user", "키 160cm, 몸무게 55kg인 사람의 BMI는?")]},
    stream_mode="updates",
):
    print(f"노드 '{node_name}' 업데이트:")
    if "messages" in update:
        last_msg = update["messages"][-1]
        print(f"  [{last_msg.__class__.__name__}]: {last_msg.content[:80]}")
```

### 예제 4: 체크포인터로 대화 지속성 유지

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


@tool
def get_user_profile(user_id: str) -> str:
    """사용자 프로필을 조회합니다. (데모용 더미 구현)"""
    profiles = {
        "alice": "이름: Alice, 나이: 28, 관심사: 독서, 여행",
        "bob": "이름: Bob, 나이: 35, 관심사: 코딩, 게임",
    }
    return profiles.get(user_id, f"'{user_id}' 사용자를 찾을 수 없습니다.")


# MemorySaver로 대화 상태 유지
memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[get_user_profile],
    checkpointer=memory,
)

# thread_id로 대화 세션 식별
config = {"configurable": {"thread_id": "conversation-1"}}

# 첫 번째 메시지
print("=== 첫 번째 턴 ===")
result1 = agent.invoke(
    {"messages": [("user", "alice의 프로필을 알려줘")]},
    config=config,
)
print(result1["messages"][-1].content)

# 두 번째 메시지 — 이전 대화 맥락이 유지됨
print("\n=== 두 번째 턴 (이전 맥락 유지) ===")
result2 = agent.invoke(
    {"messages": [("user", "방금 알려준 사람이 관심 있을 만한 책을 추천해줘")]},
    config=config,
)
print(result2["messages"][-1].content)

# 다른 스레드 — 새로운 대화 세션
print("\n=== 새 세션 (다른 thread_id) ===")
config2 = {"configurable": {"thread_id": "conversation-2"}}
result3 = agent.invoke(
    {"messages": [("user", "방금 알려준 사람의 나이가 몇 살이었지?")]},
    config=config2,
)
print(result3["messages"][-1].content)
# 새 세션이므로 이전 대화를 모름
```

### 예제 5: 에이전트 상태 스키마 확인

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


@tool
def dummy_tool(x: str) -> str:
    """더미 도구입니다."""
    return f"처리됨: {x}"


agent = create_react_agent(model=llm, tools=[dummy_tool])

# 내부 그래프 구조 확인
print("노드 목록:", list(agent.nodes.keys()))
print("입력 스키마:", agent.input_schema.schema())
print("출력 스키마:", agent.output_schema.schema())

# Mermaid 다이어그램 출력 (시각화)
try:
    print("\n그래프 구조 (Mermaid):")
    print(agent.get_graph().draw_mermaid())
except Exception as e:
    print(f"시각화 불가: {e}")
```

---

## ✏️ 실습 과제

### 과제 1: 개인 비서 에이전트
다음 도구를 갖춘 개인 비서 에이전트를 만들어보세요:
- `get_current_time()`: 현재 시간 반환 (Python `datetime` 사용)
- `calculate(expression: str)`: 수식 계산 (`eval()` 주의: 안전한 버전으로 구현)
- `save_note(title: str, content: str)`: 노트 저장 (딕셔너리에 저장)
- `list_notes()`: 저장된 노트 목록 반환

요구사항:
1. 시스템 프롬프트: "당신은 체계적이고 친절한 개인 비서입니다."
2. MemorySaver로 대화 유지
3. 멀티턴 대화로 노트 저장→조회 흐름 테스트

### 과제 2: 스트리밍 챗봇
`stream_mode="messages"` 모드를 사용하여 토큰 단위로 응답이 출력되는 챗봇을 구현하세요.
힌트: `stream_mode="messages"`는 `(chunk, metadata)` 튜플을 반환합니다.

---

## ⚠️ 흔한 함정

**1. 도구 호출을 지원하지 않는 모델 사용**
```python
# 잘못된 예: tool_use를 지원하지 않는 모델
llm = ChatOpenAI(model="openai/gpt-3.5-turbo-instruct", ...)  # completion 모델

# 올바른 예: chat completion + tool_use 지원 모델
llm = ChatOpenAI(model="openai/gpt-4o-mini", ...)
```

**2. thread_id 없이 체크포인터 사용**
```python
# 잘못된 예: config 누락 → 런타임 오류
agent.invoke({"messages": [...]})  # checkpointer가 있으면 config 필수

# 올바른 예
agent.invoke({"messages": [...]}, config={"configurable": {"thread_id": "session-1"}})
```

**3. 무한 루프 위험**
도구가 항상 오류를 반환하면 에이전트가 무한히 재시도할 수 있습니다.
`create_react_agent`의 `max_iterations` 파라미터(기본값: 없음)나 재귀 제한을 설정하세요:
```python
agent.invoke(
    {"messages": [...]},
    config={"recursion_limit": 10},  # 최대 10번 반복
)
```

**4. 너무 많은 도구 제공**
모델에 20개 이상의 도구를 제공하면 올바른 도구 선택 성능이 저하됩니다.
에이전트 목적에 맞는 5~10개 이내로 제한하는 것을 권장합니다.

---

## ✅ 셀프 체크

- [ ] `create_react_agent`로 기본 에이전트를 만들고 실행할 수 있다
- [ ] ReAct 패턴의 Thought → Action → Observation 사이클을 설명할 수 있다
- [ ] `state_modifier`로 시스템 프롬프트를 주입할 수 있다
- [ ] `stream_mode="values"`와 `stream_mode="updates"`의 차이를 안다
- [ ] `MemorySaver` + `thread_id`로 멀티턴 대화를 구현할 수 있다
- [ ] `recursion_limit`으로 무한 루프를 방지할 수 있다

---

## 🔗 참고 자료

- [LangGraph ReAct Agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/)
- [create_react_agent API](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent)
- [에이전트 스트리밍](https://langchain-ai.github.io/langgraph/how-tos/streaming/)
- [ReAct 논문 (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)

> **API 변동 안내:** `create_react_agent`의 파라미터 이름은 LangGraph 버전에 따라 변경될 수 있습니다. 최신 API는 [공식 문서](https://langchain-ai.github.io/langgraph/reference/prebuilt/)를 확인하세요.

---

◀ 이전: [Phase 26: 장기 메모리 스토어](../03-langgraph-core/26-memory-store.md)
▶ 다음: [Phase 28: 직접 만드는 에이전트 그래프](./28-custom-agent-graph.md)
