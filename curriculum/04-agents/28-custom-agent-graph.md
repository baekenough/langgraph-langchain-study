# Phase 28: 직접 만드는 에이전트 그래프

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 100분 |
| 난이도 | ★★★★☆ |
| 선행 학습 | Phase 27 (ReAct 에이전트), Phase 20 (노드/엣지/라우팅) |

---

## 🎯 학습 목표

- `StateGraph`로 ReAct 루프를 처음부터 직접 구현할 수 있습니다.
- `ToolNode`와 `tools_condition`의 역할을 이해하고 활용할 수 있습니다.
- `create_react_agent`(prebuilt)와 수동 구현의 차이와 장단점을 설명할 수 있습니다.
- 에이전트 루프에 커스텀 로직(로깅, 검증, 상태 변환)을 삽입할 수 있습니다.
- 조건부 라우팅으로 에이전트 흐름을 세밀하게 제어할 수 있습니다.

---

## 📚 핵심 개념

### prebuilt vs 수동 구현

| 항목 | create_react_agent | StateGraph 직접 구현 |
|------|-------------------|---------------------|
| 코드 양 | 적음 (3~5줄) | 많음 (30~50줄) |
| 제어력 | 제한적 | 완전한 제어 |
| 커스텀 상태 | MessagesState만 | 임의 상태 추가 가능 |
| 노드 간 로직 | 불가 | 자유롭게 삽입 |
| 학습 가치 | 빠른 시작 | 내부 원리 이해 |

**언제 직접 구현하나?**
- 에이전트 상태에 메시지 외 필드(예: `step_count`, `context`, `user_data`)가 필요할 때
- 도구 호출 전후에 커스텀 검증/변환 로직이 필요할 때
- 에이전트가 특정 조건에서 다른 흐름으로 전환해야 할 때

### ToolNode와 tools_condition

`ToolNode`: 마지막 AIMessage에 포함된 tool_calls를 자동으로 실행하고 ToolMessage를 반환합니다.

`tools_condition`: 마지막 메시지가 tool_calls를 포함하면 `"tools"`로, 그렇지 않으면 `END`로 라우팅합니다.

```
[agent 노드]
     │
     ├── tool_calls 있음 → [tools 노드] ──► 다시 [agent 노드]
     │
     └── tool_calls 없음 → END
```

---

## 💻 코드 예제

### 예제 1: 기본 수동 ReAct 그래프

```python
import os
from typing import Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import SecretStr

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


# 도구 정의
@tool
def search_wikipedia(query: str) -> str:
    """Wikipedia에서 정보를 검색합니다. (데모용 더미 구현)"""
    return f"Wikipedia 검색 결과 — '{query}': 관련 항목이 발견되었습니다. 해당 주제는 다양한 측면을 가지고 있습니다."


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """단위를 변환합니다. 지원: km↔mile, kg↔lb, celsius↔fahrenheit"""
    conversions = {
        ("km", "mile"): lambda x: x * 0.621371,
        ("mile", "km"): lambda x: x * 1.60934,
        ("kg", "lb"): lambda x: x * 2.20462,
        ("lb", "kg"): lambda x: x * 0.453592,
        ("celsius", "fahrenheit"): lambda x: x * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5 / 9,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key not in conversions:
        return f"'{from_unit}'에서 '{to_unit}'으로의 변환을 지원하지 않습니다."
    result = conversions[key](value)
    return f"{value} {from_unit} = {result:.4f} {to_unit}"


tools = [search_wikipedia, convert_units]

# LLM에 도구 바인딩
llm_with_tools = llm.bind_tools(tools)


# --- 그래프 노드 정의 ---

def agent_node(state: MessagesState) -> dict:
    """에이전트 노드: LLM을 호출하여 다음 행동을 결정합니다."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ToolNode는 도구 목록을 받아 자동으로 tool_calls를 처리합니다.
tool_node = ToolNode(tools=tools)


# --- 그래프 빌드 ---

graph = StateGraph(MessagesState)

# 노드 추가
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

# 엣지 정의
graph.add_edge(START, "agent")

# 조건부 엣지: tools_condition이 라우팅을 결정
graph.add_conditional_edges(
    "agent",
    tools_condition,  # tool_calls 있으면 "tools", 없으면 END
)

# 도구 실행 후 다시 에이전트로
graph.add_edge("tools", "agent")

# 컴파일
agent = graph.compile()

# 실행
result = agent.invoke(
    {"messages": [("user", "서울에서 부산까지 325km인데, 마일로 변환해줘")]}
)
print(result["messages"][-1].content)
```

### 예제 2: 커스텀 상태와 로깅이 있는 에이전트

```python
import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import SecretStr

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


# 커스텀 상태: messages 외에 추가 필드 포함
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    step_count: int          # 몇 번 루프를 돌았는지
    tool_calls_made: list[str]  # 사용된 도구 이름 기록


@tool
def get_stock_price(ticker: str) -> str:
    """주식 가격을 조회합니다. (데모용 더미 구현)"""
    prices = {"AAPL": 175.50, "GOOGL": 141.25, "MSFT": 378.90, "TSLA": 248.50}
    price = prices.get(ticker.upper())
    if price is None:
        return f"'{ticker}' 티커를 찾을 수 없습니다."
    return f"{ticker.upper()} 현재가: ${price:.2f}"


@tool
def calculate_portfolio_value(tickers: list[str], quantities: list[int]) -> str:
    """포트폴리오 총 가치를 계산합니다. (데모용 더미 구현)"""
    prices = {"AAPL": 175.50, "GOOGL": 141.25, "MSFT": 378.90, "TSLA": 248.50}
    total = 0.0
    breakdown = []
    for ticker, qty in zip(tickers, quantities):
        price = prices.get(ticker.upper(), 0)
        value = price * qty
        total += value
        breakdown.append(f"  {ticker}: {qty}주 × ${price:.2f} = ${value:.2f}")
    return "포트폴리오 구성:\n" + "\n".join(breakdown) + f"\n총 가치: ${total:.2f}"


tools = [get_stock_price, calculate_portfolio_value]
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState) -> dict:
    """커스텀 로직이 포함된 에이전트 노드."""
    step = state.get("step_count", 0)
    print(f"[에이전트 노드] 스텝 #{step + 1}")

    response = llm_with_tools.invoke(state["messages"])

    # tool_calls 정보 수집
    new_tool_calls = []
    if isinstance(response, AIMessage) and response.tool_calls:
        new_tool_calls = [tc["name"] for tc in response.tool_calls]
        print(f"  → 도구 호출 예정: {new_tool_calls}")

    return {
        "messages": [response],
        "step_count": step + 1,
        "tool_calls_made": state.get("tool_calls_made", []) + new_tool_calls,
    }


def tool_node_with_logging(state: AgentState) -> dict:
    """로깅이 포함된 커스텀 도구 노드."""
    print("[도구 노드] 도구 실행 중...")
    tool_node = ToolNode(tools=tools)
    result = tool_node.invoke(state)
    print(f"  → 도구 실행 완료, 결과 메시지 {len(result['messages'])}개")
    return result


def should_continue(state: AgentState) -> str:
    """커스텀 라우팅 함수: 최대 스텝 제한 포함."""
    step_count = state.get("step_count", 0)

    # 안전 장치: 5번 이상 루프하면 강제 종료
    if step_count >= 5:
        print(f"[경고] 최대 스텝({step_count})에 도달. 강제 종료.")
        return END

    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# 그래프 빌드
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node_with_logging)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

agent = graph.compile()

# 실행
initial_state = {
    "messages": [("user", "AAPL, MSFT 가격 각각 알려주고, 각각 10주씩 가지고 있을 때 총 가치도 계산해줘")],
    "step_count": 0,
    "tool_calls_made": [],
}

result = agent.invoke(initial_state)

print("\n=== 최종 결과 ===")
print(result["messages"][-1].content)
print(f"\n총 스텝 수: {result['step_count']}")
print(f"사용된 도구: {result['tool_calls_made']}")
```

### 예제 3: 조건부 흐름 — 도구 결과에 따라 다른 경로

```python
import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    has_error: bool


@tool
def risky_operation(operation: str) -> str:
    """위험할 수 있는 작업을 수행합니다. (에러 시뮬레이션용)"""
    if "delete" in operation.lower() or "drop" in operation.lower():
        raise ValueError(f"보안 정책: '{operation}' 작업은 허용되지 않습니다.")
    return f"작업 완료: {operation}"


@tool
def safe_operation(operation: str) -> str:
    """안전한 작업을 수행합니다."""
    return f"안전하게 완료: {operation}"


tools = [risky_operation, safe_operation]
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: State) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response], "has_error": False}


def safe_tool_node(state: State) -> dict:
    """에러를 잡아서 상태에 기록하는 도구 노드."""
    try:
        tool_node = ToolNode(tools=tools)
        result = tool_node.invoke(state)
        # 도구 결과에서 에러 확인
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage) and "오류" in msg.content.lower():
                return {**result, "has_error": True}
        return {**result, "has_error": False}
    except Exception as e:
        # 예외를 ToolMessage로 변환
        last_ai_msg = next(
            m for m in reversed(state["messages"]) if isinstance(m, AIMessage)
        )
        tool_call_id = last_ai_msg.tool_calls[0]["id"] if last_ai_msg.tool_calls else "unknown"
        error_msg = ToolMessage(
            content=f"도구 실행 오류: {str(e)}",
            tool_call_id=tool_call_id,
        )
        return {"messages": [error_msg], "has_error": True}


def error_handler(state: State) -> dict:
    """에러 발생 시 사용자에게 안전한 방법 안내."""
    from langchain_core.messages import AIMessage
    recovery_message = AIMessage(
        content="죄송합니다. 요청하신 작업에 오류가 발생했습니다. "
                "더 안전한 방법을 사용해 드리겠습니다. "
                "`safe_operation` 도구를 통해 작업을 진행하겠습니다."
    )
    return {"messages": [recovery_message], "has_error": False}


def route_after_tools(state: State) -> Literal["error_handler", "agent", "__end__"]:
    """도구 실행 후 라우팅: 에러면 error_handler, 아니면 agent로."""
    if state.get("has_error"):
        return "error_handler"
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "agent"
    return "__end__"


def route_after_agent(state: State) -> Literal["tools", "__end__"]:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return "__end__"


graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", safe_tool_node)
graph.add_node("error_handler", error_handler)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", route_after_agent)
graph.add_conditional_edges("tools", route_after_tools)
graph.add_edge("error_handler", "agent")

agent = graph.compile()

# 정상 요청
print("=== 정상 요청 ===")
r1 = agent.invoke({
    "messages": [("user", "데이터 읽기 작업을 실행해줘")],
    "has_error": False,
})
print(r1["messages"][-1].content)

# 위험한 요청
print("\n=== 위험한 요청 ===")
r2 = agent.invoke({
    "messages": [("user", "delete all records 작업을 실행해줘")],
    "has_error": False,
})
print(r2["messages"][-1].content)
```

---

## ✏️ 실습 과제

### 과제 1: 단계 추적 에이전트
`step_count`와 `visited_tools` 필드가 있는 커스텀 상태를 설계하고, 동일한 도구가 3번 이상 호출되면 종료하는 안전 장치를 구현해보세요.

### 과제 2: prebuilt 내부 재현
`create_react_agent`의 소스코드를 읽고(GitHub에서 확인), 동일한 동작을 `StateGraph`로 재현해보세요. 그리고 여기에 "도구 호출 횟수 제한" 기능을 추가해보세요.

---

## ⚠️ 흔한 함정

**1. ToolNode에 도구 목록 누락**
```python
# 잘못된 예: 빈 ToolNode (어떤 도구도 실행하지 못함)
tool_node = ToolNode(tools=[])

# 올바른 예
tool_node = ToolNode(tools=[search_wikipedia, convert_units])
```

**2. tools_condition vs 커스텀 라우팅**
`tools_condition`은 내부적으로 `END`와 `"tools"` 중 하나를 반환합니다.
커스텀 라우팅 함수에서 `END`를 반환하려면 `langgraph.graph.END` 상수를 import해야 합니다:
```python
from langgraph.graph import END

def my_router(state):
    return END  # 문자열 "__end__"가 아닌 상수 사용
```

**3. 상태 업데이트 누락**
커스텀 노드에서 상태 필드를 빠뜨리면 기본값 `None`이 됩니다:
```python
# 잘못된 예: step_count 업데이트 누락
def agent_node(state):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # step_count 누락!

# 올바른 예
def agent_node(state):
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "step_count": state.get("step_count", 0) + 1,
    }
```

---

## ✅ 셀프 체크

- [ ] `StateGraph`로 기본 ReAct 루프를 직접 구현할 수 있다
- [ ] `ToolNode`와 `tools_condition`의 역할을 설명할 수 있다
- [ ] `TypedDict`로 커스텀 상태 스키마를 정의할 수 있다
- [ ] 조건부 라우팅 함수를 작성할 수 있다
- [ ] prebuilt와 수동 구현의 장단점을 비교해서 설명할 수 있다
- [ ] 도구 실행 오류를 안전하게 처리하는 방법을 안다

---

## 🔗 참고 자료

- [LangGraph 에이전트 구현 가이드](https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/)
- [ToolNode API](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_node.ToolNode)
- [MessagesState](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.message.MessagesState)

> **API 변동 안내:** `tools_condition`의 반환값 형식은 LangGraph 버전에 따라 변경될 수 있습니다. 최신 API는 [공식 문서](https://langchain-ai.github.io/langgraph/reference/prebuilt/)를 확인하세요.

---

◀ 이전: [Phase 27: ReAct 에이전트](./27-react-agent.md)
▶ 다음: [Phase 29: 도구 설계 패턴](./29-tool-design-patterns.md)
