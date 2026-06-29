# Phase 29: 도구 설계 패턴

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 90분 |
| 난이도 | ★★★☆☆ |
| 선행 학습 | Phase 28 (직접 만드는 에이전트 그래프), Phase 09 (도구 호출) |

---

## 🎯 학습 목표

- LLM이 올바르게 선택하는 좋은 도구 스키마를 설계할 수 있습니다.
- 도구의 인자 검증과 에러 처리 패턴을 구현할 수 있습니다.
- `InjectedState`와 `InjectedToolCallId`로 상태 기반 도구를 만들 수 있습니다.
- `ToolNode`의 내부 동작과 에러 전파 방식을 이해합니다.
- 사람 확인(Human-in-the-loop)이 필요한 도구를 설계할 수 있습니다.

---

## 📚 핵심 개념

### 좋은 도구 스키마의 조건

LLM은 도구의 이름, docstring, 파라미터 타입, 파라미터 설명을 보고 어떤 도구를 언제 사용할지 결정합니다.
**모호한 스키마 = 잘못된 도구 선택 = 에이전트 실패**

```
나쁜 예: def process(data: str) -> str:
             """데이터 처리"""  ← 무엇을? 어떻게?

좋은 예: def search_products(
             query: str,         ← 검색할 상품명 또는 키워드
             category: str = "", ← 상품 카테고리 (빈 문자열이면 전체 검색)
             max_results: int = 5 ← 반환할 최대 결과 수 (1~20)
         ) -> list[dict]:
             """상품 카탈로그에서 키워드로 상품을 검색합니다.
             카테고리를 지정하면 해당 카테고리 내에서만 검색합니다."""
```

### InjectedState — 상태에서 값 주입

도구가 그래프 상태의 값을 읽어야 할 때, LLM에게는 숨기고 런타임에 자동 주입합니다.

```python
from langgraph.prebuilt import InjectedState

@tool
def get_user_data(user_id: str, state: Annotated[dict, InjectedState]) -> str:
    # state["user_id"]처럼 그래프 상태 접근 가능
    # LLM의 tool schema에는 state가 보이지 않음
    ...
```

### ToolNode의 에러 처리

기본적으로 `ToolNode`는 도구에서 발생한 예외를 잡아 `ToolMessage`로 변환합니다:
- `handle_tool_errors=True` (기본값): 예외 → ToolMessage(content="오류 메시지")
- `handle_tool_errors=False`: 예외가 그대로 전파되어 그래프 실행 중단

---

## 💻 코드 예제

### 예제 1: 좋은 도구 스키마 설계

```python
import os
from typing import Literal
from pydantic import BaseModel, Field, SecretStr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, StructuredTool
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


# --- 나쁜 도구 설계 ---
@tool
def bad_search(q: str) -> str:
    """검색"""  # 불충분한 docstring
    return f"결과: {q}"


# --- 좋은 도구 설계 ---
@tool
def search_knowledge_base(
    query: str,
    category: Literal["tech", "science", "history", "general"] = "general",
    max_results: int = 5,
) -> str:
    """내부 지식 베이스에서 정보를 검색합니다.

    Args:
        query: 검색할 키워드 또는 질문. 구체적일수록 좋은 결과를 얻습니다.
        category: 검색 카테고리. 'general'이면 전체 카테고리에서 검색합니다.
        max_results: 반환할 최대 결과 수. 1에서 20 사이의 값을 입력하세요.

    Returns:
        검색 결과를 포함한 문자열. 결과가 없으면 빈 결과 메시지를 반환합니다.
    """
    # 데모용 더미 구현
    return (
        f"[{category}] '{query}' 검색 결과 (최대 {max_results}개):\n"
        f"  1. 관련 문서: {query}의 정의와 배경\n"
        f"  2. 관련 문서: {query}의 주요 특징\n"
        f"  3. 관련 문서: {query} 활용 사례"
    )


# Pydantic 스키마로 도구 정의 (복잡한 입력에 유용)
class DatabaseQueryInput(BaseModel):
    table: str = Field(description="조회할 테이블 이름 (예: users, orders, products)")
    conditions: dict = Field(
        default_factory=dict,
        description="필터 조건. 키=컬럼명, 값=조건값 (예: {'status': 'active'})",
    )
    limit: int = Field(default=10, ge=1, le=100, description="반환할 최대 행 수 (1~100)")


def query_database(table: str, conditions: dict, limit: int) -> str:
    """데이터베이스에서 레코드를 조회합니다."""
    # 데모용 더미 구현
    cond_str = ", ".join(f"{k}={v}" for k, v in conditions.items()) or "조건 없음"
    return f"테이블 '{table}'에서 {limit}개 레코드 조회 완료. 조건: {cond_str}"


db_tool = StructuredTool.from_function(
    func=query_database,
    name="query_database",
    description="데이터베이스에서 특정 조건으로 레코드를 조회합니다.",
    args_schema=DatabaseQueryInput,
)

agent = create_react_agent(
    model=llm,
    tools=[search_knowledge_base, db_tool],
)

result = agent.invoke({
    "messages": [("user", "tech 카테고리에서 'machine learning' 검색해줘")]
})
print(result["messages"][-1].content)
```

### 예제 2: 인자 검증과 에러 핸들링

```python
import os
from typing import Annotated
from typing_extensions import TypedDict
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


@tool
def transfer_money(
    from_account: str,
    to_account: str,
    amount: float,
) -> str:
    """계좌 간 금액을 이체합니다.

    Args:
        from_account: 출금 계좌 번호 (형식: XXXX-XXXX-XXXX)
        to_account: 입금 계좌 번호 (형식: XXXX-XXXX-XXXX)
        amount: 이체 금액 (원). 100원 이상 10,000,000원 이하.

    Returns:
        이체 성공 메시지 또는 오류 메시지.
    """
    # 인자 검증 — 도구 내에서 수행
    import re
    account_pattern = re.compile(r"^\d{4}-\d{4}-\d{4}$")

    if not account_pattern.match(from_account):
        # 예외 대신 에러 메시지 반환 → LLM이 오류 내용을 읽고 수정 가능
        return f"오류: 출금 계좌 번호 형식이 잘못되었습니다. 'XXXX-XXXX-XXXX' 형식을 사용하세요."

    if not account_pattern.match(to_account):
        return f"오류: 입금 계좌 번호 형식이 잘못되었습니다. 'XXXX-XXXX-XXXX' 형식을 사용하세요."

    if amount < 100:
        return "오류: 최소 이체 금액은 100원입니다."

    if amount > 10_000_000:
        return "오류: 1회 최대 이체 금액은 10,000,000원입니다."

    # 정상 처리
    return (
        f"이체 완료: {from_account} → {to_account}, "
        f"금액: {amount:,.0f}원"
    )


@tool
def get_account_balance(account: str) -> str:
    """계좌 잔액을 조회합니다.

    Args:
        account: 계좌 번호 (형식: XXXX-XXXX-XXXX)
    """
    import re
    if not re.match(r"^\d{4}-\d{4}-\d{4}$", account):
        return f"오류: 계좌 번호 형식이 잘못되었습니다."

    # 데모용 더미 잔액
    balances = {
        "1234-5678-9012": 500_000,
        "9999-8888-7777": 1_200_000,
    }
    balance = balances.get(account, 0)
    return f"계좌 {account} 잔액: {balance:,}원"


tools = [transfer_money, get_account_balance]
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: MessagesState) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# handle_tool_errors=True (기본값): 예외를 ToolMessage로 변환
tool_node = ToolNode(tools=tools, handle_tool_errors=True)

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")
agent = graph.compile()

# 잘못된 계좌 번호로 테스트
result = agent.invoke({
    "messages": [("user", "계좌 1234567890에서 9999-8888-7777로 50000원 이체해줘")]
})
print("=== 오류 처리 테스트 ===")
print(result["messages"][-1].content)

# 정상 이체
result2 = agent.invoke({
    "messages": [("user", "1234-5678-9012에서 9999-8888-7777로 30000원 이체해줘")]
})
print("\n=== 정상 이체 테스트 ===")
print(result2["messages"][-1].content)
```

### 예제 3: InjectedState로 상태 기반 도구 만들기

```python
import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from pydantic import SecretStr

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


class ShoppingState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    cart: list[dict]      # 장바구니 아이템 목록
    user_name: str        # 로그인한 사용자 이름


# InjectedState: LLM에 노출되지 않고 런타임에 그래프 상태가 자동 주입됨
@tool
def add_to_cart(
    product_name: str,
    quantity: int,
    # 아래 파라미터는 LLM의 tool schema에 나타나지 않음
    state: Annotated[ShoppingState, InjectedState],
) -> str:
    """장바구니에 상품을 추가합니다.

    Args:
        product_name: 추가할 상품 이름
        quantity: 추가할 수량 (1 이상)
    """
    if quantity <= 0:
        return "오류: 수량은 1 이상이어야 합니다."

    # 상태에서 사용자 이름 접근 (LLM이 제공하지 않아도 됨)
    user = state.get("user_name", "익명")
    return f"{user}님의 장바구니에 '{product_name}' {quantity}개 추가 완료."


@tool
def view_cart(
    state: Annotated[ShoppingState, InjectedState],
) -> str:
    """현재 장바구니 내용을 확인합니다."""
    cart = state.get("cart", [])
    user = state.get("user_name", "익명")
    if not cart:
        return f"{user}님의 장바구니가 비어있습니다."
    items = "\n".join(
        f"  - {item['name']}: {item['quantity']}개" for item in cart
    )
    return f"{user}님의 장바구니:\n{items}"


tools = [add_to_cart, view_cart]
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: ShoppingState) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


tool_node = ToolNode(tools=tools)

graph = StateGraph(ShoppingState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")
agent = graph.compile()

result = agent.invoke({
    "messages": [("user", "장바구니에 뭐가 있는지 보여줘")],
    "cart": [{"name": "노트북", "quantity": 1}, {"name": "마우스", "quantity": 2}],
    "user_name": "김철수",
})
print(result["messages"][-1].content)
```

### 예제 4: 사람 확인이 필요한 도구 (interrupt)

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from pydantic import SecretStr

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


@tool
def delete_file(file_path: str) -> str:
    """파일을 삭제합니다. 이 작업은 되돌릴 수 없습니다.

    Args:
        file_path: 삭제할 파일의 전체 경로
    """
    # 실제로는 삭제하지 않고 시뮬레이션
    return f"파일 '{file_path}' 삭제 완료."


@tool
def read_file(file_path: str) -> str:
    """파일 내용을 읽습니다.

    Args:
        file_path: 읽을 파일의 전체 경로
    """
    return f"파일 '{file_path}' 내용: [샘플 파일 내용입니다]"


tools = [delete_file, read_file]
memory = MemorySaver()

# interrupt_before=["tools"]: 도구 실행 전에 항상 사람 확인
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    interrupt_before=["tools"],  # 도구 노드 실행 전 중단
)

config = {"configurable": {"thread_id": "safe-session"}}

# 1단계: 에이전트가 계획을 수립하고 중단
print("=== 1단계: 에이전트 실행 (도구 호출 전 중단) ===")
result = agent.invoke(
    {"messages": [("user", "/tmp/important.log 파일을 삭제해줘")]},
    config=config,
)

# 중단 지점에서 상태 확인
state = agent.get_state(config)
print(f"중단 위치: {state.next}")  # ('tools',)
last_msg = result["messages"][-1]
if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
    print(f"실행 예정 도구: {last_msg.tool_calls[0]['name']}")
    print(f"인자: {last_msg.tool_calls[0]['args']}")

# 2단계: 사람이 승인하면 None 입력으로 재개
print("\n=== 2단계: 사람 승인 후 재개 ===")
# None을 입력으로 주면 현재 상태에서 이어서 실행
result2 = agent.invoke(None, config=config)
print(result2["messages"][-1].content)
```

---

## ✏️ 실습 과제

### 과제 1: 안전한 계산기 도구
`calculate(expression: str)`를 구현하되:
- `eval()` 대신 안전한 방법으로 사칙연산만 지원
- 허용 문자: 숫자, `+`, `-`, `*`, `/`, `(`, `)`, 공백
- 그 외 문자가 포함되면 에러 메시지 반환

### 과제 2: InjectedToolCallId 활용
`InjectedToolCallId`를 사용해서 도구가 자신의 tool_call_id를 알고 커스텀 ToolMessage를 직접 반환하는 도구를 만들어보세요.
힌트: `from langgraph.prebuilt import InjectedToolCallId`

---

## ⚠️ 흔한 함정

**1. 너무 추상적인 도구 이름**
```python
# 나쁜 예: LLM이 언제 사용해야 할지 알 수 없음
@tool
def process(input_data: str) -> str: ...

# 좋은 예: 명확한 이름과 목적
@tool
def extract_email_addresses(text: str) -> list[str]: ...
```

**2. 예외 vs 에러 메시지 반환**
```python
# 예외를 raise하면 ToolNode가 잡아서 ToolMessage로 변환 (LLM이 "오류 발생" 메시지를 받음)
# 에러 메시지를 return하면 LLM이 내용을 읽고 수정 가능

# 권장: 예상 가능한 오류는 return, 예상 불가능한 시스템 오류는 raise
@tool
def get_data(id: str) -> str:
    if not id.isdigit():
        return "오류: ID는 숫자여야 합니다."  # LLM이 읽고 수정
    if id not in database:
        return f"오류: ID '{id}'를 찾을 수 없습니다."  # LLM이 읽고 수정
    return database[id]  # 정상 반환
```

**3. InjectedState 파라미터가 tool schema에 노출됨**
`Annotated[SomeType, InjectedState]` 형식으로 정확히 작성해야 합니다:
```python
# 잘못된 예: 타입 힌트 누락
def my_tool(state: InjectedState): ...

# 올바른 예
def my_tool(state: Annotated[MyStateType, InjectedState]): ...
```

---

## ✅ 셀프 체크

- [ ] 좋은 도구 docstring의 구성 요소를 설명할 수 있다
- [ ] `Literal` 타입으로 허용 값을 제한하는 도구를 만들 수 있다
- [ ] 도구 내에서 인자 검증 후 에러 메시지를 반환할 수 있다
- [ ] `InjectedState`로 LLM에 보이지 않는 파라미터를 구현할 수 있다
- [ ] `interrupt_before=["tools"]`로 사람 확인 흐름을 구현할 수 있다
- [ ] `handle_tool_errors`의 동작 방식을 설명할 수 있다

---

## 🔗 참고 자료

- [도구 정의 가이드](https://python.langchain.com/docs/concepts/tools/)
- [ToolNode 에러 처리](https://langchain-ai.github.io/langgraph/how-tos/tool-calling-errors/)
- [InjectedState 가이드](https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/)
- [Human-in-the-loop 도구](https://langchain-ai.github.io/langgraph/how-tos/human-in-the-loop/)

> **API 변동 안내:** `InjectedState`, `InjectedToolCallId`는 LangGraph 버전에 따라 임포트 경로가 변경될 수 있습니다. 최신 API는 [공식 문서](https://langchain-ai.github.io/langgraph/reference/prebuilt/)를 확인하세요.

---

◀ 이전: [Phase 28: 직접 만드는 에이전트 그래프](./28-custom-agent-graph.md)
▶ 다음: [Phase 30: Plan-and-Execute](./30-plan-and-execute.md)
