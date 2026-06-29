# Phase 33: 멀티에이전트 — 협업/Swarm 패턴

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 100분 |
| 난이도 | ★★★★★ |
| 선행 학습 | Phase 32 (Supervisor 패턴), Phase 29 (도구 설계 패턴) |

---

## 🎯 학습 목표

- 에이전트 네트워크(Swarm)에서 수평 핸드오프(peer-to-peer handoff)를 구현할 수 있습니다.
- "handoff 도구"로 에이전트가 스스로 제어권을 다른 에이전트에게 넘기는 방식을 이해합니다.
- Supervisor 패턴과 Swarm 패턴의 차이와 각각의 적합한 사용 사례를 설명할 수 있습니다.
- `langgraph-swarm` 패키지를 소개하고 직접 구현과의 차이를 이해합니다.
- 복잡한 협업 흐름을 설계할 수 있습니다.

---

## 📚 핵심 개념

### Supervisor vs Swarm (네트워크) 비교

| 항목 | Supervisor 패턴 | Swarm/Network 패턴 |
|------|---------------|------------------|
| 제어 구조 | 중앙 집중 (Supervisor가 모든 라우팅 결정) | 분산 (에이전트가 스스로 핸드오프 결정) |
| 확장성 | Worker 추가 시 Supervisor 프롬프트 수정 필요 | 새 에이전트를 독립적으로 추가 가능 |
| 투명성 | 높음 (Supervisor의 결정이 중앙화) | 낮음 (각 에이전트가 독립적으로 결정) |
| 적합한 작업 | 명확한 단계와 순서가 있는 작업 | 유연하고 동적인 협업이 필요한 작업 |
| 복잡도 | 낮음 (라우팅 로직 한 곳에 집중) | 높음 (각 에이전트의 핸드오프 조건 관리) |

### Handoff 도구 패턴

Swarm에서 에이전트는 **handoff 도구**를 통해 다른 에이전트로 제어권을 넘깁니다:

```python
@tool
def transfer_to_billing_agent(query: str) -> str:
    """청구 관련 문의를 처리하는 에이전트로 전환합니다."""
    # 이 도구가 호출되면 → 그래프 라우팅이 billing_agent로 이동
    ...
```

**핵심 아이디어:** 도구 호출 = 라우팅 신호

```
[Customer Service Agent]
  │ "결제 문제입니다" 인식
  │ → transfer_to_billing_agent() 호출
  │
  ▼
[Billing Agent]  ← 이제 이 에이전트가 대화 제어권 보유
```

---

## 💻 코드 예제

### 예제 1: 고객 서비스 Swarm 구현

```python
import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import ToolNode
from pydantic import SecretStr

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


# ─── Handoff 도구 정의 ───────────────────────────────────────────────────────
# 각 에이전트가 다른 에이전트로 넘길 수 있는 도구들

@tool
def transfer_to_billing(issue: str) -> str:
    """청구 및 결제 관련 문의를 처리하는 전문 에이전트로 전환합니다.
    요금 오류, 환불, 청구서 관련 질문에 사용하세요.

    Args:
        issue: 처리할 청구 관련 문제 설명
    """
    return f"[시스템] billing_agent로 전환합니다. 문제: {issue}"


@tool
def transfer_to_technical(issue: str) -> str:
    """기술 지원 에이전트로 전환합니다.
    서비스 장애, 오류 코드, 기술적 문제에 사용하세요.

    Args:
        issue: 처리할 기술적 문제 설명
    """
    return f"[시스템] technical_agent로 전환합니다. 문제: {issue}"


@tool
def transfer_to_general(reason: str) -> str:
    """일반 고객 서비스 에이전트로 전환합니다.
    일반적인 문의, 제품 정보, 기타 요청에 사용하세요.

    Args:
        reason: 일반 에이전트로 전환하는 이유
    """
    return f"[시스템] general_agent로 전환합니다. 이유: {reason}"


# ─── 실제 업무 도구 정의 ─────────────────────────────────────────────────────

@tool
def process_refund(order_id: str, amount: float) -> str:
    """환불을 처리합니다. (데모용 더미 구현)"""
    return f"주문 {order_id}에 대해 {amount:,.0f}원 환불이 처리되었습니다. 3-5 영업일 내 입금 예정."


@tool
def check_service_status(service: str) -> str:
    """서비스 상태를 확인합니다. (데모용 더미 구현)"""
    return f"'{service}' 서비스 상태: 정상 운영 중 (가동률 99.9%)"


@tool
def get_product_info(product: str) -> str:
    """제품 정보를 제공합니다. (데모용 더미 구현)"""
    return f"'{product}' 제품 정보: 월 구독료 9,900원, 무제한 사용, 24시간 지원 포함."


# ─── 에이전트 노드 생성 헬퍼 ─────────────────────────────────────────────────

def create_agent_node(
    system_prompt: str,
    tools: list,
    agent_name: str,
):
    """Swarm 에이전트 노드를 생성합니다."""
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState) -> Command:
        print(f"\n[{agent_name}] 처리 중...")
        response = llm_with_tools.invoke([
            ("system", system_prompt),
            *state["messages"],
        ])

        # 핸드오프 도구 호출 확인
        if response.tool_calls:
            tool_name = response.tool_calls[0]["name"]
            tool_args = response.tool_calls[0]["args"]
            tool_id = response.tool_calls[0]["id"]

            print(f"  → 도구 호출: {tool_name}")

            # 핸드오프 도구인지 확인
            handoff_map = {
                "transfer_to_billing": "billing_agent",
                "transfer_to_technical": "technical_agent",
                "transfer_to_general": "general_agent",
            }

            if tool_name in handoff_map:
                # 핸드오프: 도구 결과 메시지 추가 후 다른 에이전트로 이동
                next_agent = handoff_map[tool_name]
                print(f"  → {next_agent}로 핸드오프")

                # ToolMessage로 핸드오프 기록 (대화 히스토리 유지)
                tool_result = ToolMessage(
                    content=f"[핸드오프] {next_agent}로 전환됨",
                    tool_call_id=tool_id,
                )
                return Command(
                    goto=next_agent,
                    update={"messages": [response, tool_result]},
                )
            else:
                # 일반 도구 호출: ToolNode로 처리
                return Command(
                    goto="tools",
                    update={"messages": [response]},
                )

        # 도구 호출 없음: 최종 응답
        print(f"  → 응답 생성 완료")
        return Command(
            goto=END,
            update={"messages": [response]},
        )

    return agent_node


def make_tool_node(all_tools: list):
    """모든 실무 도구를 처리하는 ToolNode를 생성합니다."""
    from langgraph.prebuilt import ToolNode

    tool_node = ToolNode(tools=all_tools)

    def node(state: MessagesState) -> Command:
        result = tool_node.invoke(state)
        # 도구 실행 후 마지막 에이전트로 돌아가는 대신 일반 에이전트로 복귀
        # (단순화를 위해 general_agent로 복귀)
        return Command(
            goto="general_agent",
            update=result,
        )

    return node


# ─── 각 에이전트 정의 ─────────────────────────────────────────────────────────

GENERAL_TOOLS = [get_product_info, transfer_to_billing, transfer_to_technical]
BILLING_TOOLS = [process_refund, transfer_to_general, transfer_to_technical]
TECHNICAL_TOOLS = [check_service_status, transfer_to_general, transfer_to_billing]

general_node = create_agent_node(
    system_prompt=(
        "당신은 일반 고객 서비스 담당자입니다. 제품 정보를 안내하고, "
        "청구 문제는 billing 에이전트로, 기술 문제는 technical 에이전트로 전환하세요. "
        "한국어로 친절하게 응답하세요."
    ),
    tools=GENERAL_TOOLS,
    agent_name="general_agent",
)

billing_node = create_agent_node(
    system_prompt=(
        "당신은 청구 및 결제 전문 담당자입니다. 환불 처리와 요금 문의를 처리하세요. "
        "기술 문제는 technical 에이전트로, 일반 문의는 general 에이전트로 전환하세요. "
        "한국어로 전문적으로 응답하세요."
    ),
    tools=BILLING_TOOLS,
    agent_name="billing_agent",
)

technical_node = create_agent_node(
    system_prompt=(
        "당신은 기술 지원 전문가입니다. 서비스 장애와 기술적 문제를 해결하세요. "
        "청구 문제는 billing 에이전트로, 일반 문의는 general 에이전트로 전환하세요. "
        "한국어로 기술적으로 정확하게 응답하세요."
    ),
    tools=TECHNICAL_TOOLS,
    agent_name="technical_agent",
)

# ─── 그래프 빌드 ──────────────────────────────────────────────────────────────

all_work_tools = [get_product_info, process_refund, check_service_status]

graph = StateGraph(MessagesState)
graph.add_node("general_agent", general_node)
graph.add_node("billing_agent", billing_node)
graph.add_node("technical_agent", technical_node)
graph.add_node("tools", make_tool_node(all_work_tools))

# 진입점: 일반 에이전트부터 시작
graph.add_edge(START, "general_agent")
# Command(goto=...)가 나머지 라우팅 처리

swarm = graph.compile()

# 테스트 1: 일반 문의
print("=== 테스트 1: 일반 문의 ===")
r1 = swarm.invoke({
    "messages": [("user", "프리미엄 플랜 가격이 얼마인가요?")]
})
print(r1["messages"][-1].content)

# 테스트 2: 청구 문의 (핸드오프 발생)
print("\n=== 테스트 2: 청구 문의 (핸드오프) ===")
r2 = swarm.invoke({
    "messages": [("user", "지난달 요금이 잘못 청구된 것 같아요. 주문번호 ORD-2024-001, 5000원 환불 받고 싶어요.")]
})
print(r2["messages"][-1].content)
```

### 예제 2: 언제 Supervisor vs Swarm을 쓸까?

```python
"""
의사결정 가이드:

Supervisor 패턴을 선택하세요 (Phase 32):
  ✓ 작업 순서가 명확하게 정해진 경우 (A→B→C)
  ✓ 중앙에서 품질을 관리해야 하는 경우
  ✓ 각 Worker의 실행 여부를 Supervisor가 결정해야 하는 경우
  예: 리서치 → 작성 → 검토 파이프라인

Swarm/Network 패턴을 선택하세요:
  ✓ 사용자 요청에 따라 동적으로 전문가가 바뀌는 경우
  ✓ 에이전트가 스스로 자신의 한계를 인식하고 전환해야 하는 경우
  ✓ 비선형적이고 유연한 흐름이 필요한 경우
  예: 고객 서비스, 다국어 지원, 전문 분야별 Q&A
"""

# 간단한 비교 데모
print("Supervisor 패턴:")
print("  사용자 요청 → Supervisor가 순서 결정 → Worker A → Worker B → Worker C → 완료")
print()
print("Swarm 패턴:")
print("  사용자 요청 → Agent A → (필요 시) Agent B로 핸드오프 → (필요 시) Agent C로 핸드오프 → 완료")
```

### 예제 3: langgraph-swarm 패키지 소개

```python
# langgraph-swarm은 Swarm 패턴을 더 간결하게 구현할 수 있는 패키지입니다.
# 설치: pip install langgraph-swarm
#
# 아래는 참고용 코드입니다. 실행하려면 패키지를 별도 설치해야 합니다.

"""
from langgraph_swarm import create_swarm, create_handoff_tool
from langgraph.prebuilt import create_react_agent

# Handoff 도구를 자동 생성 (create_handoff_tool이 처리)
transfer_to_billing = create_handoff_tool(
    agent_name="billing_agent",
    description="청구 및 결제 문제를 처리하는 에이전트로 전환합니다.",
)

transfer_to_technical = create_handoff_tool(
    agent_name="technical_agent",
    description="기술 지원 에이전트로 전환합니다.",
)

# 에이전트 생성 — 핸드오프 도구 포함
general_agent = create_react_agent(
    model=llm,
    tools=[get_product_info, transfer_to_billing, transfer_to_technical],
    name="general_agent",
    prompt="일반 고객 서비스 담당자입니다.",
)

billing_agent = create_react_agent(
    model=llm,
    tools=[process_refund, transfer_to_general],
    name="billing_agent",
    prompt="청구 전문 담당자입니다.",
)

# Swarm 생성
swarm = create_swarm(
    agents=[general_agent, billing_agent],
    default_active_agent="general_agent",  # 시작 에이전트
).compile()

result = swarm.invoke({
    "messages": [("user", "환불 받고 싶어요")]
})
"""

# langgraph-swarm의 장단점:
# 장점: create_handoff_tool로 핸드오프 도구 자동 생성, 보일러플레이트 감소
# 단점: 내부 동작 불투명, 커스텀 핸드오프 로직 삽입 어려움
# 권장: 표준 고객서비스 패턴에는 langgraph-swarm, 복잡한 비즈니스 로직에는 직접 구현

print("langgraph-swarm 소개 완료")
print("설치: pip install langgraph-swarm")
print("문서: https://github.com/langchain-ai/langgraph-swarm-py")
```

---

## ✏️ 실습 과제

### 과제 1: 다국어 지원 에이전트 네트워크
다음 에이전트로 구성된 Swarm을 구현하세요:
- `korean_agent`: 한국어 요청 처리
- `english_agent`: 영어 요청 처리
- `translation_agent`: 번역이 필요한 경우 처리

각 에이전트가 자신의 언어가 아닌 요청을 받으면 적절한 에이전트로 핸드오프해야 합니다.

### 과제 2: Supervisor vs Swarm 벤치마크
동일한 작업(예: "리서치 후 보고서 작성")을 Supervisor 패턴(Phase 32)과 Swarm 패턴으로 각각 구현하고, 총 LLM 호출 횟수와 응답 품질을 비교해보세요.

---

## ⚠️ 흔한 함정

**1. 핸드오프 루프**
에이전트 A가 B로, B가 다시 A로 핸드오프하면 무한 루프가 됩니다.
반드시 `recursion_limit`을 설정하고, 핸드오프 조건을 명확히 정의하세요:
```python
swarm.invoke(state, config={"recursion_limit": 10})
```

**2. 핸드오프 도구와 일반 도구 구분 혼동**
핸드오프 도구가 호출되면 즉시 다른 에이전트로 이동해야 합니다.
일반 도구 실행 흐름(ToolNode)과 핸드오프 흐름을 명확히 분리하세요.

**3. 대화 히스토리 손실**
핸드오프 시 전체 대화 히스토리를 공유 상태에 유지해야 합니다.
새 에이전트가 이전 맥락 없이 시작하면 중복 질문이 발생합니다.

**4. 에이전트 과도한 세분화**
에이전트가 너무 많으면 각 에이전트의 역할이 모호해집니다.
처음에는 2~3개 에이전트로 시작하고 필요에 따라 확장하세요.

---

## ✅ 셀프 체크

- [ ] Supervisor 패턴과 Swarm 패턴의 차이를 3가지 설명할 수 있다
- [ ] "handoff 도구"의 개념과 구현 방식을 이해한다
- [ ] 에이전트가 도구 호출로 다른 에이전트로 제어권을 넘기는 코드를 작성할 수 있다
- [ ] 핸드오프 후 대화 히스토리가 유지되도록 구현할 수 있다
- [ ] 언제 Supervisor vs Swarm 패턴을 선택하면 좋은지 판단할 수 있다
- [ ] `langgraph-swarm`의 `create_handoff_tool`의 역할을 설명할 수 있다

---

## 🔗 참고 자료

- [에이전트 네트워크 개념](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [Swarm 튜토리얼](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/swarm/)
- [langgraph-swarm](https://github.com/langchain-ai/langgraph-swarm-py)
- [핸드오프 패턴](https://langchain-ai.github.io/langgraph/how-tos/agent-handoffs/)

> **API 변동 안내:** `Command` 기반 핸드오프 패턴은 LangGraph의 핵심 기능이지만, 고수준 API(`langgraph-swarm` 등)는 빠르게 발전 중입니다. 최신 API는 [공식 문서](https://langchain-ai.github.io/langgraph/)를 확인하세요.

---

◀ 이전: [Phase 32: 멀티에이전트 — Supervisor](./32-multiagent-supervisor.md)
▶ 다음: [Phase 34: Agentic RAG](./34-agentic-rag.md)
