# Phase 32: 멀티에이전트 — Supervisor 패턴

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 110분 |
| 난이도 | ★★★★☆ |
| 선행 학습 | Phase 28 (직접 만드는 에이전트 그래프), Phase 25 (서브그래프) |

---

## 🎯 학습 목표

- Supervisor 패턴의 구조와 역할 분담을 설명할 수 있습니다.
- `Command(goto=...)` 핸드오프로 Supervisor가 Worker를 제어하는 방식을 구현할 수 있습니다.
- Worker를 서브그래프로 구성하는 방법을 이해합니다.
- 공유 상태(`MessagesState`)로 Worker 간 결과를 공유할 수 있습니다.
- `langgraph-supervisor` 패키지를 소개하고 언제 사용하면 좋은지 판단할 수 있습니다.

---

## 📚 핵심 개념

### Supervisor 패턴이란?

하나의 Supervisor 에이전트가 여러 Worker 에이전트를 조율하는 패턴입니다.

```
사용자 요청
     │
     ▼
[Supervisor] ← 무엇을 누구에게 맡길지 결정
     │
     ├──► [Worker A: 리서처] ──► 결과 반환
     │
     ├──► [Worker B: 작가]   ──► 결과 반환
     │
     └──► [Worker C: 검토자] ──► 결과 반환
     │
     ▼
최종 응답 (Supervisor가 종합)
```

**핵심 특징:**
- Supervisor는 라우터 역할만 함 (직접 도구를 실행하지 않음)
- Worker는 독립적으로 도구를 실행
- 모든 결과는 공유 상태(Messages)를 통해 Supervisor에게 전달

### Command(goto=...) 핸드오프

`Command`는 LangGraph에서 노드가 다음에 실행할 노드를 동적으로 결정하는 메커니즘입니다:

```python
from langgraph.types import Command

def supervisor_node(state):
    # ... LLM이 다음 worker를 결정 ...
    return Command(goto="researcher")  # researcher 노드로 이동
    # 또는
    return Command(goto=END)  # 완료
```

---

## 💻 코드 예제

### 예제 1: 기본 Supervisor-Worker 패턴

```python
import os
from typing import Annotated, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, SecretStr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


# --- Worker 도구 정의 ---

@tool
def search_academic_papers(topic: str) -> str:
    """학술 논문을 검색합니다. (데모용 더미 구현)"""
    return (
        f"'{topic}' 관련 논문 검색 결과:\n"
        f"  1. Smith et al. (2023) - {topic}의 최신 동향\n"
        f"  2. Johnson et al. (2022) - {topic} 방법론 비교\n"
        f"  3. Lee et al. (2024) - {topic} 실제 적용 사례"
    )


@tool
def analyze_trends(data: str) -> str:
    """데이터 트렌드를 분석합니다. (데모용 더미 구현)"""
    return f"트렌드 분석 결과:\n  - 최근 3년간 '{data}' 분야 연구 30% 증가\n  - 주요 키워드: AI, 자동화, 효율성"


@tool
def write_summary(content: str, style: Literal["technical", "plain"] = "plain") -> str:
    """내용을 요약 문서로 작성합니다. (데모용 더미 구현)"""
    prefix = "[기술 문서]" if style == "technical" else "[일반 문서]"
    return f"{prefix} 요약:\n{content[:200]}...\n\n결론: 위 내용을 바탕으로 전략적 접근이 필요합니다."


@tool
def check_factual_accuracy(text: str) -> str:
    """텍스트의 사실 정확성을 검토합니다. (데모용 더미 구현)"""
    return f"팩트체크 완료:\n  - 주요 사실: 확인됨\n  - 오류 발견: 없음\n  - 신뢰도: 높음\n검토 대상: {text[:100]}..."


# --- Worker 에이전트 생성 (create_react_agent로 간단하게) ---

researcher_agent = create_react_agent(
    model=llm,
    tools=[search_academic_papers, analyze_trends],
    state_modifier="당신은 리서치 전문가입니다. 주어진 주제를 철저히 조사하세요. 한국어로 답변하세요.",
)

writer_agent = create_react_agent(
    model=llm,
    tools=[write_summary],
    state_modifier="당신은 콘텐츠 작가입니다. 리서치 결과를 명확하고 읽기 좋은 문서로 작성하세요. 한국어로 답변하세요.",
)

reviewer_agent = create_react_agent(
    model=llm,
    tools=[check_factual_accuracy],
    state_modifier="당신은 편집 검토자입니다. 작성된 콘텐츠의 정확성과 품질을 검토하세요. 한국어로 답변하세요.",
)


# --- Supervisor 라우팅 스키마 ---

WORKERS = ["researcher", "writer", "reviewer"]

class SupervisorDecision(BaseModel):
    """Supervisor의 다음 행동 결정."""
    next: Literal["researcher", "writer", "reviewer", "FINISH"] = Field(
        description=(
            "다음에 작업할 Worker 또는 'FINISH'. "
            "'researcher': 정보 수집, "
            "'writer': 문서 작성, "
            "'reviewer': 검토, "
            "'FINISH': 모든 작업 완료"
        )
    )
    instruction: str = Field(
        description="선택된 Worker에게 전달할 구체적인 지시사항"
    )


supervisor_llm = llm.with_structured_output(SupervisorDecision)

SUPERVISOR_SYSTEM = """당신은 팀 Supervisor입니다. 사용자의 요청을 완성하기 위해
다음 Worker들을 적절히 활용하세요:

- researcher: 정보 수집 및 분석
- writer: 문서 작성
- reviewer: 내용 검토 및 팩트체크

지금까지의 대화를 보고 다음 단계를 결정하세요.
모든 작업이 완료되면 'FINISH'를 선택하세요."""


def supervisor_node(state: MessagesState) -> Command:
    """Supervisor: 다음 Worker를 결정하고 지시합니다."""
    decision = supervisor_llm.invoke([
        ("system", SUPERVISOR_SYSTEM),
        *state["messages"],
    ])

    print(f"\n[Supervisor] 결정: {decision.next}")
    print(f"  지시: {decision.instruction[:80]}")

    if decision.next == "FINISH":
        return Command(goto=END)

    # Worker에게 지시사항을 추가하여 전달
    return Command(
        goto=decision.next,
        update={
            "messages": [
                HumanMessage(
                    content=decision.instruction,
                    name="supervisor",
                )
            ]
        },
    )


# --- Worker 노드 함수 ---

def make_worker_node(agent, name: str):
    """Worker 노드를 생성하는 헬퍼 함수."""
    def worker_node(state: MessagesState) -> Command:
        print(f"[{name}] 작업 실행 중...")
        result = agent.invoke(state)
        last_msg = result["messages"][-1]

        # Worker 결과를 Messages에 추가하고 Supervisor로 돌아감
        return Command(
            goto="supervisor",
            update={
                "messages": [
                    HumanMessage(
                        content=last_msg.content,
                        name=name,
                    )
                ]
            },
        )
    return worker_node


# --- 그래프 빌드 ---

graph = StateGraph(MessagesState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", make_worker_node(researcher_agent, "researcher"))
graph.add_node("writer", make_worker_node(writer_agent, "writer"))
graph.add_node("reviewer", make_worker_node(reviewer_agent, "reviewer"))

graph.add_edge(START, "supervisor")
# Command(goto=...)가 라우팅을 처리하므로 추가 엣지 불필요

multi_agent = graph.compile()

# 실행
print("=== Supervisor 멀티에이전트 시스템 ===\n")
result = multi_agent.invoke({
    "messages": [("user", "LangGraph에 대한 기술 보고서를 작성해줘. 최신 트렌드 조사, 요약 작성, 검토까지 진행해줘.")]
})

print("\n=== 최종 결과 ===")
print(result["messages"][-1].content)
```

### 예제 2: Worker를 서브그래프로 구성

```python
import os
from typing import Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, SecretStr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.types import Command

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


# --- 코딩 팀 서브그래프 ---

class CodingTeamState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    code: str
    review_passed: bool


@tool
def write_python_code(spec: str) -> str:
    """Python 코드를 작성합니다. (데모용 더미 구현)"""
    return f"```python\ndef solution():\n    # {spec}을 구현\n    return 'result'\n```"


@tool
def review_code(code: str) -> str:
    """코드를 검토합니다. (데모용 더미 구현)"""
    return "코드 검토 완료: 로직 정확, 타입 힌트 누락 (경고), 전반적으로 양호."


def coder_node(state: CodingTeamState) -> dict:
    """코드를 작성하는 노드."""
    llm_with_tools = llm.bind_tools([write_python_code])
    response = llm_with_tools.invoke([
        ("system", "Python 코드를 작성하는 전문 개발자입니다."),
        *state["messages"],
    ])
    code_result = write_python_code.invoke({"spec": state["messages"][-1].content})
    return {
        "messages": [AIMessage(content=f"코드 작성 완료:\n{code_result}")],
        "code": code_result,
    }


def reviewer_node(state: CodingTeamState) -> dict:
    """코드를 검토하는 노드."""
    review = review_code.invoke({"code": state.get("code", "")})
    passed = "양호" in review or "좋음" in review
    return {
        "messages": [AIMessage(content=f"검토 결과: {review}")],
        "review_passed": passed,
    }


# 코딩 팀 서브그래프 컴파일
coding_team_graph = StateGraph(CodingTeamState)
coding_team_graph.add_node("coder", coder_node)
coding_team_graph.add_node("reviewer", reviewer_node)
coding_team_graph.add_edge(START, "coder")
coding_team_graph.add_edge("coder", "reviewer")
coding_team_graph.add_edge("reviewer", END)
coding_team = coding_team_graph.compile()


# --- 메인 Supervisor 그래프 ---

class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task_completed: bool


class RouterDecision(BaseModel):
    destination: str = Field(description="'coding_team' 또는 'FINISH'")
    task: str = Field(description="팀에게 전달할 구체적 작업")


router_llm = llm.with_structured_output(RouterDecision)


def supervisor_node(state: SupervisorState) -> Command:
    decision = router_llm.invoke([
        ("system", "당신은 소프트웨어 팀 Supervisor입니다. 코딩 작업은 coding_team에 위임하세요. 완료되면 FINISH."),
        *state["messages"],
    ])

    print(f"[Supervisor] → {decision.destination}: {decision.task[:60]}")

    if decision.destination == "FINISH":
        return Command(goto=END, update={"task_completed": True})

    # 서브그래프 호출 방식: 노드로 등록된 서브그래프 이름으로 goto
    return Command(
        goto="coding_team",
        update={
            "messages": [HumanMessage(content=decision.task, name="supervisor")]
        },
    )


def coding_team_node(state: SupervisorState) -> Command:
    """서브그래프를 노드로 감싸는 래퍼."""
    # 서브그래프에 현재 메시지 전달
    sub_result = coding_team.invoke({
        "messages": state["messages"],
        "code": "",
        "review_passed": False,
    })

    # 서브그래프 결과를 상위 메시지에 추가하고 Supervisor로 귀환
    last_msg = sub_result["messages"][-1]
    return Command(
        goto="supervisor",
        update={
            "messages": [
                HumanMessage(
                    content=f"[coding_team 완료] 검토 통과: {sub_result['review_passed']}\n{last_msg.content}",
                    name="coding_team",
                )
            ]
        },
    )


main_graph = StateGraph(SupervisorState)
main_graph.add_node("supervisor", supervisor_node)
main_graph.add_node("coding_team", coding_team_node)
main_graph.add_edge(START, "supervisor")

orchestrator = main_graph.compile()

print("=== Supervisor + 서브그래프 ===\n")
result = orchestrator.invoke({
    "messages": [("user", "사용자 인증 기능을 구현하는 Python 코드를 작성하고 검토해줘")],
    "task_completed": False,
})

print("\n=== 완료 ===")
print(f"작업 완료: {result['task_completed']}")
print(result["messages"][-1].content[:300])
```

### 예제 3: langgraph-supervisor 패키지 소개

```python
# langgraph-supervisor는 Supervisor 패턴을 더 간결하게 구현할 수 있는 패키지입니다.
# 설치: pip install langgraph-supervisor
#
# 아래는 참고용 코드입니다. 실행하려면 패키지를 별도 설치해야 합니다.

"""
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

# Worker 에이전트 생성 (이름이 중요)
researcher = create_react_agent(
    model=llm,
    tools=[search_tool],
    name="researcher",  # Supervisor가 이 이름으로 호출
    prompt="당신은 리서치 전문가입니다.",
)

writer = create_react_agent(
    model=llm,
    tools=[write_tool],
    name="writer",
    prompt="당신은 작가입니다.",
)

# Supervisor 생성 — create_supervisor가 라우팅 로직을 자동으로 처리
supervisor = create_supervisor(
    model=llm,
    agents=[researcher, writer],
    prompt="팀 Supervisor로서 리서치와 작성 작업을 조율하세요.",
).compile()

result = supervisor.invoke({
    "messages": [("user", "AI 트렌드 보고서 작성해줘")]
})
"""

# langgraph-supervisor의 장단점:
# 장점: 보일러플레이트 코드 감소, Worker 등록 간편
# 단점: 커스텀 로직 삽입 어려움, 내부 동작 불투명
# 권장: 표준 패턴에는 langgraph-supervisor, 커스텀 흐름이 필요하면 직접 구현

print("langgraph-supervisor 소개 완료")
print("설치: pip install langgraph-supervisor")
print("문서: https://github.com/langchain-ai/langgraph-supervisor-py")
```

---

## ✏️ 실습 과제

### 과제 1: 뉴스 분석 팀
다음 Worker로 구성된 Supervisor 시스템을 구현하세요:
- `news_fetcher`: 주제 관련 뉴스 수집
- `sentiment_analyzer`: 뉴스 감성 분석
- `report_writer`: 분석 보고서 작성

Supervisor는 1→2→3 순서로 실행하고, 각 Worker의 결과를 다음 Worker에게 전달해야 합니다.

### 과제 2: 동적 Worker 선택
사용자 요청의 언어에 따라 `korean_writer` 또는 `english_writer`를 선택하는 Supervisor를 구현해보세요.

---

## ⚠️ 흔한 함정

**1. Command의 update가 MessagesState와 호환되지 않음**
`MessagesState`는 `add_messages` 리듀서를 사용합니다. Command의 update에 messages를 포함할 때 리스트로 감싸야 합니다:
```python
# 올바른 예
Command(goto="worker", update={"messages": [HumanMessage(content="지시")]})
# 잘못된 예
Command(goto="worker", update={"messages": HumanMessage(content="지시")})
```

**2. Supervisor가 루프에 빠짐**
LLM이 FINISH를 선택하지 않으면 무한 루프가 됩니다.
항상 `recursion_limit`을 설정하세요:
```python
multi_agent.invoke(state, config={"recursion_limit": 15})
```

**3. Worker 결과를 Supervisor가 못 읽음**
Worker의 출력이 공유 Messages에 추가되지 않으면 Supervisor가 결과를 알 수 없습니다.
Command(goto="supervisor", update={"messages": [...]}) 패턴을 항상 사용하세요.

---

## ✅ 셀프 체크

- [ ] Supervisor 패턴의 역할 분담(Supervisor/Worker)을 설명할 수 있다
- [ ] `Command(goto=...)` 핸드오프의 동작 방식을 이해한다
- [ ] `with_structured_output`으로 Supervisor 라우팅 결정을 구현할 수 있다
- [ ] Worker 노드에서 Supervisor로 결과를 올바르게 반환할 수 있다
- [ ] 서브그래프를 Worker로 사용하는 방법을 안다
- [ ] `langgraph-supervisor` 패키지의 장단점을 설명할 수 있다

---

## 🔗 참고 자료

- [멀티에이전트 시스템 개요](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [Supervisor 튜토리얼](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)
- [Command API](https://langchain-ai.github.io/langgraph/reference/types/#langgraph.types.Command)
- [langgraph-supervisor](https://github.com/langchain-ai/langgraph-supervisor-py)

> **API 변동 안내:** `Command` 타입의 import 경로(`langgraph.types`)와 사용법은 버전마다 다를 수 있습니다. 최신 API는 [공식 문서](https://langchain-ai.github.io/langgraph/reference/types/)를 확인하세요.

---

◀ 이전: [Phase 31: Reflection / 자기교정](./31-reflection-self-correction.md)
▶ 다음: [Phase 33: 멀티에이전트 — 협업/Swarm](./33-multiagent-collaboration.md)
