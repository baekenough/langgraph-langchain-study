# Phase 30: Plan-and-Execute

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 100분 |
| 난이도 | ★★★★☆ |
| 선행 학습 | Phase 29 (도구 설계 패턴), Phase 10 (구조화된 출력) |

---

## 🎯 학습 목표

- Plan-and-Execute 패턴의 작동 방식과 ReAct와의 차이를 설명할 수 있습니다.
- Pydantic으로 구조화된 실행 계획을 정의하고 생성할 수 있습니다.
- planner → executor → replan 3단계 그래프를 구현할 수 있습니다.
- 실행 결과에 따라 계획을 동적으로 수정하는 replan 노드를 만들 수 있습니다.
- 언제 Plan-and-Execute가 ReAct보다 적합한지 판단할 수 있습니다.

---

## 📚 핵심 개념

### ReAct vs Plan-and-Execute

| 항목 | ReAct | Plan-and-Execute |
|------|-------|-----------------|
| 계획 수립 | 매 스텝 즉흥적으로 결정 | 실행 전 전체 계획 수립 |
| 적합한 작업 | 단순하고 즉각적인 작업 | 복잡하고 다단계 작업 |
| 컨텍스트 효율 | 낮음 (모든 과거 메시지 전달) | 높음 (계획+결과만 전달) |
| 유연성 | 높음 (즉각 방향 전환) | 낮음 (계획 변경 비용 있음) |
| 예측 가능성 | 낮음 | 높음 |

Plan-and-Execute가 빛나는 상황:
- 수행할 단계가 10개 이상인 복잡한 작업
- 사용자에게 "이런 순서로 진행하겠습니다"를 보여줘야 할 때
- 긴 실행 과정에서 LLM 컨텍스트 길이를 절약해야 할 때

### 그래프 구조

```
START
  │
  ▼
[planner] ──► 구조화된 계획(Plan) 생성
  │
  ▼
[executor] ──► 현재 단계 실행 (도구 사용)
  │
  ├── 모든 단계 완료? ──► END
  │
  └── 아직 단계 남음 ──► [replan] ──► 계획 재조정
                              │
                              └──────────► [executor]
```

---

## 💻 코드 예제

### 예제 1: 기본 Plan-and-Execute 구현

```python
import os
from typing import Annotated, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, SecretStr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


# --- 도구 정의 ---

@tool
def search_web(query: str) -> str:
    """웹에서 정보를 검색합니다. (데모용 더미 구현)"""
    responses = {
        "python": "Python은 1991년 귀도 반 로섬이 만든 고급 프로그래밍 언어입니다.",
        "langchain": "LangChain은 LLM 애플리케이션을 구축하는 Python/JavaScript 프레임워크입니다.",
        "langgraph": "LangGraph는 LangChain 위에서 동작하는 그래프 기반 에이전트 프레임워크입니다.",
    }
    for key, val in responses.items():
        if key.lower() in query.lower():
            return val
    return f"'{query}'에 대한 검색 결과: 관련 정보를 찾았습니다."


@tool
def summarize_text(text: str, max_sentences: int = 3) -> str:
    """텍스트를 요약합니다. (데모용 더미 구현)"""
    sentences = text.split(".")[:max_sentences]
    return ". ".join(s.strip() for s in sentences if s.strip()) + "."


@tool
def write_report(title: str, content: str) -> str:
    """보고서를 작성합니다."""
    return f"=== {title} ===\n{content}\n[보고서 작성 완료]"


tools = [search_web, summarize_text, write_report]


# --- 상태 및 스키마 정의 ---

class Plan(BaseModel):
    """실행할 단계들의 목록."""
    steps: list[str] = Field(
        description="순서대로 실행할 작업 단계 목록. 각 단계는 구체적이고 실행 가능해야 합니다."
    )


class PlanExecuteState(TypedDict):
    input: str                           # 원래 사용자 요청
    plan: list[str]                      # 실행할 단계 목록
    past_steps: list[tuple[str, str]]    # [(단계, 결과), ...]
    response: str                        # 최종 응답


# --- 노드 정의 ---

# Planner: 구조화된 계획 생성
planner_llm = llm.with_structured_output(Plan)

PLANNER_PROMPT = """당신은 복잡한 작업을 단계별 계획으로 분해하는 전문가입니다.
사용자의 요청을 받아, 다음 도구들을 활용하여 수행할 수 있는 구체적인 단계 목록을 만드세요.
사용 가능한 도구: {tools}

각 단계는 "도구명을 사용하여 [구체적 작업]" 형식으로 작성하세요."""


def planner_node(state: PlanExecuteState) -> dict:
    """사용자 요청을 받아 단계별 실행 계획을 생성합니다."""
    tool_names = [t.name for t in tools]
    prompt = PLANNER_PROMPT.format(tools=", ".join(tool_names))

    plan = planner_llm.invoke([
        ("system", prompt),
        ("human", state["input"]),
    ])

    print(f"[Planner] 생성된 계획:")
    for i, step in enumerate(plan.steps, 1):
        print(f"  {i}. {step}")

    return {"plan": plan.steps}


# Executor: 현재 단계 실행
executor_agent = create_react_agent(model=llm, tools=tools)

EXECUTOR_PROMPT = """현재 실행할 작업: {task}

이전에 완료된 단계들:
{past_steps}

위 컨텍스트를 참고하여 현재 작업을 실행하세요."""


def executor_node(state: PlanExecuteState) -> dict:
    """계획의 첫 번째 단계를 실행합니다."""
    current_step = state["plan"][0]
    past_steps_text = "\n".join(
        f"  - {step}: {result}"
        for step, result in state.get("past_steps", [])
    ) or "없음"

    prompt = EXECUTOR_PROMPT.format(
        task=current_step,
        past_steps=past_steps_text,
    )

    print(f"\n[Executor] 실행 중: {current_step}")
    result = executor_agent.invoke({"messages": [("user", prompt)]})
    result_text = result["messages"][-1].content

    print(f"  → 결과: {result_text[:80]}...")

    return {
        "past_steps": state.get("past_steps", []) + [(current_step, result_text)],
        "plan": state["plan"][1:],  # 실행한 단계 제거
    }


# Replan: 실행 결과를 바탕으로 계획 수정 또는 최종 응답 생성
class ReplanOutput(BaseModel):
    """재계획 결과: 새 계획 또는 최종 응답."""
    response: str = Field(
        default="",
        description="모든 단계가 완료되었을 때 사용자에게 전달할 최종 응답. 완료된 경우에만 작성.",
    )
    new_steps: list[str] = Field(
        default_factory=list,
        description="아직 실행해야 할 추가 단계 목록. 더 수행할 작업이 있으면 작성.",
    )


replan_llm = llm.with_structured_output(ReplanOutput)

REPLAN_PROMPT = """원래 목표: {input}

현재까지 완료된 단계:
{past_steps}

남은 계획:
{remaining_plan}

지금까지의 진행 상황을 평가하여:
1. 목표가 달성되었으면 'response'에 최종 답변을 작성하고 'new_steps'는 비워두세요.
2. 아직 작업이 남았으면 'response'는 비워두고 'new_steps'에 남은 단계를 작성하세요."""


def replan_node(state: PlanExecuteState) -> dict:
    """실행 결과를 평가하고 계획을 수정하거나 완료합니다."""
    past_steps_text = "\n".join(
        f"  {i+1}. {step}: {result}"
        for i, (step, result) in enumerate(state.get("past_steps", []))
    )
    remaining_plan_text = "\n".join(
        f"  - {step}" for step in state.get("plan", [])
    ) or "없음"

    output = replan_llm.invoke([
        ("system", REPLAN_PROMPT.format(
            input=state["input"],
            past_steps=past_steps_text,
            remaining_plan=remaining_plan_text,
        ))
    ])

    if output.response:
        print(f"\n[Replan] 목표 달성 — 최종 응답 생성")
        return {"response": output.response, "plan": []}
    else:
        print(f"\n[Replan] 수정된 계획: {len(output.new_steps)}개 단계")
        return {"plan": output.new_steps}


# --- 라우팅 ---

def should_end(state: PlanExecuteState) -> str:
    """최종 응답이 있으면 종료, 아니면 executor 계속."""
    if state.get("response"):
        return END
    if not state.get("plan"):
        # 계획이 비어있는데 응답도 없으면 replan
        return "replan"
    return "executor"


# --- 그래프 빌드 ---

graph = StateGraph(PlanExecuteState)

graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("replan", replan_node)

graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "replan")
graph.add_conditional_edges("replan", should_end)

agent = graph.compile()

# 실행
print("=== Plan-and-Execute 에이전트 실행 ===\n")
result = agent.invoke({
    "input": "Python, LangChain, LangGraph를 각각 검색하고 요약한 후 비교 보고서를 작성해줘",
    "plan": [],
    "past_steps": [],
    "response": "",
})

print("\n=== 최종 보고서 ===")
print(result["response"])
```

### 예제 2: 재귀 제한과 진행 상황 표시

```python
import os
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, SecretStr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


@tool
def analyze_data(data_description: str) -> str:
    """데이터를 분석합니다. (데모용 더미 구현)"""
    return f"분석 완료: {data_description}에서 주요 패턴 3가지 발견됨."


@tool
def generate_visualization(chart_type: str, data: str) -> str:
    """데이터 시각화를 생성합니다. (데모용 더미 구현)"""
    return f"{chart_type} 차트 생성 완료: {data}"


@tool
def write_conclusion(findings: str) -> str:
    """분석 결론을 작성합니다. (데모용 더미 구현)"""
    return f"결론: {findings}를 바탕으로 다음 분기 전략을 권장합니다."


tools = [analyze_data, generate_visualization, write_conclusion]


class PlanState(TypedDict):
    input: str
    plan: list[str]
    completed: list[str]
    response: str
    iteration: int


class SimplePlan(BaseModel):
    steps: list[str] = Field(description="실행 단계 목록 (최대 5단계)")


def make_plan(state: PlanState) -> dict:
    plan_llm = llm.with_structured_output(SimplePlan)
    tool_names = ", ".join(t.name for t in tools)
    result = plan_llm.invoke([
        ("system", f"사용 가능 도구: {tool_names}. 최대 5단계로 계획을 세워주세요."),
        ("human", state["input"]),
    ])
    return {"plan": result.steps, "iteration": 0}


def execute_step(state: PlanState) -> dict:
    """현재 단계를 간단히 실행 (도구 직접 호출 시뮬레이션)."""
    current = state["plan"][0]
    iteration = state.get("iteration", 0) + 1
    total = len(state["plan"]) + len(state.get("completed", []))
    done_count = len(state.get("completed", []))

    print(f"[{done_count + 1}/{total + 1}] 실행: {current[:50]}...")

    # 실제로는 executor_agent.invoke()를 사용
    result = f"단계 '{current[:30]}...' 완료"

    return {
        "plan": state["plan"][1:],
        "completed": state.get("completed", []) + [f"{current}: {result}"],
        "iteration": iteration,
    }


def finalize(state: PlanState) -> dict:
    """모든 단계 완료 후 최종 응답 생성."""
    summary = "\n".join(f"  ✓ {c}" for c in state["completed"])
    response = f"모든 작업이 완료되었습니다.\n\n완료된 단계:\n{summary}"
    return {"response": response}


def route(state: PlanState) -> str:
    """계획이 남아있으면 execute, 아니면 finalize."""
    if state.get("iteration", 0) > 10:  # 안전 장치
        return "finalize"
    return "execute" if state["plan"] else "finalize"


graph = StateGraph(PlanState)
graph.add_node("plan", make_plan)
graph.add_node("execute", execute_step)
graph.add_node("finalize", finalize)

graph.add_edge(START, "plan")
graph.add_edge("plan", "execute")
graph.add_conditional_edges("execute", route)
graph.add_edge("finalize", END)

agent = graph.compile()

result = agent.invoke({
    "input": "판매 데이터를 분석하고 시각화한 후 결론을 도출해줘",
    "plan": [],
    "completed": [],
    "response": "",
    "iteration": 0,
}, config={"recursion_limit": 25})

print("\n" + result["response"])
```

---

## ✏️ 실습 과제

### 과제 1: 여행 계획 에이전트
`search_flights`, `book_hotel`, `create_itinerary` 도구를 만들고, Plan-and-Execute 패턴으로 사용자의 여행 계획 요청을 처리하는 에이전트를 구현하세요.

### 과제 2: 동적 계획 수정
executor가 실행하다가 "정보 부족"이라는 결과를 받으면 replan에서 추가 검색 단계를 삽입하는 로직을 구현해보세요.

---

## ⚠️ 흔한 함정

**1. 계획이 너무 세분화됨**
LLM이 20개 이상의 단계를 만들면 실행 시간이 길어지고 비용이 증가합니다.
Pydantic 스키마에 제약을 추가하세요:
```python
class Plan(BaseModel):
    steps: list[str] = Field(
        description="최대 7단계. 각 단계는 하나의 도구 호출로 완료 가능해야 합니다.",
        max_length=7,
    )
```

**2. replan이 무한 루프**
replan이 매번 새 단계를 추가하면 루프가 끝나지 않습니다.
반드시 `recursion_limit`을 설정하세요:
```python
agent.invoke(state, config={"recursion_limit": 20})
```

**3. 이전 단계 결과를 executor에게 전달 안 함**
executor가 이전 단계 결과를 모르면 중복 작업을 수행합니다.
`past_steps`를 executor 프롬프트에 항상 포함시키세요.

---

## ✅ 셀프 체크

- [ ] Plan-and-Execute와 ReAct의 차이점을 3가지 설명할 수 있다
- [ ] Pydantic으로 구조화된 Plan 스키마를 정의할 수 있다
- [ ] planner → executor → replan 3단계 그래프를 구현할 수 있다
- [ ] `with_structured_output(Plan)`으로 구조화된 계획을 생성할 수 있다
- [ ] `recursion_limit`으로 무한 루프를 방지할 수 있다
- [ ] 언제 Plan-and-Execute가 ReAct보다 적합한지 설명할 수 있다

---

## 🔗 참고 자료

- [Plan-and-Execute 가이드](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/)
- [구조화된 출력](https://python.langchain.com/docs/concepts/structured_outputs/)
- [Plan-and-Execute 논문 (Wang et al., 2023)](https://arxiv.org/abs/2305.04091)

> **API 변동 안내:** `with_structured_output`의 동작 방식은 LLM 제공자에 따라 다를 수 있습니다. 최신 사용법은 [공식 문서](https://python.langchain.com/docs/concepts/structured_outputs/)를 확인하세요.

---

◀ 이전: [Phase 29: 도구 설계 패턴](./29-tool-design-patterns.md)
▶ 다음: [Phase 31: Reflection / 자기교정](./31-reflection-self-correction.md)
