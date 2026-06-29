# Phase 21: 순환과 반복

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 100분 |
| 난이도 | ★★★☆☆ |
| 선행 학습 | Phase 20 (노드·엣지·라우팅) |

---

## 🎯 학습 목표

- 루프(순환)를 포함한 그래프를 만들 수 있습니다.
- `recursion_limit` 설정으로 무한 루프를 방지합니다.
- 올바른 종료 조건(exit condition)을 설계할 수 있습니다.
- "에이전트 루프"(model ↔ tools)의 기본 형태를 이해합니다.
- 재시도·개선·자기 수정 패턴을 구현할 수 있습니다.

---

## 📚 핵심 개념

### 왜 순환이 필요한가?

선형 파이프라인(chain)은 한 번만 실행됩니다. 하지만 많은 AI 워크플로우는 **반복적 개선**이 필요합니다.

| 패턴 | 설명 |
|------|------|
| 재시도(Retry) | 실패 시 같은 작업 재실행 |
| 개선 반복(Refinement) | 결과물 품질이 목표에 도달할 때까지 반복 |
| 에이전트 루프 | "생각 → 도구 사용 → 결과 확인"을 반복 |
| 검색-생성 루프 | 부족한 정보 더 검색, 충분하면 답변 생성 |
| 다단계 검증 | 통과할 때까지 생성-검증 반복 |

### 루프 그래프의 구조

```
               ┌────────────────────────────┐
               │          (루프)             │
               ▼                            │
START ─► process_node ─► check_node ────────┘
                                  │
                                  └─(완료)─► END
```

핵심: `check_node`의 조건부 엣지가 `process_node`로 **되돌아가거나** `END`로 **나가는** 두 가지 경로를 제공합니다.

### recursion_limit: 무한 루프 방지

LangGraph는 기본적으로 `recursion_limit=25`를 적용합니다. 이 값에 도달하면 `GraphRecursionError`가 발생합니다.

```python
# 기본값 (25 스텝)
result = graph.invoke(initial_state)

# 명시적으로 변경
result = graph.invoke(
    initial_state,
    config={"recursion_limit": 50}  # 50 스텝까지 허용
)
```

**스텝(step)**: 노드 하나가 실행될 때마다 1씩 증가합니다. 루프가 5번 돌고 각 루프에 3개 노드가 있으면 15 스텝입니다.

### 종료 조건 설계 원칙

좋은 종료 조건을 설계하는 세 가지 방법:

**1. 상태 기반 종료**: 특정 상태 값이 목표에 도달하면 종료
```python
def check_quality(state) -> str:
    if state["quality_score"] >= 0.9:
        return END
    return "improve"
```

**2. 카운터 기반 종료**: 최대 반복 횟수 제한
```python
def check_attempts(state) -> str:
    if state["attempts"] >= 3 or state["success"]:
        return END
    return "retry"
```

**3. 복합 조건 종료**: 여러 조건 조합
```python
def smart_check(state) -> str:
    reached_goal = state["score"] >= TARGET_SCORE
    exhausted = state["iterations"] >= MAX_ITERATIONS
    if reached_goal or exhausted:
        return END
    return "continue"
```

### 에이전트 루프 미리보기

LangGraph에서 가장 흔한 루프 패턴은 **에이전트 루프**입니다. Part 4에서 자세히 다루지만, 기본 구조는 다음과 같습니다.

```
                    ┌─────────────────────┐
                    │      (도구 호출)      │
                    ▼                     │
START ─► model_node ─► tool_condition ───┘
                              │
                              └─(완료)─► END
```

- `model_node`: LLM이 메시지를 처리하고 도구 호출 여부 결정
- `tool_condition`: 도구 호출이 필요하면 `tool_node`로, 아니면 `END`로
- `tool_node`: 도구 실행 후 다시 `model_node`로

---

## 💻 코드 예제

### 예제 1: 기본 재시도 루프

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
import random


class RetryState(TypedDict):
    task: str
    attempts: int
    max_attempts: int
    success: bool
    result: str


def attempt_node(state: RetryState) -> dict:
    """작업 시도. 랜덤하게 성공/실패 시뮬레이션."""
    attempts = state["attempts"] + 1
    # 실제로는 외부 API 호출, LLM 생성 등
    success = random.random() > 0.5  # 50% 성공률
    print(f"시도 {attempts}/{state['max_attempts']}: {'성공' if success else '실패'}")

    if success:
        return {
            "attempts": attempts,
            "success": True,
            "result": f"'{state['task']}' 완료 (시도 {attempts}회)",
        }
    return {"attempts": attempts, "success": False}


def check_node(state: RetryState) -> str:
    """성공 여부와 재시도 횟수로 다음 경로 결정."""
    if state["success"]:
        return "done_node"
    if state["attempts"] >= state["max_attempts"]:
        return "fail_node"
    return "attempt_node"  # 재시도


def done_node(state: RetryState) -> dict:
    print(f"성공: {state['result']}")
    return {}


def fail_node(state: RetryState) -> dict:
    print(f"실패: {state['max_attempts']}회 시도 후 포기")
    return {"result": "작업 실패"}


builder = StateGraph(RetryState)
builder.add_node("attempt_node", attempt_node)
builder.add_node("check_node", check_node)
builder.add_node("done_node", done_node)
builder.add_node("fail_node", fail_node)

builder.add_edge(START, "attempt_node")
builder.add_edge("attempt_node", "check_node")
builder.add_conditional_edges(
    "check_node",
    lambda s: s,  # check_node가 이미 문자열 반환
    {
        "attempt_node": "attempt_node",
        "done_node": "done_node",
        "fail_node": "fail_node",
    }
)

# 실제로는 check_node가 str을 직접 반환하므로:
# add_conditional_edges의 두 번째 인자가 라우팅 함수여야 함
# 올바른 방법:
builder2 = StateGraph(RetryState)
builder2.add_node("attempt_node", attempt_node)
builder2.add_node("done_node", done_node)
builder2.add_node("fail_node", fail_node)

builder2.add_edge(START, "attempt_node")
# check 로직을 라우팅 함수로 직접 사용
builder2.add_conditional_edges(
    "attempt_node",
    check_node,  # 노드 함수가 아닌 라우팅 함수로도 사용 가능
    {
        "attempt_node": "attempt_node",
        "done_node": "done_node",
        "fail_node": "fail_node",
    }
)
builder2.add_edge("done_node", END)
builder2.add_edge("fail_node", END)

graph = builder2.compile()
result = graph.invoke(
    {"task": "데이터 처리", "attempts": 0, "max_attempts": 5, "success": False, "result": ""},
    config={"recursion_limit": 20},
)
print(f"최종 결과: {result['result']}")
```

**그래프 구조**:
```
START
  │
  ▼
attempt_node ◄─────────────────┐
  │                             │
  └─► check (라우팅 함수)        │
          ├─(재시도)─────────────┘
          ├─(성공)──► done_node ─► END
          └─(포기)──► fail_node ─► END
```

### 예제 2: 품질 개선 루프 (LLM 글쓰기)

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI


class WritingState(TypedDict):
    topic: str
    draft: str
    feedback: str
    iteration: int
    max_iterations: int
    quality_passed: bool


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
evaluator = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def write_node(state: WritingState) -> dict:
    """초안 또는 개선된 글 생성."""
    if state["iteration"] == 0:
        prompt = f"다음 주제로 짧은 문단을 작성해주세요: {state['topic']}"
    else:
        prompt = (
            f"다음 글을 피드백에 따라 개선해주세요.\n\n"
            f"원본:\n{state['draft']}\n\n"
            f"피드백:\n{state['feedback']}\n\n"
            f"개선된 버전만 출력하세요."
        )

    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"\n[글쓰기 반복 {state['iteration'] + 1}]")
    print(f"초안: {response.content[:100]}...")
    return {
        "draft": response.content,
        "iteration": state["iteration"] + 1,
    }


def evaluate_node(state: WritingState) -> dict:
    """글의 품질을 평가하고 피드백 생성."""
    response = evaluator.invoke([
        SystemMessage(content=(
            "글의 품질을 평가하세요. "
            "명확하고 간결하며 주제에 충실하면 'PASS', "
            "그렇지 않으면 'FAIL'과 구체적인 개선점을 제시하세요. "
            "형식: PASS 또는 FAIL: [이유]"
        )),
        HumanMessage(content=f"주제: {state['topic']}\n\n글:\n{state['draft']}"),
    ])

    feedback = response.content
    quality_passed = feedback.startswith("PASS")
    print(f"평가: {feedback[:80]}")
    return {"feedback": feedback, "quality_passed": quality_passed}


def quality_router(state: WritingState) -> str:
    """품질 통과 여부와 반복 횟수로 경로 결정."""
    if state["quality_passed"]:
        return "done"
    if state["iteration"] >= state["max_iterations"]:
        print(f"최대 반복 횟수({state['max_iterations']}) 도달, 종료")
        return "done"
    return "improve"


builder = StateGraph(WritingState)
builder.add_node("write", write_node)
builder.add_node("evaluate", evaluate_node)

builder.add_edge(START, "write")
builder.add_edge("write", "evaluate")
builder.add_conditional_edges(
    "evaluate",
    quality_router,
    {"improve": "write", "done": END},
)

graph = builder.compile()

result = graph.invoke({
    "topic": "인공지능이 일상생활을 변화시키는 방법",
    "draft": "",
    "feedback": "",
    "iteration": 0,
    "max_iterations": 3,
    "quality_passed": False,
}, config={"recursion_limit": 20})

print(f"\n최종 글 ({result['iteration']}회 반복):")
print(result["draft"])
```

**그래프 구조**:
```
START
  │
  ▼
write ◄────────────────┐
  │                     │(개선 필요)
  ▼                     │
evaluate ───────────────┘
  │
  └─(통과 또는 한도 초과)─► END
```

### 예제 3: 에이전트 루프 기본 형태 (도구 사용)

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


# 간단한 계산기 도구
@tool
def calculator(expression: str) -> str:
    """수학 표현식을 계산합니다. 예: '2 + 3 * 4'"""
    try:
        result = eval(expression)  # 프로덕션에서는 안전한 파서 사용
        return str(result)
    except Exception as e:
        return f"오류: {e}"


tools = [calculator]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# 도구 이름 → 함수 매핑
tool_map = {t.name: t for t in tools}


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def model_node(state: AgentState) -> dict:
    """LLM에 메시지를 전달하고 응답(도구 호출 포함 가능) 반환."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": response}


def tool_node(state: AgentState) -> dict:
    """마지막 AI 메시지의 도구 호출을 실행."""
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_func = tool_map[tool_call["name"]]
        result = tool_func.invoke(tool_call["args"])
        tool_results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )
        )
    return {"messages": tool_results}


def tool_condition(state: AgentState) -> str:
    """마지막 메시지에 도구 호출이 있으면 tool_node로, 없으면 END."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return END


builder = StateGraph(AgentState)
builder.add_node("model", model_node)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "model")
builder.add_conditional_edges("model", tool_condition)
builder.add_edge("tool_node", "model")  # 도구 실행 후 다시 모델로

graph = builder.compile()

result = graph.invoke({
    "messages": [HumanMessage(content="(15 * 8) + (100 / 4)를 계산해줘")]
})

for msg in result["messages"]:
    print(f"[{msg.type}] {msg.content}")
```

**그래프 구조**:
```
           ┌─────────────────────────┐
           │        (루프)            │
           ▼                        │
START ─► model ─►(도구 필요?)─► tool_node
                        │
                        └─(완료)─► END
```

---

## ✏️ 실습 과제

### 과제 1: 숫자 맞추기 게임

- 상태: `{"secret": int, "guess": int, "attempts": int, "hint": str}`
- `guess_node`: 랜덤 숫자 추측
- `check_node`: 너무 높음/낮음/정답 힌트 반환
- 라우팅: 정답이면 END, 10회 초과면 실패 노드, 아니면 재시도

### 과제 2: 자기 수정 코드 생성기

LLM을 사용해 코드를 생성하고, `exec()`으로 실행해서 오류가 없을 때까지 반복하는 그래프를 구현하세요. (최대 3회)

### 과제 3 (도전): 재귀 요약기

긴 텍스트를 청크로 나눠 요약하고, 요약이 여전히 길면 다시 요약하는 계층적 요약 루프를 구현하세요.

---

## ⚠️ 흔한 함정

### 함정 1: 종료 조건 없는 루프

```python
# ❌ 잘못된 예: 항상 자기 자신으로 돌아옴
def router(state):
    return "my_node"  # END로 나가는 경로 없음!

# GraphRecursionError 발생 (recursion_limit 초과)

# ✓ 올바른 예: 반드시 END로 나가는 경로 포함
def router(state):
    if state["done"]:
        return END
    return "my_node"
```

### 함정 2: recursion_limit을 너무 크게 설정

```python
# ⚠️ 주의: 무한 루프가 있어도 오류 없이 오래 실행됨
result = graph.invoke(state, config={"recursion_limit": 10000})

# ✓ 권장: 적절한 값 설정 후, 상태 기반 종료 조건도 별도로 구현
result = graph.invoke(state, config={"recursion_limit": 50})
```

### 함정 3: 루프 상태에서 카운터를 업데이트하지 않음

```python
# ❌ 잘못된 예: attempts를 증가시키지 않아 무한 루프
def process_node(state: RetryState) -> dict:
    # attempts를 반환하지 않음 → 값이 변하지 않음
    return {"success": False}  # attempts가 영원히 0

def router(state):
    if state["attempts"] >= 3:  # 절대 참이 되지 않음
        return END
    return "process_node"

# ✓ 올바른 예
def process_node(state: RetryState) -> dict:
    return {"attempts": state["attempts"] + 1, "success": False}
```

### 함정 4: 에이전트 루프에서 tool_calls 확인 누락

```python
# ❌ 잘못된 예: tool_calls가 없을 수도 있는데 무조건 접근
def bad_tool_node(state):
    last_msg = state["messages"][-1]
    for call in last_msg.tool_calls:  # AttributeError 가능!
        ...

# ✓ 올바른 예: 속성 확인 후 접근
def good_tool_node(state):
    last_msg = state["messages"][-1]
    if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
        return {}
    for call in last_msg.tool_calls:
        ...
```

---

## ✅ 셀프 체크

- [ ] 루프를 포함한 그래프를 만들고 순환 경로가 올바르게 작동하는지 확인할 수 있다.
- [ ] `recursion_limit`의 기본값(25)을 알고, `config`로 조정할 수 있다.
- [ ] 종료 조건이 없는 루프가 `GraphRecursionError`를 발생시킴을 이해한다.
- [ ] 카운터/플래그 기반의 올바른 종료 조건을 설계할 수 있다.
- [ ] 에이전트 루프(model ↔ tools)의 기본 구조와 `tool_condition` 라우터의 역할을 설명할 수 있다.
- [ ] 루프 내에서 상태 카운터를 업데이트하지 않으면 무한 루프가 생기는 이유를 안다.

---

## 🔗 참고 자료

- [LangGraph - Cycles and Branching](https://langchain-ai.github.io/langgraph/concepts/low_level/#cycles)
- [LangGraph - Agent Architectures](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/)
- [LangGraph - ReAct Agent Tutorial](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [LangGraph - recursion_limit](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.graph.CompiledGraph.invoke)

> **참고**: LangGraph API는 빠르게 발전하고 있습니다. 코드 실행 전 반드시 최신 공식 문서를 확인하세요.

---

## 네비게이션

| 이전 | 다음 |
|------|------|
| [Phase 20: 노드·엣지·라우팅](./20-nodes-edges-routing.md) | [Phase 22: 영속성과 체크포인터](./22-persistence-checkpointers.md) |
