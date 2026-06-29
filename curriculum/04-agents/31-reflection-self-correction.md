# Phase 31: Reflection / 자기교정

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 90분 |
| 난이도 | ★★★★☆ |
| 선행 학습 | Phase 30 (Plan-and-Execute), Phase 20 (노드/엣지/라우팅) |

---

## 🎯 학습 목표

- Reflection(자기반성) 패턴의 개념과 작동 원리를 설명할 수 있습니다.
- generate ↔ reflect 루프를 `StateGraph`로 구현할 수 있습니다.
- 종료 조건(품질 기준 충족, 최대 반복 횟수)을 설계할 수 있습니다.
- Reflexion 패턴(에피소드 메모리 활용)의 개념을 이해합니다.
- 자기교정이 유용한 작업 유형을 판단할 수 있습니다.

---

## 📚 핵심 개념

### Reflection 패턴이란?

Reflection은 LLM이 자신의 출력을 평가하고 반복적으로 개선하는 패턴입니다.
인간이 글을 쓰고 → 검토하고 → 수정하는 과정을 흉내냅니다.

```
START
  │
  ▼
[generate] ──── 초안 생성
  │
  ▼
[reflect] ──── 자기비평 (어떤 점이 부족한가?)
  │
  ├── 충분히 좋음 ──► END
  │
  └── 개선 필요 ──► [generate] (비평을 참고해서 재생성)
```

### 언제 사용하나?

| 작업 유형 | Reflection 적합성 |
|----------|-----------------|
| 코드 생성/디버깅 | 높음 — 테스트 결과로 자기교정 |
| 글쓰기/번역 | 높음 — 품질 평가 후 재작성 |
| 수학 문제 풀이 | 중간 — 단계별 검증 가능 |
| 단순 Q&A | 낮음 — 반복 비용 대비 효과 미미 |

### Reflexion이란?

Shinn et al. (2023)의 논문에서 제안된 패턴입니다.
기본 Reflection에서 한 걸음 더 나아가, **에피소드 메모리**에 반성 내용을 저장하고
다음 시도에서 참고합니다:

```
시도 1: 생성 → 실패 → 반성 기록
시도 2: [이전 반성 참고] → 생성 → 실패 → 반성 기록
시도 3: [이전 반성들 참고] → 생성 → 성공!
```

---

## 💻 코드 예제

### 예제 1: 기본 Reflection 루프 — 에세이 작성

```python
import os
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,  # 창의적 작업이므로 temperature 높임
)


class ReflectionState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    iteration: int


GENERATOR_SYSTEM = """당신은 전문 작가입니다. 사용자의 요청에 따라 고품질 에세이를 작성합니다.
비평을 받으면 그 내용을 반영하여 글을 개선합니다.
항상 한국어로 작성하세요."""

REFLECTOR_SYSTEM = """당신은 엄격한 편집자입니다. 다음 기준으로 에세이를 평가하세요:
1. 명확성: 논지가 명확한가?
2. 근거: 주장을 뒷받침하는 근거가 충분한가?
3. 구조: 서론-본론-결론 구조가 있는가?
4. 독창성: 새로운 관점이나 통찰이 있는가?

부족한 점을 구체적으로 지적하고, 개선 방향을 제시하세요.
만약 에세이가 충분히 좋다면 '[승인]'으로 시작하는 응답을 작성하세요."""


def generate_node(state: ReflectionState) -> dict:
    """에세이를 생성하거나 비평을 바탕으로 개선합니다."""
    iteration = state.get("iteration", 0)
    print(f"\n[Generate] 반복 #{iteration + 1}")

    response = llm.invoke([
        ("system", GENERATOR_SYSTEM),
        *state["messages"],
    ])

    print(f"  → 생성 완료 ({len(response.content)}자)")
    return {
        "messages": [response],
        "iteration": iteration + 1,
    }


def reflect_node(state: ReflectionState) -> dict:
    """마지막으로 생성된 텍스트를 비평합니다."""
    print(f"[Reflect] 비평 중...")

    # 비평자는 생성된 텍스트를 "사용자 메시지"로 받아 평가
    # 실제 대화 흐름: 생성자의 출력 → 비평자의 입력
    generated_text = state["messages"][-1].content

    critique = llm.invoke([
        ("system", REFLECTOR_SYSTEM),
        ("human", f"다음 에세이를 평가해주세요:\n\n{generated_text}"),
    ])

    is_approved = critique.content.startswith("[승인]")
    print(f"  → {'승인됨' if is_approved else '개선 필요'}")

    # 비평 내용을 HumanMessage로 추가 (생성자가 다음 라운드에서 읽음)
    return {"messages": [HumanMessage(content=critique.content)]}


def should_continue(state: ReflectionState) -> str:
    """종료 조건 확인."""
    iteration = state.get("iteration", 0)

    # 최대 3번 반복
    if iteration >= 3:
        print("[종료] 최대 반복 횟수 도달")
        return END

    # 마지막 비평이 승인이면 종료
    messages = state["messages"]
    # 비평 메시지 찾기 (HumanMessage 중 마지막 것)
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and msg.content.startswith("[승인]"):
            print("[종료] 품질 기준 충족")
            return END

    return "generate"


graph = StateGraph(ReflectionState)
graph.add_node("generate", generate_node)
graph.add_node("reflect", reflect_node)

graph.add_edge(START, "generate")
graph.add_edge("generate", "reflect")
graph.add_conditional_edges("reflect", should_continue)

agent = graph.compile()

# 실행
print("=== Reflection 에세이 작성 에이전트 ===")
result = agent.invoke({
    "messages": [HumanMessage(content="인공지능이 창의성을 가질 수 있는가에 대한 300자 에세이를 작성해주세요.")],
    "iteration": 0,
})

print("\n=== 최종 에세이 ===")
# 마지막 AIMessage (최종 생성본)
final_essay = next(
    msg for msg in reversed(result["messages"])
    if isinstance(msg, AIMessage)
)
print(final_essay.content)
print(f"\n총 반복 횟수: {result['iteration']}")
```

### 예제 2: 코드 생성 + 실행으로 자기교정

```python
import os
import subprocess
import tempfile
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


class CodeGenState(TypedDict):
    task: str                    # 구현할 기능 설명
    messages: Annotated[list[BaseMessage], add_messages]
    code: str                    # 생성된 코드
    test_result: str             # 테스트 실행 결과
    iteration: int
    success: bool


CODE_GEN_SYSTEM = """당신은 Python 전문가입니다. 사용자가 요청하는 기능을 구현하는 
완전하고 실행 가능한 Python 코드를 작성하세요.

규칙:
- 코드만 작성하세요 (설명 없이 ```python 블록만)
- 모든 import를 포함하세요
- 코드가 올바르게 동작하는지 확인하는 테스트를 마지막에 포함하세요 (print로 결과 출력)
- 이전 오류가 있으면 반드시 수정하세요"""


def extract_code(text: str) -> str:
    """응답에서 Python 코드 블록을 추출합니다."""
    import re
    pattern = r"```python\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0].strip() if matches else text.strip()


def run_code(code: str) -> tuple[bool, str]:
    """임시 파일에 코드를 저장하고 실행합니다."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ["python", temp_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "오류: 실행 시간 초과 (10초)"
    except Exception as e:
        return False, f"오류: {str(e)}"
    finally:
        import os as os_module
        os_module.unlink(temp_path)


def code_generator(state: CodeGenState) -> dict:
    """코드를 생성하거나 오류를 수정합니다."""
    iteration = state.get("iteration", 0)
    print(f"\n[CodeGen] 반복 #{iteration + 1}")

    messages = [SystemMessage(content=CODE_GEN_SYSTEM)]

    if iteration == 0:
        # 첫 번째: 요청으로부터 코드 생성
        messages.append(HumanMessage(content=f"다음 기능을 구현하세요: {state['task']}"))
    else:
        # 후속: 오류를 포함한 전체 대화 이력 전달
        messages.extend(state["messages"])

    response = llm.invoke(messages)
    code = extract_code(response.content)
    print(f"  → 코드 생성 완료 ({len(code)}자)")

    return {
        "messages": [
            HumanMessage(content=f"다음 기능을 구현하세요: {state['task']}") if iteration == 0
            else HumanMessage(content="코드를 수정해주세요."),
            response,
        ],
        "code": code,
        "iteration": iteration + 1,
    }


def code_executor(state: CodeGenState) -> dict:
    """코드를 실행하고 결과를 상태에 저장합니다."""
    print(f"[Executor] 코드 실행 중...")
    success, output = run_code(state["code"])

    print(f"  → {'성공' if success else '실패'}: {output[:60]}...")

    if not success:
        # 실패 시 오류 메시지를 다음 생성 요청에 포함
        error_feedback = HumanMessage(
            content=f"코드 실행 오류:\n```\n{output}\n```\n위 오류를 수정해주세요."
        )
        return {
            "messages": [error_feedback],
            "test_result": output,
            "success": False,
        }

    return {
        "test_result": output,
        "success": True,
    }


def route_after_execution(state: CodeGenState) -> str:
    """성공하거나 최대 반복에 도달하면 종료."""
    if state.get("success"):
        return END
    if state.get("iteration", 0) >= 4:
        print("[종료] 최대 반복 횟수 도달 (실패)")
        return END
    return "generate"


graph = StateGraph(CodeGenState)
graph.add_node("generate", code_generator)
graph.add_node("execute", code_executor)

graph.add_edge(START, "generate")
graph.add_edge("generate", "execute")
graph.add_conditional_edges("execute", route_after_execution)

agent = graph.compile()

# 실행
print("=== 코드 자기교정 에이전트 ===")
result = agent.invoke({
    "task": "피보나치 수열의 n번째 값을 반환하는 함수 fibonacci(n)을 구현하고, n=10의 결과를 출력하세요",
    "messages": [],
    "code": "",
    "test_result": "",
    "iteration": 0,
    "success": False,
})

print("\n=== 최종 결과 ===")
print(f"성공 여부: {result['success']}")
print(f"총 반복 횟수: {result['iteration']}")
if result["success"]:
    print(f"실행 결과:\n{result['test_result']}")
```

### 예제 3: 품질 점수 기반 Reflexion 개념 구현

```python
import os
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0.5,
)


class QualityEvaluation(BaseModel):
    """콘텐츠 품질 평가 결과."""
    score: int = Field(description="품질 점수 (0~100)", ge=0, le=100)
    strengths: list[str] = Field(description="잘된 점 목록")
    weaknesses: list[str] = Field(description="부족한 점 목록")
    improvement_tips: list[str] = Field(description="구체적인 개선 방향")


class ReflexionState(TypedDict):
    task: str
    content: str                  # 현재 생성된 콘텐츠
    evaluations: list[dict]       # 이전 평가 기록 (에피소드 메모리)
    iteration: int
    final_score: int


eval_llm = llm.with_structured_output(QualityEvaluation)


def generate_content(state: ReflexionState) -> dict:
    """이전 평가들을 참고하여 콘텐츠를 생성합니다 (Reflexion 핵심)."""
    iteration = state.get("iteration", 0)

    # 이전 평가 메모리를 프롬프트에 포함
    memory_text = ""
    if state.get("evaluations"):
        memory_text = "\n\n이전 시도에서의 피드백:\n"
        for i, eval_record in enumerate(state["evaluations"], 1):
            memory_text += f"\n[시도 {i}]\n"
            memory_text += f"  점수: {eval_record['score']}/100\n"
            memory_text += f"  부족한 점: {', '.join(eval_record['weaknesses'])}\n"
            memory_text += f"  개선 방향: {', '.join(eval_record['improvement_tips'])}\n"

    prompt = f"""다음 작업을 수행하세요: {state['task']}
{memory_text}
위의 피드백을 반드시 반영하여 더 좋은 결과를 만들어주세요.
한국어로 작성하세요."""

    print(f"\n[Generate] 시도 #{iteration + 1} (이전 피드백 {len(state.get('evaluations', []))}개 참고)")
    response = llm.invoke([("human", prompt)])

    return {
        "content": response.content,
        "iteration": iteration + 1,
    }


def evaluate_content(state: ReflexionState) -> dict:
    """생성된 콘텐츠를 평가하고 기억에 저장합니다."""
    print(f"[Evaluate] 품질 평가 중...")

    evaluation = eval_llm.invoke([
        ("system", "다음 콘텐츠의 품질을 엄격하게 평가하세요."),
        ("human", f"작업: {state['task']}\n\n콘텐츠:\n{state['content']}"),
    ])

    print(f"  → 점수: {evaluation.score}/100")
    print(f"  → 부족한 점: {', '.join(evaluation.weaknesses[:2])}")

    # 평가 기록을 에피소드 메모리에 추가 (Reflexion의 핵심)
    new_evaluation = {
        "score": evaluation.score,
        "strengths": evaluation.strengths,
        "weaknesses": evaluation.weaknesses,
        "improvement_tips": evaluation.improvement_tips,
        "content_preview": state["content"][:100],
    }

    return {
        "evaluations": state.get("evaluations", []) + [new_evaluation],
        "final_score": evaluation.score,
    }


def should_continue(state: ReflexionState) -> str:
    """품질 점수가 80 이상이거나 3번 시도했으면 종료."""
    if state.get("final_score", 0) >= 80:
        print(f"[종료] 품질 기준 충족 (점수: {state['final_score']}/100)")
        return END
    if state.get("iteration", 0) >= 3:
        print(f"[종료] 최대 시도 횟수 도달 (최종 점수: {state['final_score']}/100)")
        return END
    return "generate"


graph = StateGraph(ReflexionState)
graph.add_node("generate", generate_content)
graph.add_node("evaluate", evaluate_content)

graph.add_edge(START, "generate")
graph.add_edge("generate", "evaluate")
graph.add_conditional_edges("evaluate", should_continue)

agent = graph.compile()

result = agent.invoke({
    "task": "파이썬 제너레이터(generator)를 초보자에게 설명하는 150자 설명문 작성",
    "content": "",
    "evaluations": [],
    "iteration": 0,
    "final_score": 0,
})

print("\n=== 최종 콘텐츠 ===")
print(result["content"])
print(f"\n최종 점수: {result['final_score']}/100")
print(f"총 시도 횟수: {result['iteration']}")
```

---

## ✏️ 실습 과제

### 과제 1: 번역 품질 개선
한국어 → 영어 번역을 생성하고, "자연스러움", "정확성", "원문 뉘앙스 보존" 세 가지 기준으로 평가하는 Reflection 루프를 구현하세요. 각 기준 점수의 평균이 80 이상이면 종료하도록 설계하세요.

### 과제 2: SQL 쿼리 자기교정
SQL 쿼리를 생성하고, 실제 실행 없이 "문법 검사"를 수행하는 LLM 비평자를 만들어보세요. 비평자가 문법 오류를 발견하면 수정 프롬프트와 함께 재생성을 요청하는 루프를 구현하세요.

---

## ⚠️ 흔한 함정

**1. 비평이 너무 관대하거나 너무 엄격함**
비평자의 시스템 프롬프트에 구체적인 기준과 예시를 포함시키세요:
```python
# 나쁜 예: 모호한 기준
"좋은 글을 평가하세요"

# 좋은 예: 구체적인 체크리스트
"다음 기준으로 평가하세요:\n1. 논지 명확성 (0-30점)\n2. 근거 충분성 (0-30점)\n3. 문체 완성도 (0-40점)"
```

**2. 비평 메시지가 대화 히스토리에 쌓임**
반복이 많아지면 컨텍스트가 길어집니다. 최근 N개의 비평만 유지하는 메모리 관리 로직을 추가하세요.

**3. 무한 개선 시도**
품질이 향상되지 않아도 재시도를 계속합니다. 반드시 최대 반복 횟수로 탈출 조건을 추가하세요.

---

## ✅ 셀프 체크

- [ ] Reflection 패턴의 generate ↔ reflect 루프를 설명할 수 있다
- [ ] 종료 조건(품질 기준 + 최대 반복)을 올바르게 구현할 수 있다
- [ ] Reflexion과 기본 Reflection의 차이(에피소드 메모리)를 설명할 수 있다
- [ ] 코드 실행 결과를 자기교정 루프에 활용할 수 있다
- [ ] Pydantic으로 구조화된 품질 평가 스키마를 설계할 수 있다
- [ ] 비평 메시지를 다음 생성 프롬프트에 효과적으로 주입할 수 있다

---

## 🔗 참고 자료

- [Reflection 가이드](https://langchain-ai.github.io/langgraph/tutorials/reflection/reflection/)
- [Reflexion 논문 (Shinn et al., 2023)](https://arxiv.org/abs/2303.11366)
- [Self-Refine (Madaan et al., 2023)](https://arxiv.org/abs/2303.17651)

> **API 변동 안내:** LangGraph의 그래프 구성 API는 계속 발전하고 있습니다. 최신 패턴은 [공식 튜토리얼](https://langchain-ai.github.io/langgraph/tutorials/)을 확인하세요.

---

◀ 이전: [Phase 30: Plan-and-Execute](./30-plan-and-execute.md)
▶ 다음: [Phase 32: 멀티에이전트 — Supervisor](./32-multiagent-supervisor.md)
