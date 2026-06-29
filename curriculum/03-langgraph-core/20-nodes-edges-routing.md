# Phase 20: 노드·엣지·라우팅

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 120분 |
| 난이도 | ★★★☆☆ |
| 선행 학습 | Phase 19 (상태와 리듀서) |

---

## 🎯 학습 목표

- `add_node()`로 노드를 등록하는 다양한 방법을 알고 있습니다.
- 일반 엣지와 조건부 엣지의 차이를 설명하고 구현할 수 있습니다.
- 라우팅 함수를 작성해 조건에 따라 실행 경로를 제어합니다.
- `Command`를 사용해 상태 업데이트와 다음 노드 이동을 동시에 처리합니다.
- 팬아웃(fan-out)/팬인(fan-in) 패턴으로 병렬 분기를 구현합니다.

---

## 📚 핵심 개념

### 노드 등록: add_node()

```python
builder = StateGraph(MyState)

# 방법 1: 함수명을 노드 이름으로 사용
def my_function(state: MyState) -> dict:
    return {}

builder.add_node(my_function)  # 노드 이름 = "my_function"

# 방법 2: 별도 이름 지정 (권장)
builder.add_node("step_one", my_function)  # 노드 이름 = "step_one"

# 방법 3: 람다 (간단한 변환에 유용)
builder.add_node("uppercase", lambda s: {"text": s["text"].upper()})
```

### 일반 엣지 vs 조건부 엣지

**일반 엣지(Normal Edge)**: 항상 같은 노드로 이동합니다.

```python
builder.add_edge("node_a", "node_b")  # node_a → node_b 항상
builder.add_edge("node_b", END)       # node_b 후 종료
```

**조건부 엣지(Conditional Edge)**: 라우팅 함수의 반환값에 따라 이동 대상이 달라집니다.

```python
def router(state: MyState) -> str:
    """현재 상태를 보고 다음 노드 이름을 반환합니다."""
    if state["score"] >= 80:
        return "pass_node"
    else:
        return "retry_node"

builder.add_conditional_edges(
    "evaluate_node",  # 이 노드 실행 후 라우팅 적용
    router,           # 라우팅 함수
    # 선택적: 명시적 매핑 (문서화에 유용)
    {
        "pass_node": "pass_node",
        "retry_node": "retry_node",
    }
)
```

### 라우팅 함수 반환 타입

라우팅 함수는 다음 중 하나를 반환할 수 있습니다.

| 반환 타입 | 의미 |
|----------|------|
| `str` | 단일 노드 이름 |
| `list[str]` | 여러 노드 이름 (병렬 실행, 팬아웃) |
| `Send` | 특정 노드에 커스텀 상태 전달 |
| `END` | 그래프 종료 |

### Command: 상태 업데이트 + 이동을 동시에

노드가 일반 딕셔너리 대신 `Command`를 반환하면, 상태 업데이트와 다음 노드 지정을 한 번에 처리할 수 있습니다.

```python
from langgraph.types import Command

def my_node(state: MyState) -> Command:
    return Command(
        update={"field": "new_value"},  # 상태 업데이트
        goto="next_node",               # 다음 노드 이름
    )
```

`Command`의 장점:
- 조건부 엣지 없이도 노드 내부에서 다음 경로를 결정할 수 있음
- `update`와 `goto`를 한 객체로 명확하게 표현
- `goto=END`로 그래프 종료 가능

### 팬아웃(Fan-out)과 팬인(Fan-in)

**팬아웃**: 한 노드에서 여러 노드로 동시에 분기 → **병렬 실행**

```
                    ┌─ branch_a ─┐
start ─► router ───┤             ├─► merge
                    └─ branch_b ─┘
```

**팬인**: 여러 노드에서 하나의 노드로 합류

```python
# 팬아웃: 라우팅 함수가 리스트 반환
def fan_out_router(state: MyState) -> list[str]:
    return ["branch_a", "branch_b"]  # 두 노드 동시 실행

builder.add_conditional_edges("router", fan_out_router)

# 팬인: 각 브랜치 노드에서 merge_node로 연결
builder.add_edge("branch_a", "merge")
builder.add_edge("branch_b", "merge")
```

---

## 💻 코드 예제

### 예제 1: 조건부 엣지로 감성 분류

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class SentimentState(TypedDict):
    text: str
    sentiment: str
    response: str


def analyze_node(state: SentimentState) -> dict:
    """텍스트 감성을 분석합니다 (간단한 규칙 기반)."""
    text = state["text"].lower()
    if any(word in text for word in ["좋아", "훌륭", "최고", "행복"]):
        sentiment = "positive"
    elif any(word in text for word in ["싫어", "나쁜", "최악", "슬퍼"]):
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return {"sentiment": sentiment}


# 라우팅 함수: 감성에 따라 분기
def sentiment_router(state: SentimentState) -> Literal["positive_node", "negative_node", "neutral_node"]:
    return f"{state['sentiment']}_node"


def positive_node(state: SentimentState) -> dict:
    return {"response": "긍정적인 피드백 감사합니다!"}


def negative_node(state: SentimentState) -> dict:
    return {"response": "불편을 드려 죄송합니다. 개선하겠습니다."}


def neutral_node(state: SentimentState) -> dict:
    return {"response": "소중한 의견 감사합니다."}


# 그래프 구성
builder = StateGraph(SentimentState)
builder.add_node("analyze", analyze_node)
builder.add_node("positive_node", positive_node)
builder.add_node("negative_node", negative_node)
builder.add_node("neutral_node", neutral_node)

builder.add_edge(START, "analyze")
builder.add_conditional_edges(
    "analyze",
    sentiment_router,
    {
        "positive_node": "positive_node",
        "negative_node": "negative_node",
        "neutral_node": "neutral_node",
    }
)
# 모든 감성 노드에서 END로
builder.add_edge("positive_node", END)
builder.add_edge("negative_node", END)
builder.add_edge("neutral_node", END)

graph = builder.compile()

# 테스트
for text in ["서비스가 훌륭해요!", "최악이에요", "보통이에요"]:
    result = graph.invoke({"text": text, "sentiment": "", "response": ""})
    print(f"입력: {text!r}")
    print(f"감성: {result['sentiment']}, 응답: {result['response']}\n")
```

**그래프 구조**:
```
START
  │
  ▼
analyze
  │
  ├─(positive)─► positive_node ─► END
  ├─(negative)─► negative_node ─► END
  └─(neutral)──► neutral_node  ─► END
```

### 예제 2: Command로 루프 제어

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


class RetryState(TypedDict):
    attempt: int
    success: bool
    result: str


def try_node(state: RetryState) -> Command:
    """작업을 시도하고 성공/실패에 따라 다음 경로를 결정합니다."""
    import random
    attempt = state["attempt"] + 1
    success = random.random() > 0.6  # 40% 성공 확률 시뮬레이션

    print(f"시도 {attempt}: {'성공' if success else '실패'}")

    if success:
        return Command(
            update={"attempt": attempt, "success": True, "result": f"시도 {attempt}에 성공"},
            goto="done_node",
        )
    elif attempt >= 3:
        # 최대 재시도 초과
        return Command(
            update={"attempt": attempt, "success": False, "result": "최대 재시도 초과"},
            goto="fail_node",
        )
    else:
        # 재시도
        return Command(
            update={"attempt": attempt, "success": False},
            goto="try_node",  # 자기 자신으로 이동 (루프)
        )


def done_node(state: RetryState) -> dict:
    print(f"완료: {state['result']}")
    return {}


def fail_node(state: RetryState) -> dict:
    print(f"실패: {state['result']}")
    return {}


builder = StateGraph(RetryState)
builder.add_node("try_node", try_node)
builder.add_node("done_node", done_node)
builder.add_node("fail_node", fail_node)

builder.add_edge(START, "try_node")
builder.add_edge("done_node", END)
builder.add_edge("fail_node", END)

graph = builder.compile()

result = graph.invoke({"attempt": 0, "success": False, "result": ""})
```

**그래프 구조**:
```
START
  │
  ▼
try_node ◄──────┐
  │              │(재시도)
  ├─(성공)──────┘
  │
  ├─(성공)────► done_node ─► END
  └─(실패/초과)─► fail_node ─► END
```

### 예제 3: 팬아웃/팬인 병렬 처리

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END


def concat_reducer(a: list[str], b: list[str]) -> list[str]:
    return a + b


class ParallelState(TypedDict):
    query: str
    results: Annotated[list[str], concat_reducer]


def search_web(state: ParallelState) -> dict:
    """웹 검색 시뮬레이션."""
    print(f"[web] '{state['query']}' 검색 중...")
    return {"results": [f"웹 결과: {state['query']}에 대한 답변"]}


def search_db(state: ParallelState) -> dict:
    """데이터베이스 검색 시뮬레이션."""
    print(f"[db] '{state['query']}' 조회 중...")
    return {"results": [f"DB 결과: {state['query']} 관련 레코드"]}


def search_docs(state: ParallelState) -> dict:
    """문서 검색 시뮬레이션."""
    print(f"[docs] '{state['query']}' 문서 검색 중...")
    return {"results": [f"문서 결과: {state['query']} 관련 섹션"]}


def merge_node(state: ParallelState) -> dict:
    """모든 검색 결과를 합칩니다."""
    print(f"\n통합된 결과 ({len(state['results'])}개):")
    for r in state["results"]:
        print(f"  - {r}")
    return {}


# 라우팅 함수: 세 브랜치로 팬아웃
def fan_out(state: ParallelState) -> list[str]:
    return ["search_web", "search_db", "search_docs"]


builder = StateGraph(ParallelState)
builder.add_node("search_web", search_web)
builder.add_node("search_db", search_db)
builder.add_node("search_docs", search_docs)
builder.add_node("merge", merge_node)

# START → 팬아웃 라우터
builder.add_conditional_edges(START, fan_out)

# 팬인: 모든 브랜치 → merge
builder.add_edge("search_web", "merge")
builder.add_edge("search_db", "merge")
builder.add_edge("search_docs", "merge")
builder.add_edge("merge", END)

graph = builder.compile()

result = graph.invoke({"query": "LangGraph", "results": []})
print(f"\n총 {len(result['results'])}개 결과 수집됨")
```

**그래프 구조**:
```
         ┌─► search_web  ─┐
         │                │
START ───┼─► search_db   ─┼─► merge ─► END
         │                │
         └─► search_docs ─┘
```

### 예제 4: 다중 조건 라우팅 + Literal 타입 힌트

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class OrderState(TypedDict):
    order_id: str
    amount: float
    status: str
    fraud_score: float
    message: str


def validate_node(state: OrderState) -> dict:
    """주문 유효성 검사."""
    if state["amount"] <= 0:
        return {"status": "invalid"}
    return {"status": "valid", "fraud_score": 0.1 if state["amount"] < 1000 else 0.8}


# Literal 타입 힌트로 가능한 경로를 명시적으로 선언
def order_router(state: OrderState) -> Literal["fraud_check", "high_risk", "approve", "reject"]:
    if state["status"] == "invalid":
        return "reject"
    if state["fraud_score"] > 0.7:
        return "high_risk"
    if state["amount"] > 500:
        return "fraud_check"
    return "approve"


def fraud_check_node(state: OrderState) -> dict:
    print(f"[fraud] 주문 {state['order_id']} 사기 검사 중...")
    return {"message": "사기 검사 통과"}


def high_risk_node(state: OrderState) -> dict:
    return {"message": "고위험 주문 - 수동 검토 필요", "status": "pending"}


def approve_node(state: OrderState) -> dict:
    return {"message": "주문 승인됨", "status": "approved"}


def reject_node(state: OrderState) -> dict:
    return {"message": "주문 거부됨", "status": "rejected"}


builder = StateGraph(OrderState)
for name, func in [
    ("validate", validate_node),
    ("fraud_check", fraud_check_node),
    ("high_risk", high_risk_node),
    ("approve", approve_node),
    ("reject", reject_node),
]:
    builder.add_node(name, func)

builder.add_edge(START, "validate")
builder.add_conditional_edges("validate", order_router)

# fraud_check 후 → approve
builder.add_edge("fraud_check", "approve")

for terminal in ["high_risk", "approve", "reject"]:
    builder.add_edge(terminal, END)

graph = builder.compile()

# 테스트
test_orders = [
    {"order_id": "O001", "amount": 50.0, "status": "", "fraud_score": 0.0, "message": ""},
    {"order_id": "O002", "amount": 800.0, "status": "", "fraud_score": 0.0, "message": ""},
    {"order_id": "O003", "amount": -10.0, "status": "", "fraud_score": 0.0, "message": ""},
]

for order in test_orders:
    result = graph.invoke(order)
    print(f"{order['order_id']}: {result['message']} (상태: {result['status']})")
```

---

## ✏️ 실습 과제

### 과제 1: 학점 계산기

- 상태: `{"score": int, "grade": str, "feedback": str}`
- `calculate_node`: 점수를 받아 등급(A/B/C/D/F) 결정
- `grade_router`: 등급에 따라 각 노드로 라우팅
- 각 등급 노드: 해당 피드백 메시지 설정
- `Literal` 타입 힌트로 가능한 경로 명시

### 과제 2: Command 기반 워크플로우

- `Command`를 사용해 데이터 처리 파이프라인 구현
- 노드: `fetch → validate → (valid → transform → store) / (invalid → error_log)`
- `validate` 노드에서 `Command`로 상태 업데이트와 경로 결정을 동시에 처리

### 과제 3 (도전): 동적 팬아웃

입력 상태의 `tasks: list[str]` 필드에 담긴 작업 수에 따라 동적으로 팬아웃하는 그래프를 구현하세요. `Send`를 사용해 각 작업 노드에 독립적인 상태를 전달해 보세요.

---

## ⚠️ 흔한 함정

### 함정 1: 조건부 엣지와 일반 엣지를 같은 소스 노드에 혼용

```python
# ❌ 잘못된 예: 같은 노드에서 일반 엣지와 조건부 엣지 동시 사용
builder.add_edge("node_a", "node_b")           # 항상 node_b로
builder.add_conditional_edges("node_a", router) # 동시에 조건부 분기?
# → 예상치 못한 동작 발생

# ✓ 올바른 예: 하나만 사용
builder.add_conditional_edges("node_a", router)
```

### 함정 2: 라우팅 함수가 존재하지 않는 노드 이름 반환

```python
def bad_router(state):
    return "nonexistent_node"  # 등록되지 않은 노드 이름

# → 런타임 오류: "Node 'nonexistent_node' not found"
# ✓ 등록된 노드 이름만 반환하도록 주의
```

### 함정 3: 팬인 시 리듀서 없이 결과 합치기

```python
# ❌ 잘못된 예: 팬아웃 후 리듀서 없는 필드에 각 브랜치가 동시에 쓰기
class BadState(TypedDict):
    results: list[str]  # 리듀서 없음! 병렬 쓰기 시 하나만 남음

# ✓ 올바른 예: 리듀서 사용
class GoodState(TypedDict):
    results: Annotated[list[str], lambda a, b: a + b]
```

### 함정 4: Command와 조건부 엣지 동시 설정

```python
# ❌ 잘못된 예: Command를 반환하는 노드에 add_conditional_edges도 설정
def my_node(state):
    return Command(goto="target")

builder.add_conditional_edges("my_node", router)  # 중복! Command가 우선

# ✓ 올바른 예: Command 사용 시 엣지 설정 불필요
# (Command의 goto가 라우팅을 담당)
```

---

## ✅ 셀프 체크

- [ ] `add_node()`의 세 가지 사용 방법(이름 자동/명시/람다)을 알고 있다.
- [ ] `add_edge()`와 `add_conditional_edges()`의 차이를 설명할 수 있다.
- [ ] 라우팅 함수의 반환 타입(str, list[str], END)을 구분해 사용할 수 있다.
- [ ] `Command(update=..., goto=...)`로 상태 업데이트와 이동을 동시에 처리할 수 있다.
- [ ] 팬아웃/팬인 패턴을 구현하고, 이때 리듀서가 필요한 이유를 설명할 수 있다.
- [ ] `Literal` 타입 힌트로 가능한 라우팅 경로를 명시하는 방법을 안다.

---

## 🔗 참고 자료

- [LangGraph - Nodes](https://langchain-ai.github.io/langgraph/concepts/low_level/#nodes)
- [LangGraph - Edges](https://langchain-ai.github.io/langgraph/concepts/low_level/#edges)
- [LangGraph - Command](https://langchain-ai.github.io/langgraph/concepts/low_level/#command)
- [LangGraph - Fan-out/Fan-in](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)

> **참고**: LangGraph API는 빠르게 발전하고 있습니다. 코드 실행 전 반드시 최신 공식 문서를 확인하세요.

---

## 네비게이션

| 이전 | 다음 |
|------|------|
| [Phase 19: 상태와 리듀서](./19-state-reducers.md) | [Phase 21: 순환과 반복](./21-cycles-iteration.md) |
