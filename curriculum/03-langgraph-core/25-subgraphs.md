# Phase 25: 서브그래프

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 105분 |
| 난이도 | ★★★★☆ |
| 선행 학습 | Phase 24 (Human-in-the-Loop) |

---

## 🎯 학습 목표

- 컴파일된 그래프를 다른 그래프의 노드로 추가할 수 있습니다.
- 같은 상태 스키마와 다른 상태 스키마의 서브그래프 사용 패턴을 이해합니다.
- 서브그래프로 복잡한 워크플로우를 모듈화할 수 있습니다.
- `subgraphs=True`로 서브그래프 내부까지 스트리밍할 수 있습니다.

---

## 📚 핵심 개념

### 왜 서브그래프인가?

그래프가 복잡해지면 단일 거대 그래프를 관리하기가 어렵습니다. 서브그래프는 다음을 가능하게 합니다:

| 필요 | 서브그래프 없이 | 서브그래프 있이 |
|------|----------------|----------------|
| 재사용 | 로직 복붙 | 한 번 정의, 여러 곳에서 사용 |
| 테스트 | 전체 그래프를 돌려야 함 | 서브그래프 단독 테스트 가능 |
| 팀 협업 | 충돌 위험 | 각자 담당 서브그래프 독립 개발 |
| 가독성 | 노드 수십 개의 복잡한 그래프 | 계층적 구조 |

### 합성 패턴

```
부모 그래프 (ParentGraph)
┌────────────────────────────────────────┐
│  START → node_a → [서브그래프] → node_b → END
│                        │
│              ┌──────────────────┐
│              │  SubGraph        │
│              │  sub_a → sub_b  │
│              └──────────────────┘
└────────────────────────────────────────┘
```

### 두 가지 합성 패턴

**패턴 1: 상태 키가 겹치는 경우 (직접 합성)**

부모와 서브그래프가 동일한 키를 공유하면 컴파일된 서브그래프를 바로 노드로 추가할 수 있습니다.

```python
# 공유 키: messages
parent_builder.add_node("sub_process", subgraph)  # 직접 추가
```

**패턴 2: 상태 스키마가 다른 경우 (래퍼 함수)**

스키마가 다르면 래퍼 함수로 입출력을 변환합니다.

```python
def call_subgraph(parent_state: ParentState) -> dict:
    sub_input = {"query": parent_state["user_input"]}  # 변환
    sub_output = subgraph.invoke(sub_input)
    return {"result": sub_output["answer"]}            # 역변환

parent_builder.add_node("sub_process", call_subgraph)
```

### 네임스페이스와 스트리밍

서브그래프 내부 이벤트를 스트리밍할 때는 `subgraphs=True`를 사용합니다.

```python
for chunk in graph.stream(input, stream_mode="updates", subgraphs=True):
    # chunk는 (namespace_tuple, update_dict)
    namespace, update = chunk
    print(f"네임스페이스: {namespace}")  # ("", ) 또는 ("sub_process:...",)
    print(f"업데이트: {update}")
```

`namespace`는 현재 이벤트가 어느 그래프(부모/서브)에서 왔는지를 알려줍니다.

---

## 💻 코드 예제

### 예제 1: 가장 단순한 서브그래프 합성

```python
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# ─── 공유 상태 (부모와 서브그래프가 같은 키를 사용) ───
class SharedState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    language: str


# ─── 서브그래프: 번역 파이프라인 ───
def detect_language_node(state: SharedState) -> dict:
    """마지막 메시지의 언어를 감지."""
    last_msg = state["messages"][-1].content
    response = llm.invoke(
        f"다음 텍스트의 언어를 한 단어로 알려주세요: '{last_msg}'"
    )
    detected = response.content.strip()
    print(f"  [서브그래프] 감지된 언어: {detected}")
    return {"language": detected}


def translate_node(state: SharedState) -> dict:
    """감지된 언어에 따라 번역."""
    last_msg = state["messages"][-1].content
    if "한국어" in state["language"] or "Korean" in state["language"]:
        prompt = f"다음을 영어로 번역하세요: {last_msg}"
    else:
        prompt = f"다음을 한국어로 번역하세요: {last_msg}"
    response = llm.invoke(prompt)
    print(f"  [서브그래프] 번역 완료")
    return {"messages": response}


# 서브그래프 빌드
sub_builder = StateGraph(SharedState)
sub_builder.add_node("detect_language", detect_language_node)
sub_builder.add_node("translate", translate_node)
sub_builder.add_edge(START, "detect_language")
sub_builder.add_edge("detect_language", "translate")
sub_builder.add_edge("translate", END)
translation_subgraph = sub_builder.compile()

# ─── 부모 그래프 ───
def intro_node(state: SharedState) -> dict:
    """번역 전 메시지 준비."""
    print("[부모] 번역 파이프라인 시작")
    return {}


def outro_node(state: SharedState) -> dict:
    """번역 완료 후 처리."""
    print("[부모] 번역 파이프라인 완료")
    return {}


parent_builder = StateGraph(SharedState)
parent_builder.add_node("intro", intro_node)
parent_builder.add_node("translate_pipeline", translation_subgraph)  # 서브그래프 직접 추가
parent_builder.add_node("outro", outro_node)
parent_builder.add_edge(START, "intro")
parent_builder.add_edge("intro", "translate_pipeline")
parent_builder.add_edge("translate_pipeline", "outro")
parent_builder.add_edge("outro", END)
graph = parent_builder.compile()

# 실행
result = graph.invoke({
    "messages": [HumanMessage(content="Hello, how are you today?")],
    "language": "",
})
print(f"\n번역 결과: {result['messages'][-1].content}")
print(f"감지 언어: {result['language']}")
```

### 예제 2: 다른 상태 스키마를 가진 서브그래프

```python
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# ─── 서브그래프 전용 상태 (부모와 다름) ───
class ResearchSubState(TypedDict):
    topic: str           # 서브그래프 입력
    findings: list[str]  # 서브그래프 내부 상태
    summary: str         # 서브그래프 출력


def gather_node(state: ResearchSubState) -> dict:
    """정보 수집 단계."""
    response = llm.invoke(
        f"'{state['topic']}'에 대한 핵심 사실 3가지를 간단히 나열하세요."
    )
    findings = response.content.split("\n")[:3]
    return {"findings": findings}


def summarize_node(state: ResearchSubState) -> dict:
    """수집된 정보 요약."""
    findings_text = "\n".join(state["findings"])
    response = llm.invoke(f"다음 내용을 한 문장으로 요약하세요:\n{findings_text}")
    return {"summary": response.content}


# 서브그래프 빌드
research_builder = StateGraph(ResearchSubState)
research_builder.add_node("gather", gather_node)
research_builder.add_node("summarize", summarize_node)
research_builder.add_edge(START, "gather")
research_builder.add_edge("gather", "summarize")
research_builder.add_edge("summarize", END)
research_subgraph = research_builder.compile()


# ─── 부모 그래프 상태 (서브그래프와 다름) ───
class ParentState(TypedDict):
    user_query: str   # 부모 입력
    topic: str        # 추출된 토픽
    report: str       # 최종 보고서


def extract_topic_node(state: ParentState) -> dict:
    """사용자 쿼리에서 토픽 추출."""
    response = llm.invoke(
        f"다음 질문의 핵심 주제를 한 단어 또는 짧은 구로 추출하세요: {state['user_query']}"
    )
    topic = response.content.strip()
    print(f"[부모] 추출된 토픽: {topic}")
    return {"topic": topic}


def call_research_subgraph(state: ParentState) -> dict:
    """래퍼 함수: 부모 상태 → 서브그래프 입력 → 서브그래프 출력 → 부모 상태."""
    # 1. 부모 상태에서 서브그래프 입력 구성
    sub_input = {
        "topic": state["topic"],
        "findings": [],
        "summary": "",
    }
    # 2. 서브그래프 실행
    sub_output = research_subgraph.invoke(sub_input)
    # 3. 서브그래프 출력을 부모 상태로 변환
    return {"report": f"[{state['topic']} 리서치 결과]\n{sub_output['summary']}"}


def format_report_node(state: ParentState) -> dict:
    """최종 보고서 형식화."""
    response = llm.invoke(
        f"다음 리서치 결과를 사용자 친화적으로 다시 작성하세요:\n{state['report']}"
    )
    return {"report": response.content}


# 부모 그래프 빌드
parent_builder = StateGraph(ParentState)
parent_builder.add_node("extract_topic", extract_topic_node)
parent_builder.add_node("research", call_research_subgraph)  # 래퍼 함수 사용
parent_builder.add_node("format_report", format_report_node)
parent_builder.add_edge(START, "extract_topic")
parent_builder.add_edge("extract_topic", "research")
parent_builder.add_edge("research", "format_report")
parent_builder.add_edge("format_report", END)
graph = parent_builder.compile()

result = graph.invoke({"user_query": "머신러닝이 의료 분야에 어떻게 활용되나요?", "topic": "", "report": ""})
print(f"\n최종 보고서:\n{result['report']}")
```

### 예제 3: 서브그래프 스트리밍 (subgraphs=True)

```python
import os
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    score: int


# ─── 서브그래프: 품질 평가 ───
def evaluate_node(state: State) -> dict:
    last_msg = state["messages"][-1].content
    response = llm.invoke(
        f"다음 텍스트의 품질을 1-10 점수로만 답하세요: '{last_msg[:100]}'"
    )
    try:
        score = int(response.content.strip().split()[0])
    except (ValueError, IndexError):
        score = 5
    return {"score": score}


sub_builder = StateGraph(State)
sub_builder.add_node("evaluate", evaluate_node)
sub_builder.add_edge(START, "evaluate")
sub_builder.add_edge("evaluate", END)
eval_subgraph = sub_builder.compile()

# ─── 부모 그래프 ───
def generate_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": response}


parent_builder = StateGraph(State)
parent_builder.add_node("generate", generate_node)
parent_builder.add_node("quality_check", eval_subgraph)
parent_builder.add_edge(START, "generate")
parent_builder.add_edge("generate", "quality_check")
parent_builder.add_edge("quality_check", END)
graph = parent_builder.compile()

# ── subgraphs=True: 서브그래프 내부 이벤트까지 스트리밍 ──
print("=== 서브그래프 포함 스트리밍 ===")
for namespace, chunk in graph.stream(
    {"messages": [HumanMessage(content="파이썬 장점을 설명해주세요.")], "score": 0},
    stream_mode="updates",
    subgraphs=True,  # 서브그래프 내부 이벤트도 포함
):
    # namespace: ("",) = 부모, ("quality_check:xxx",) = 서브그래프
    depth = len(namespace) - 1
    indent = "  " * depth
    if namespace == ():
        graph_label = "부모"
    else:
        graph_label = f"서브그래프({namespace[-1].split(':')[0]})"
    print(f"{indent}[{graph_label}] {list(chunk.keys())}")
```

### 예제 4: 병렬 서브그래프 (팬아웃 패턴)

```python
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


class AnalysisState(TypedDict):
    text: str
    sentiment: str
    keywords: list[str]
    summary: str


# ─── 감정 분석 서브그래프 ───
class SentimentState(TypedDict):
    text: str
    sentiment: str


def sentiment_node(state: SentimentState) -> dict:
    response = llm.invoke(f"다음 텍스트의 감정을 '긍정'/'중립'/'부정' 중 하나로만 답하세요: {state['text']}")
    return {"sentiment": response.content.strip()}


sentiment_builder = StateGraph(SentimentState)
sentiment_builder.add_node("analyze", sentiment_node)
sentiment_builder.add_edge(START, "analyze")
sentiment_builder.add_edge("analyze", END)
sentiment_subgraph = sentiment_builder.compile()

# ─── 키워드 추출 서브그래프 ───
class KeywordState(TypedDict):
    text: str
    keywords: list[str]


def keyword_node(state: KeywordState) -> dict:
    response = llm.invoke(f"다음 텍스트에서 핵심 키워드 3개를 콤마로 구분하여 추출하세요: {state['text']}")
    keywords = [k.strip() for k in response.content.split(",")]
    return {"keywords": keywords}


keyword_builder = StateGraph(KeywordState)
keyword_builder.add_node("extract", keyword_node)
keyword_builder.add_edge(START, "extract")
keyword_builder.add_edge("extract", END)
keyword_subgraph = keyword_builder.compile()

# ─── 부모 그래프: 두 서브그래프를 래퍼로 병렬 실행 ───
def run_sentiment(state: AnalysisState) -> dict:
    result = sentiment_subgraph.invoke({"text": state["text"], "sentiment": ""})
    return {"sentiment": result["sentiment"]}


def run_keywords(state: AnalysisState) -> dict:
    result = keyword_subgraph.invoke({"text": state["text"], "keywords": []})
    return {"keywords": result["keywords"]}


def aggregate_node(state: AnalysisState) -> dict:
    summary = f"감정: {state['sentiment']} | 키워드: {', '.join(state['keywords'])}"
    return {"summary": summary}


parent_builder = StateGraph(AnalysisState)
parent_builder.add_node("sentiment_analysis", run_sentiment)
parent_builder.add_node("keyword_extraction", run_keywords)
parent_builder.add_node("aggregate", aggregate_node)

parent_builder.add_edge(START, "sentiment_analysis")
parent_builder.add_edge(START, "keyword_extraction")  # 팬아웃: 동시 실행
parent_builder.add_edge("sentiment_analysis", "aggregate")
parent_builder.add_edge("keyword_extraction", "aggregate")
parent_builder.add_edge("aggregate", END)
graph = parent_builder.compile()

result = graph.invoke({
    "text": "LangGraph는 복잡한 AI 에이전트를 구축하기에 훌륭한 프레임워크입니다.",
    "sentiment": "",
    "keywords": [],
    "summary": "",
})
print(f"분석 결과: {result['summary']}")
```

---

## ✏️ 실습 과제

### 과제 1: 재사용 가능한 RAG 서브그래프

검색(Retrieval) + 생성(Generation) 로직을 서브그래프로 캡슐화하고, 부모 그래프에서 다른 질문 유형에 따라 이 RAG 서브그래프를 선택적으로 호출하세요.

### 과제 2: 중첩 서브그래프

서브그래프 안에 또 다른 서브그래프가 있는 2단계 중첩 구조를 만들고, `subgraphs=True` 스트리밍으로 각 계층의 이벤트를 구분하여 출력하세요.

### 과제 3: 서브그래프 단독 테스트

예제 2의 `research_subgraph`를 부모 없이 단독으로 실행하고 결과를 검증하는 테스트 코드를 작성하세요.

---

## ⚠️ 흔한 함정

### 1. 상태 스키마 불일치 시 직접 합성 시도

```python
# ❌ 오류: 서브그래프에 없는 키('user_query')를 부모가 전달
parent_builder.add_node("sub", subgraph)  # 키가 다른 경우 런타임 오류

# ✅ 올바름: 래퍼 함수로 변환
def call_subgraph(state: ParentState) -> dict:
    sub_out = subgraph.invoke({"topic": state["user_query"]})
    return {"result": sub_out["summary"]}
parent_builder.add_node("sub", call_subgraph)
```

### 2. 서브그래프 체크포인터 설정

부모 그래프에 checkpointer를 설정하면 서브그래프는 자동으로 부모의 체크포인터를 상속합니다. 서브그래프에 별도 checkpointer를 설정하지 마세요.

```python
# ✅ 부모에만 설정
memory = MemorySaver()
graph = parent_builder.compile(checkpointer=memory)
# subgraph는 compile()만 (checkpointer 없음)
subgraph = sub_builder.compile()
```

### 3. subgraphs=False 기본값 망각

`graph.stream()`의 `subgraphs` 기본값은 `False`입니다. 서브그래프 내부 이벤트가 보이지 않는다면 `subgraphs=True`를 명시적으로 전달하세요.

---

## ✅ 셀프 체크

- [ ] 컴파일된 서브그래프를 부모 그래프의 노드로 추가할 수 있다.
- [ ] 상태 스키마가 다를 때 래퍼 함수 패턴을 사용할 수 있다.
- [ ] `subgraphs=True`로 서브그래프 내부 이벤트를 스트리밍할 수 있다.
- [ ] 병렬 서브그래프(팬아웃) 패턴을 구현할 수 있다.
- [ ] 서브그래프를 단독으로 테스트할 수 있다.

---

## 🔗 참고 자료

- [LangGraph 서브그래프 공식 문서](https://langchain-ai.github.io/langgraph/concepts/subgraphs/)
- [서브그래프 How-to 가이드](https://langchain-ai.github.io/langgraph/how-tos/subgraph/)
- [서브그래프 스트리밍 가이드](https://langchain-ai.github.io/langgraph/how-tos/streaming-subgraphs/)

> **API 변동 주의**: 서브그래프 스트리밍의 반환 형태(`(namespace, chunk)` 튜플)는 LangGraph 버전에 따라 달라질 수 있습니다. `subgraphs=True` 사용 시 반드시 공식 문서와 버전 호환성을 확인하세요.

---

⬅️ [Phase 24: Human-in-the-Loop](./24-human-in-the-loop.md) | ➡️ [Phase 26: 장기 메모리 스토어](./26-memory-store.md)
