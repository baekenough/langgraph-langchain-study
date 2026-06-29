# Phase 00: 오리엔테이션

> **예상 소요시간**: 30분 | **난이도**: ★☆☆☆☆ | **선행 페이즈**: 없음

---

## 🎯 학습 목표

- LangChain과 LangGraph가 각각 무엇인지 설명할 수 있습니다
- 두 프레임워크의 차이와 어떤 상황에서 무엇을 선택해야 하는지 판단할 수 있습니다
- LangChain 생태계의 패키지 구조를 이해합니다
- 이 커리큘럼의 전체 흐름과 학습 방법을 파악합니다

---

## 📚 핵심 개념

### LangChain이란?

LangChain은 LLM(Large Language Model) 기반 애플리케이션을 구축하기 위한 Python(및 JavaScript) 프레임워크입니다.

핵심 아이디어는 **"체이닝"**입니다. 여러 컴포넌트(프롬프트 → 모델 → 출력 파서)를 파이프처럼 연결하여 복잡한 처리 흐름을 만듭니다.

```
사용자 입력 → 프롬프트 템플릿 → LLM → 출력 파서 → 결과
```

**LCEL(LangChain Expression Language)**은 이 체이닝을 `|` 연산자로 표현하는 LangChain의 핵심 추상화입니다:

```python
chain = prompt | model | output_parser
result = chain.invoke({"input": "안녕하세요"})
```

### LangGraph란?

LangGraph는 LangChain 팀이 만든 **상태 기계(State Machine) 기반의 에이전트 프레임워크**입니다.

LangChain의 선형 체인과 달리, LangGraph는 **그래프** 구조로 흐름을 정의합니다:
- **노드(Node)**: 처리 단계(함수)
- **엣지(Edge)**: 노드 간 연결 및 조건부 분기
- **상태(State)**: 전체 실행 흐름에서 공유되는 데이터

```
시작 → [도구 선택] → [도구 실행] → [결과 판단] → (계속? 종료?)
                           ↑__________________________|
```

**순환(Cycle)이 가능하다**는 점이 LangGraph의 가장 중요한 특징입니다.  
에이전트가 "생각하고 → 행동하고 → 관찰하고 → 다시 생각하는" 루프를 표현할 수 있습니다.

### LangChain vs LangGraph: 언제 무엇을 쓰나?

| 상황 | 선택 | 이유 |
|------|------|------|
| 단순한 Q&A, 요약, 번역 | LangChain (LCEL) | 선형 파이프라인으로 충분 |
| RAG 파이프라인 구축 | LangChain (LCEL) | 문서 검색 → LLM 체인 |
| 복잡한 에이전트 (도구 사용, 루프) | LangGraph | 조건 분기 + 반복이 필요 |
| 멀티 에이전트 시스템 | LangGraph | 에이전트 간 협력, 상태 공유 |
| 인간 개입(Human-in-the-loop) | LangGraph | 중단점, 체크포인터 지원 |
| 장기 기억(Long-term memory) | LangGraph | 영속적 상태 관리 |

> **핵심 규칙**: 흐름이 선형이면 LangChain, 반복/분기/상태가 필요하면 LangGraph.  
> 실제로는 두 가지를 함께 씁니다 — LangGraph 노드 안에서 LCEL 체인을 사용하는 식으로.

---

### 생태계 패키지 지도

LangChain은 여러 패키지로 분리되어 있습니다. 각자의 역할을 이해하면 import 경로를 헷갈리지 않습니다.

```
┌─────────────────────────────────────────────────┐
│                 langchain-core                  │  ← 핵심 추상화 (의존성 없음)
│  BaseMessage, BasePromptTemplate, Runnable, ...  │
└────────────────┬────────────────────────────────┘
                 │ 의존
┌────────────────▼────────────────────────────────┐
│                  langchain                      │  ← 체인, 에이전트, 기타 유틸
│  (langchain-core를 조합한 고수준 컴포넌트)         │
└────────────────┬────────────────────────────────┘
                 │ 모델/벡터스토어 통합
    ┌────────────┴──────────────────┐
    │                               │
┌───▼───────────┐   ┌───────────────▼─────┐
│langchain-openai│   │langchain-community  │  ← 공급자별 패키지
│langchain-chroma│   │(OpenRouter는 ChatOpenAI│
└───────────────┘   │ + base_url로 접근)   │
                    └─────────────────────┘

┌─────────────────────────────────────────────────┐
│                   langgraph                     │  ← 상태 기계 / 에이전트
│  StateGraph, START, END, MemorySaver, ...        │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│                   langsmith                     │  ← 관찰가능성, 평가, 트레이싱
└─────────────────────────────────────────────────┘
```

**패키지별 역할 요약:**

| 패키지 | 역할 | 주요 클래스/모듈 |
|--------|------|-----------------|
| `langchain-core` | 핵심 추상화, 인터페이스 | `BaseMessage`, `Runnable`, `ChatPromptTemplate` |
| `langchain` | 고수준 체인, 에이전트 유틸 | `create_retrieval_chain`, `create_react_agent` |
| `langchain-openai` | OpenAI/OpenRouter 연동 | `ChatOpenAI`(OpenRouter 채팅), `OpenAIEmbeddings`(임베딩) |
| `langchain-community` | 커뮤니티 통합 | 다양한 벡터스토어, 로더 |
| `langchain-chroma` | Chroma 벡터 DB | `Chroma` |
| `langgraph` | 상태 그래프 엔진 | `StateGraph`, `START`, `END` |
| `langsmith` | 트레이싱, 평가 | SDK + 웹 플랫폼 |

> ⚠️ **중요**: API가 빠르게 발전하므로, 코드 작성 전 [공식 문서](https://python.langchain.com/docs/)에서 최신 import 경로를 확인하세요.

---

### 이 커리큘럼의 흐름

```
Part 0 (기초)
  ↓ 환경 준비, 첫 API 호출
Part 1 (LangChain Core)
  ↓ 채팅 모델, 프롬프트, LCEL, 스트리밍, 도구 호출
Part 2 (RAG)
  ↓ 문서 처리, 임베딩, 벡터 검색, RAG 파이프라인
Part 3 (LangGraph Core)
  ↓ 상태 그래프, 노드, 순환, 체크포인터
Part 4 (에이전트)
  ↓ ReAct, 커스텀 에이전트, 멀티 에이전트
Part 5 (프로덕션)
  ↓ LangSmith, 테스트, 최적화, 배포
Capstone
  ↓ 종합 프로젝트
```

---

### 학습 방법 제안

**각 페이즈는 다음 순서로 학습하세요:**

1. **읽기**: 핵심 개념을 먼저 읽습니다 (코드 없이 개념만)
2. **실행**: 코드 예제를 직접 타이핑하고 실행합니다 (복붙 금지!)
3. **변형**: 값을 바꿔보고, 에러를 만들어보고, 수정합니다
4. **과제**: 실습 과제를 스스로 풀어봅니다
5. **체크**: 셀프 체크로 이해를 확인합니다

**팁:**
- Jupyter Notebook(`.ipynb`)으로 실험하고, 완성된 코드는 `.py`로 옮기세요
- LangSmith 트레이싱을 Part 0부터 켜두면 디버깅이 훨씬 편합니다
- 에러 메시지를 두려워하지 마세요 — LangChain 에러는 대체로 설명이 친절합니다

---

## 💻 코드 예제

이 페이즈에서는 실행할 코드가 없습니다. 다음 페이즈(Phase 01)에서 환경을 구성한 후 첫 코드를 실행합니다.

---

## ✏️ 실습 과제

1. [LangChain 공식 문서](https://python.langchain.com/docs/introduction/)의 "Introduction" 페이지를 읽어보세요
2. [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)의 "Why LangGraph?" 섹션을 읽어보세요
3. 내가 만들고 싶은 LLM 앱을 하나 떠올리고, LangChain과 LangGraph 중 어떤 것이 더 적합한지 판단해보세요

---

## ⚠️ 흔한 함정

- **`from langchain import ChatOpenAI` — 이제 작동하지 않습니다**  
  올바른 경로: `from langchain_openai import ChatOpenAI`  
  LangChain v0.2 이후 패키지가 분리되었습니다.

- **LangChain과 LangGraph를 별개로 생각하기**  
  실제로는 LangGraph 노드 안에서 LCEL 체인을 사용하는 경우가 많습니다. 두 가지를 함께 배우세요.

- **버전 확인 소홀**  
  LangChain은 빠르게 발전합니다. 검색 결과가 구버전 API일 수 있으니, 항상 공식 문서의 버전을 확인하세요.

---

## ✅ 셀프 체크

- [ ] LangChain이 "체이닝"을 통해 LLM 앱을 만드는 프레임워크임을 이해했다
- [ ] LangGraph가 순환 가능한 상태 기계 기반 에이전트 프레임워크임을 이해했다
- [ ] 단순 파이프라인은 LangChain, 복잡한 에이전트는 LangGraph를 선택하는 기준을 이해했다
- [ ] `langchain-core`, `langchain`, `langchain-openai`, `langgraph`의 역할 차이를 설명할 수 있다
- [ ] 커리큘럼 전체 6개 파트의 흐름을 파악했다

---

## 🔗 참고 자료

- [LangChain 공식 문서](https://python.langchain.com/docs/introduction/)
- [LangGraph 공식 문서](https://langchain-ai.github.io/langgraph/)
- [LangChain 생태계 패키지 목록](https://python.langchain.com/docs/integrations/providers/)
- [LangSmith 플랫폼](https://smith.langchain.com/)

---

다음: [Phase 01: UV로 Python 환경 구성](./01-uv-python-setup.md) →
