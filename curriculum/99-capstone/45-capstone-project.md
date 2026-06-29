# Phase 45: 캡스톤 프로젝트

**이전**: [Phase 44 — 관측성과 모니터링](../05-production/44-observability-monitoring.md)  
**다음**: [커리큘럼 인덱스로 돌아가기](../README.md)

---

## 개요

이 캡스톤 프로젝트는 Phase 00~44에서 학습한 내용을 **하나의 실제 동작하는 시스템**으로 통합하는 최종 과제입니다.  
단순한 예제가 아니라, 실제 프로덕션 배포가 가능한 수준의 LLM 애플리케이션을 완성합니다.

### 제안 프로젝트: 문서 기반 멀티에이전트 어시스턴트

> 사용자가 업로드한 문서(PDF, 웹 페이지, 텍스트)를 기반으로 질문에 답하고,  
> 필요하면 외부 검색까지 수행하는 **멀티에이전트 RAG 어시스턴트**를 구축합니다.

**핵심 특징:**
- RAG(Part 2) + LangGraph 상태/영속성/HITL(Part 3) + 멀티에이전트(Part 4) + 평가/배포(Part 5) 통합
- 채팅 모델: OpenRouter (`OPENROUTER_API_KEY`)
- 임베딩: OpenAI `text-embedding-3-small` (`OPENAI_API_KEY`)
- 다중 대화 세션을 지원하는 체크포인터 기반 영속성
- LangSmith 트레이싱으로 전 과정 관측

---

## 학습 목표

이 프로젝트를 완료하면 다음을 할 수 있습니다:

- [ ] 여러 파트의 지식을 결합하여 완전한 LLM 시스템을 설계하고 구현한다
- [ ] 멀티에이전트 아키텍처에서 각 에이전트의 역할과 통신 방법을 이해한다
- [ ] RAG 파이프라인을 에이전트와 통합하여 동적 검색을 구현한다
- [ ] Human-in-the-Loop으로 민감한 작업에 사용자 승인을 추가한다
- [ ] LangSmith로 시스템 전체의 품질을 평가하고 개선 포인트를 찾는다
- [ ] FastAPI 또는 LangGraph Platform으로 시스템을 배포한다

---

## 시스템 아키텍처

```
사용자 입력
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                  슈퍼바이저 에이전트                   │
│  (Supervisor Agent — Phase 32)                      │
│  - 사용자 의도 분류                                   │
│  - 하위 에이전트 라우팅                               │
└──────────┬──────────────┬──────────────┬────────────┘
           │              │              │
           ▼              ▼              ▼
   ┌───────────┐  ┌──────────────┐  ┌──────────────┐
   │  RAG 에이  │  │  검색 에이전  │  │  분석 에이전  │
   │  전트      │  │  트 (Tavily) │  │  트           │
   │ (Phase 34)│  │ (Phase 27,29)│  │ (Phase 28,31)│
   └─────┬─────┘  └──────┬───────┘  └──────┬───────┘
         │               │                 │
         ▼               ▼                 ▼
   ┌─────────────────────────────────────────────┐
   │            벡터 스토어 (ChromaDB)             │
   │   + 체크포인터 (영속성, Phase 22)             │
   │   + 메모리 스토어 (장기 기억, Phase 26)        │
   └─────────────────────────────────────────────┘
         │
         ▼
   ┌──────────────────────────────┐
   │   Human-in-the-Loop 게이트   │
   │   (민감한 작업 시 사용자 승인) │
   │   (Phase 24)                 │
   └──────────────────────────────┘
         │
         ▼
   ┌──────────────────────────────┐
   │       응답 생성 및 반환        │
   │  + LangSmith 트레이싱 기록    │
   └──────────────────────────────┘
         │
         ▼
   FastAPI 엔드포인트 (Phase 42)
   또는 LangGraph Platform (Phase 43)
```

---

## 마일스톤

### M1: 기반 RAG 시스템 구축

**참조 페이즈:** Part 2 전체 ([Phase 11](../02-rag/11-document-loaders.md)~[17](../02-rag/17-rag-evaluation.md))

**목표:** 문서를 업로드하고 질문에 답하는 기본 RAG 시스템을 완성합니다.

```python
# M1 완료 기준: 다음 코드가 동작해야 합니다
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 문서 로드 → 분할 → 임베딩 → 벡터 스토어 저장
# 질문 → 검색 → 답변 생성
```

**체크리스트:**
- [ ] PDF/텍스트/웹 페이지 로드 (`DocumentLoader`)
- [ ] 적절한 청크 크기로 텍스트 분할 (`RecursiveCharacterTextSplitter`)
- [ ] OpenAI 임베딩으로 벡터화 (`text-embedding-3-small`)
- [ ] ChromaDB에 저장하고 유사도 검색 동작 확인
- [ ] 기본 RAG 체인 구성 (retriever + prompt + llm)
- [ ] RAG 품질 평가 (Context Precision, Answer Relevance)

---

### M2: LangGraph 상태 그래프로 확장

**참조 페이즈:** Part 3 전체 ([Phase 18](../03-langgraph-core/18-langgraph-intro-stategraph.md)~[26](../03-langgraph-core/26-memory-store.md))

**목표:** 단순 체인을 LangGraph StateGraph로 변환하고 영속성을 추가합니다.

```python
# M2 완료 기준: thread_id로 대화 세션을 유지합니다
from langgraph.graph import StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = StateGraph(MessagesState)
# ... 노드, 엣지 정의 ...
app = graph.compile(checkpointer=checkpointer)

# 같은 thread_id로 여러 번 호출해도 맥락이 유지됩니다
config = {"configurable": {"thread_id": "session-001"}}
app.invoke({"messages": [("user", "안녕하세요")]}, config=config)
```

**체크리스트:**
- [ ] `StateGraph` + `MessagesState`로 그래프 구성
- [ ] 리트리버 노드, 답변 생성 노드 분리
- [ ] `MemorySaver` 체크포인터로 대화 이력 유지
- [ ] `InMemoryStore`로 사용자별 장기 선호도 저장
- [ ] 스트리밍 응답 지원 (`astream_events`)

---

### M3: 멀티에이전트 아키텍처 도입

**참조 페이즈:** Part 4 ([Phase 27](../04-agents/27-react-agent.md), [32](../04-agents/32-multiagent-supervisor.md), [33](../04-agents/33-multiagent-collaboration.md), [34](../04-agents/34-agentic-rag.md))

**목표:** 슈퍼바이저 에이전트가 RAG 에이전트와 검색 에이전트를 조율합니다.

```
슈퍼바이저 (의도 분류)
├── "문서에서 답할 수 있음" → RAG 에이전트
├── "최신 정보 필요" → 검색 에이전트 (Tavily)
└── "복잡한 분석" → 분석 에이전트 (Plan-and-Execute)
```

**체크리스트:**
- [ ] 슈퍼바이저 노드: 사용자 의도를 분류하여 적절한 에이전트로 라우팅
- [ ] RAG 에이전트: 벡터 스토어에서 검색 후 답변 생성
- [ ] 검색 에이전트: Tavily로 실시간 웹 검색
- [ ] 분석 에이전트: 복잡한 질문을 계획하고 순차 실행
- [ ] 에이전트 간 결과 통합 및 최종 응답 생성

---

### M4: Human-in-the-Loop 및 안전장치 추가

**참조 페이즈:** [Phase 24](../03-langgraph-core/24-human-in-the-loop.md), [Phase 40](../05-production/40-error-handling-resilience.md), [Phase 41](../05-production/41-security-guardrails.md)

**목표:** 민감한 작업에 사용자 승인 게이트를 추가하고 에러에 강한 시스템을 만듭니다.

```python
# M4 완료 기준: 외부 검색 전 사용자 승인을 요청합니다
from langgraph.types import interrupt

def search_agent_node(state):
    if state["requires_external_search"]:
        user_approval = interrupt({
            "message": "외부 검색을 수행합니다. 계속하시겠습니까?",
            "query": state["search_query"]
        })
        if not user_approval:
            return {"response": "외부 검색을 취소했습니다."}
    # 검색 실행...
```

**체크리스트:**
- [ ] 외부 검색 전 `interrupt`로 사용자 승인 요청
- [ ] 민감한 키워드 감지 가드레일 구현
- [ ] `with_retry()`로 API 호출 재시도 로직 추가
- [ ] `with_fallbacks()`로 모델 장애 시 대체 모델 전환
- [ ] 에러 타입별 처리 로직 구현

---

### M5: 평가, 배포, 모니터링

**참조 페이즈:** Part 5 전체 ([Phase 35](../05-production/35-langsmith-tracing.md)~[44](../05-production/44-observability-monitoring.md))

**목표:** 시스템을 정량적으로 평가하고 API 서버로 배포합니다.

**체크리스트:**

_평가_
- [ ] LangSmith 트레이싱 설정 (`LANGSMITH_API_KEY` + `LANGSMITH_PROJECT`)
- [ ] 평가용 질문-답변 데이터셋 최소 20개 작성
- [ ] LangSmith Evaluator로 답변 정확도, 컨텍스트 활용도 측정
- [ ] 취약 케이스(edge case) 식별 및 프롬프트 개선

_배포_
- [ ] FastAPI 서버 구현 (`/chat`, `/upload`, `/health` 엔드포인트)
- [ ] 스트리밍 응답 지원 (`StreamingResponse`)
- [ ] 또는 LangGraph Platform CLI로 배포 (`langgraph up`)

_모니터링_
- [ ] LangSmith 대시보드에서 에러율, 지연시간, 비용 확인
- [ ] 이상 응답 알림 설정

---

## 평가 기준 (Rubric)

| 항목 | 배점 | 기준 |
|------|------|------|
| **M1 RAG 기반 기능** | 20점 | 문서 업로드, 검색, 답변 생성이 정확히 동작함 |
| **M2 LangGraph 통합** | 20점 | StateGraph, 체크포인터, 스트리밍이 모두 동작함 |
| **M3 멀티에이전트** | 25점 | 슈퍼바이저가 올바른 에이전트로 라우팅함 |
| **M4 HITL + 안전장치** | 15점 | interrupt 동작, 에러 처리, 가드레일 구현 |
| **M5 평가 + 배포** | 20점 | LangSmith 평가 결과 문서화, API 배포 완료 |
| **보너스: 코드 품질** | +10점 | 타입 힌트, 문서화, 테스트 코드 포함 |
| **보너스: 확장 기능** | +10점 | 아래 확장 아이디어 중 1개 이상 구현 |

**합격 기준:** 60점 이상 (보너스 제외 기준)

---

## 아키텍처 결정 사항

시스템을 설계하면서 다음 질문에 대한 답을 `docs/architecture-decisions.md`에 문서화하세요.

1. **청크 크기와 오버랩**: 어떤 기준으로 결정했나요? 실험 결과는?
2. **임베딩 모델 선택**: `text-embedding-3-small` vs `text-embedding-3-large` 비교
3. **벡터 스토어**: ChromaDB를 선택한 이유. 프로덕션에서는?
4. **슈퍼바이저 라우팅 전략**: 규칙 기반인가요, LLM 기반인가요? 트레이드오프는?
5. **체크포인터**: `MemorySaver`(메모리) vs `SqliteSaver`(파일) vs `PostgresSaver`(DB)

---

## 확장 아이디어

기본 구현을 완료한 후 다음 기능을 추가해볼 수 있습니다:

### 기능 확장
- **문서 업데이트 감지**: 동일 문서가 다시 업로드될 때 벡터 스토어를 갱신하는 로직
- **인용 출처 표시**: 답변에 사용된 문서 청크의 출처(파일명, 페이지)를 함께 반환
- **다국어 지원**: 사용자 질문 언어 감지 → 같은 언어로 답변
- **요약 기능**: 긴 문서를 자동 요약하는 별도 에이전트

### 기술 심화
- **Agentic Chunking**: LLM이 문서 구조를 이해하고 의미 단위로 분할
- **Adaptive RAG**: 검색 결과 품질에 따라 재검색 또는 외부 검색 자동 전환
- **Reflection Loop**: 생성된 답변을 다시 검토하고 자기교정 ([Phase 31](../04-agents/31-reflection-self-correction.md) 적용)
- **멀티모달**: 이미지가 포함된 PDF 처리 (Claude Vision 모델 활용)

### 인프라
- **PostgreSQL 체크포인터**: 멀티 서버 환경을 위한 영속적 체크포인터
- **Redis 캐싱**: 동일 질문에 대한 임베딩/응답 캐싱
- **Docker Compose**: 전체 시스템 컨테이너화

---

## 프로젝트 디렉토리 구조 예시

```
my-doc-assistant/
├── pyproject.toml
├── .env
├── .env.example
├── README.md
├── docs/
│   └── architecture-decisions.md
├── src/
│   ├── __init__.py
│   ├── config.py              # 환경 변수, 설정
│   ├── ingest.py              # 문서 로드 + 임베딩 + 저장
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py           # AgentState 정의
│   │   ├── supervisor.py      # 슈퍼바이저 노드
│   │   ├── rag_agent.py       # RAG 에이전트
│   │   ├── search_agent.py    # 검색 에이전트
│   │   └── builder.py         # 그래프 조립
│   └── api/
│       ├── __init__.py
│       └── server.py          # FastAPI 서버
├── tests/
│   ├── test_ingest.py
│   ├── test_rag_agent.py
│   └── test_graph.py
└── evaluation/
    ├── dataset.jsonl           # 평가 데이터셋
    └── eval_runner.py         # LangSmith 평가 실행
```

---

## 제출 및 회고 체크리스트

### 기술 요구사항

- [ ] M1~M5 모든 마일스톤 완료
- [ ] `README.md`에 시스템 설명, 설치 방법, 실행 방법 포함
- [ ] `docs/architecture-decisions.md`에 주요 설계 결정 사항 문서화
- [ ] LangSmith 평가 결과 스크린샷 또는 리포트 첨부
- [ ] API 엔드포인트 배포 완료 및 동작 확인

### 회고 질문

프로젝트를 완료한 후 다음 질문에 답하며 학습을 정리하세요.

1. **가장 어려웠던 부분**은 무엇이었나요? 어떻게 해결했나요?
2. **예상과 달랐던 것**은 무엇인가요? (성능, API 동작, 비용 등)
3. **다시 한다면 다르게 할 것**은 무엇인가요?
4. **LangGraph vs 단순 LangChain 체인**: 어떤 상황에서 LangGraph가 유리한가요?
5. **프로덕션 고려 사항**: 이 시스템을 실제 서비스로 운영한다면 추가로 무엇이 필요한가요?

---

## 참고 자료

| 자료 | 링크 |
|------|------|
| LangGraph 공식 문서 | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) |
| LangGraph How-to Guides | [How-to Guides](https://langchain-ai.github.io/langgraph/how-tos/) |
| LangChain 공식 문서 | [python.langchain.com](https://python.langchain.com) |
| LangSmith 문서 | [docs.smith.langchain.com](https://docs.smith.langchain.com) |
| OpenRouter 문서 | [openrouter.ai/docs](https://openrouter.ai/docs) |
| 용어집 | [appendix/glossary.md](../appendix/glossary.md) |
| 치트시트 | [appendix/cheatsheet.md](../appendix/cheatsheet.md) |
| 트러블슈팅 | [appendix/troubleshooting.md](../appendix/troubleshooting.md) |

---

**이전**: [Phase 44 — 관측성과 모니터링](../05-production/44-observability-monitoring.md)  
**다음**: [커리큘럼 인덱스로 돌아가기](../README.md)
