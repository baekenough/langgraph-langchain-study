# LangChain + LangGraph 학습 커리큘럼

Python으로 LLM 애플리케이션을 만드는 실전 커리큘럼입니다.  
LangChain v0.3+, LangGraph 최신 API, OpenRouter(채팅) + OpenAI(임베딩) 스택을 기준으로 작성되었습니다.

---

## 이 커리큘럼이란?

Part 0의 환경 설정부터 Part 5의 프로덕션 배포까지, **46개 페이즈(Phase 00~45)**를 순서대로 따라가며  
LangChain과 LangGraph의 핵심 개념을 직접 코드로 익히는 자기주도 학습 과정입니다.

### 누구를 위한 커리큘럼인가?

| 대상 | 전제 조건 |
|------|----------|
| Python 기반 LLM 개발 입문자 | Python 기초 문법 (함수, 클래스, 리스트/딕셔너리) |
| LangChain/LangGraph를 처음 접하는 엔지니어 | pip/가상환경 개념 이해 |
| RAG, 에이전트 시스템을 만들고 싶은 개발자 | 터미널 기본 명령어 사용 가능 |

> LangChain이나 LLM API를 전혀 사용해본 적 없어도 Part 0부터 따라갈 수 있습니다.

---

## 도구 스택

| 구분 | 도구 |
|------|------|
| **언어** | Python 3.11+ |
| **패키지 관리** | [UV](https://docs.astral.sh/uv/) |
| **에디터** | VSCode + Python 확장 |
| **채팅 모델** | [OpenRouter](https://openrouter.ai) (`OPENROUTER_API_KEY`) |
| **임베딩 모델** | [OpenAI](https://platform.openai.com) (`OPENAI_API_KEY`) |
| **트레이싱(선택)** | [LangSmith](https://smith.langchain.com) (`LANGSMITH_API_KEY`) |
| **검색 툴(선택)** | [Tavily](https://tavily.com) (`TAVILY_API_KEY`) |

> OpenRouter를 채팅 모델에 사용하는 이유: 단일 API 키로 GPT-4o, Claude, Llama 등 다양한 모델을 비교하며 학습할 수 있습니다.  
> 임베딩은 OpenAI `text-embedding-3-small`을 기준으로 합니다.

---

## 빠른 시작 (Quick Start)

```bash
# 1. 저장소 클론 후 프로젝트 루트로 이동
cd langraph_langchain

# 2. UV로 의존성 설치 (pyproject.toml 기준)
uv sync --dev

# 3. 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 아래 키 입력
# OPENROUTER_API_KEY=sk-or-...
# OPENAI_API_KEY=sk-...
# LANGSMITH_API_KEY=ls__...   (선택)
# TAVILY_API_KEY=tvly-...     (선택)

# 4. 커리큘럼 시작
open curriculum/00-foundations/00-orientation.md
# 또는 VSCode에서 파일 열기
```

> `.env.example`과 `pyproject.toml`은 프로젝트 루트에 위치합니다.  
> UV가 설치되지 않았다면 Phase 01을 먼저 참고하세요: [01-uv-python-setup.md](00-foundations/01-uv-python-setup.md)

---

## 전체 로드맵

### Part 0: Foundations — 환경 준비

> Python 개발 환경과 도구를 설정하고 첫 번째 LLM API 호출을 경험합니다.  
> **예상 시간: 약 2.5시간 | 난이도: 입문**

| Phase | 제목 | 링크 | 난이도 | 예상 시간 |
|-------|------|------|--------|----------|
| 00 | 오리엔테이션 | [00-orientation.md](00-foundations/00-orientation.md) | 입문 | 30분 |
| 01 | UV + Python 환경 설정 | [01-uv-python-setup.md](00-foundations/01-uv-python-setup.md) | 입문 | 45분 |
| 02 | VSCode 설정 | [02-vscode-setup.md](00-foundations/02-vscode-setup.md) | 입문 | 30분 |
| 03 | 모델과 API 키 | [03-models-and-keys.md](00-foundations/03-models-and-keys.md) | 입문 | 45분 |

---

### Part 1: LangChain Core — 핵심 빌딩 블록

> LangChain의 핵심 추상화인 ChatModel, PromptTemplate, OutputParser, LCEL을 학습합니다.  
> 툴 호출과 구조화된 출력까지 커버하여 실용적인 LLM 애플리케이션의 기초를 쌓습니다.  
> **예상 시간: 약 6시간 | 난이도: 초급**

| Phase | 제목 | 링크 | 난이도 | 예상 시간 |
|-------|------|------|--------|----------|
| 04 | 채팅 모델과 메시지 | [04-chat-models-messages.md](01-langchain-core/04-chat-models-messages.md) | 초급 | 60분 |
| 05 | 프롬프트 템플릿 | [05-prompt-templates.md](01-langchain-core/05-prompt-templates.md) | 초급 | 45분 |
| 06 | 출력 파싱 | [06-output-parsing.md](01-langchain-core/06-output-parsing.md) | 초급 | 45분 |
| 07 | LCEL과 Runnable | [07-lcel-runnables.md](01-langchain-core/07-lcel-runnables.md) | 초급 | 60분 |
| 08 | 스트리밍과 비동기 | [08-streaming-async.md](01-langchain-core/08-streaming-async.md) | 초급 | 45분 |
| 09 | 툴 호출 | [09-tool-calling.md](01-langchain-core/09-tool-calling.md) | 초급 | 60분 |
| 10 | 구조화된 출력 | [10-structured-output.md](01-langchain-core/10-structured-output.md) | 초급 | 45분 |

---

### Part 2: RAG — 검색 증강 생성

> 외부 문서를 LLM에 연결하는 RAG 파이프라인을 단계별로 구축합니다.  
> 문서 로딩부터 임베딩, 벡터 검색, 고급 RAG 기법, 평가까지 다룹니다.  
> **예상 시간: 약 7시간 | 난이도: 초급~중급**

| Phase | 제목 | 링크 | 난이도 | 예상 시간 |
|-------|------|------|--------|----------|
| 11 | 문서 로더 | [11-document-loaders.md](02-rag/11-document-loaders.md) | 초급 | 45분 |
| 12 | 텍스트 분할 | [12-text-splitting.md](02-rag/12-text-splitting.md) | 초급 | 45분 |
| 13 | 임베딩 | [13-embeddings.md](02-rag/13-embeddings.md) | 초급 | 45분 |
| 14 | 벡터 스토어 | [14-vector-stores.md](02-rag/14-vector-stores.md) | 초급 | 60분 |
| 15 | 리트리버와 기본 RAG | [15-retrievers-basic-rag.md](02-rag/15-retrievers-basic-rag.md) | 중급 | 60분 |
| 16 | 고급 RAG | [16-advanced-rag.md](02-rag/16-advanced-rag.md) | 중급 | 90분 |
| 17 | RAG 평가 | [17-rag-evaluation.md](02-rag/17-rag-evaluation.md) | 중급 | 60분 |

---

### Part 3: LangGraph Core — 상태 기반 워크플로우

> LangGraph의 StateGraph로 복잡한 워크플로우를 표현하는 방법을 학습합니다.  
> 영속성, 스트리밍, 인간 개입(HITL), 서브그래프, 장기 메모리까지 다룹니다.  
> **예상 시간: 약 9시간 | 난이도: 중급~고급**

| Phase | 제목 | 링크 | 난이도 | 예상 시간 |
|-------|------|------|--------|----------|
| 18 | LangGraph 소개와 StateGraph | [18-langgraph-intro-stategraph.md](03-langgraph-core/18-langgraph-intro-stategraph.md) | 중급 | 60분 |
| 19 | 상태와 리듀서 | [19-state-reducers.md](03-langgraph-core/19-state-reducers.md) | 중급 | 60분 |
| 20 | 노드, 엣지, 라우팅 | [20-nodes-edges-routing.md](03-langgraph-core/20-nodes-edges-routing.md) | 중급 | 60분 |
| 21 | 사이클과 반복 | [21-cycles-iteration.md](03-langgraph-core/21-cycles-iteration.md) | 중급 | 60분 |
| 22 | 영속성과 체크포인터 | [22-persistence-checkpointers.md](03-langgraph-core/22-persistence-checkpointers.md) | 중급 | 60분 |
| 23 | 스트리밍 | [23-streaming.md](03-langgraph-core/23-streaming.md) | 중급 | 45분 |
| 24 | Human-in-the-Loop | [24-human-in-the-loop.md](03-langgraph-core/24-human-in-the-loop.md) | 중급 | 60분 |
| 25 | 서브그래프 | [25-subgraphs.md](03-langgraph-core/25-subgraphs.md) | 고급 | 75분 |
| 26 | 메모리 스토어 | [26-memory-store.md](03-langgraph-core/26-memory-store.md) | 고급 | 60분 |

---

### Part 4: Agents — 자율 에이전트 시스템

> ReAct 에이전트부터 멀티에이전트 협업까지, 복잡한 작업을 자율적으로 수행하는 에이전트를 구축합니다.  
> Plan-and-Execute, 반성/자기교정, 슈퍼바이저 패턴, 에이전틱 RAG를 다룹니다.  
> **예상 시간: 약 11시간 | 난이도: 고급**

| Phase | 제목 | 링크 | 난이도 | 예상 시간 |
|-------|------|------|--------|----------|
| 27 | ReAct 에이전트 | [27-react-agent.md](04-agents/27-react-agent.md) | 중급 | 75분 |
| 28 | 커스텀 에이전트 그래프 | [28-custom-agent-graph.md](04-agents/28-custom-agent-graph.md) | 고급 | 90분 |
| 29 | 툴 설계 패턴 | [29-tool-design-patterns.md](04-agents/29-tool-design-patterns.md) | 고급 | 75분 |
| 30 | Plan-and-Execute | [30-plan-and-execute.md](04-agents/30-plan-and-execute.md) | 고급 | 90분 |
| 31 | 반성과 자기교정 | [31-reflection-self-correction.md](04-agents/31-reflection-self-correction.md) | 고급 | 75분 |
| 32 | 멀티에이전트 슈퍼바이저 | [32-multiagent-supervisor.md](04-agents/32-multiagent-supervisor.md) | 고급 | 90분 |
| 33 | 멀티에이전트 협업 | [33-multiagent-collaboration.md](04-agents/33-multiagent-collaboration.md) | 고급 | 90분 |
| 34 | 에이전틱 RAG | [34-agentic-rag.md](04-agents/34-agentic-rag.md) | 고급 | 90분 |

---

### Part 5: Production — 프로덕션 준비

> 실제 서비스를 위한 필수 요소들을 학습합니다.  
> LangSmith 트레이싱/평가, 테스트, 비용 최적화, 에러 처리, 보안, FastAPI/LangGraph Platform 배포까지 다룹니다.  
> **예상 시간: 약 11시간 | 난이도: 중급~고급**

| Phase | 제목 | 링크 | 난이도 | 예상 시간 |
|-------|------|------|--------|----------|
| 35 | LangSmith 트레이싱 | [35-langsmith-tracing.md](05-production/35-langsmith-tracing.md) | 중급 | 60분 |
| 36 | LangSmith 평가 | [36-langsmith-evaluation.md](05-production/36-langsmith-evaluation.md) | 중급 | 75분 |
| 37 | 프롬프트 관리 | [37-prompt-management.md](05-production/37-prompt-management.md) | 중급 | 45분 |
| 38 | 테스팅 전략 | [38-testing-strategies.md](05-production/38-testing-strategies.md) | 중급 | 60분 |
| 39 | 비용과 캐싱 최적화 | [39-cost-caching-optimization.md](05-production/39-cost-caching-optimization.md) | 중급 | 60분 |
| 40 | 에러 처리와 복원력 | [40-error-handling-resilience.md](05-production/40-error-handling-resilience.md) | 중급 | 60분 |
| 41 | 보안과 가드레일 | [41-security-guardrails.md](05-production/41-security-guardrails.md) | 고급 | 75분 |
| 42 | FastAPI로 배포 | [42-deploy-fastapi.md](05-production/42-deploy-fastapi.md) | 고급 | 90분 |
| 43 | LangGraph Platform 배포 | [43-deploy-langgraph-platform.md](05-production/43-deploy-langgraph-platform.md) | 고급 | 75분 |
| 44 | 관측성과 모니터링 | [44-observability-monitoring.md](05-production/44-observability-monitoring.md) | 고급 | 60분 |

---

### Capstone — 종합 프로젝트

> Part 0~5의 학습 내용을 통합하여 실제 동작하는 LLM 애플리케이션을 완성합니다.  
> **예상 시간: 8~16시간 | 난이도: 종합**

| Phase | 제목 | 링크 | 난이도 | 예상 시간 |
|-------|------|------|--------|----------|
| 45 | 캡스톤 프로젝트 | [45-capstone-project.md](99-capstone/45-capstone-project.md) | 종합 | 8~16시간 |

---

### 부록 (Appendix)

| 문서 | 설명 |
|------|------|
| [glossary.md](appendix/glossary.md) | LangChain/LangGraph 핵심 용어 한국어 정의 |
| [resources.md](appendix/resources.md) | 공식 문서, GitHub, 블로그, 커뮤니티 링크 |
| [cheatsheet.md](appendix/cheatsheet.md) | 자주 쓰는 코드 스니펫 빠른 참조 |
| [troubleshooting.md](appendix/troubleshooting.md) | 흔한 에러와 해결 방법 |

---

## 학습 방법 제안

1. **순서대로 진행합니다.** 각 페이즈는 이전 페이즈의 개념을 전제로 합니다.  
   단, Part 1(LangChain Core)과 Part 2(RAG)는 독립적으로 학습해도 괜찮습니다.

2. **코드는 직접 타이핑합니다.** 복붙보다 손으로 치는 과정에서 기억에 더 잘 남습니다.  
   `cheatsheet.md`는 막혔을 때 참고용으로 사용하세요.

3. **막히면 두 곳을 먼저 확인합니다.**
   - [appendix/troubleshooting.md](appendix/troubleshooting.md) — 흔한 에러 해결
   - [공식 LangChain 문서](https://python.langchain.com) / [LangGraph 문서](https://langchain-ai.github.io/langgraph/)

4. **실습 코드를 직접 변형해봅니다.** 예시 코드를 이해했다면, 파라미터를 바꾸거나  
   새로운 입력을 넣어보면서 동작을 관찰하세요.

5. **LangSmith 트레이싱은 일찍 설정합니다.** Phase 35를 마치기 전이라도  
   `LANGSMITH_API_KEY`를 설정해두면 디버깅에 크게 도움이 됩니다.

---

## 진도 체크리스트

아래 체크박스를 복사하여 본인의 노트나 이 파일에서 직접 체크하며 진도를 관리하세요.

### Part 0: Foundations

- [ ] Phase 00: 오리엔테이션
- [ ] Phase 01: UV + Python 환경 설정
- [ ] Phase 02: VSCode 설정
- [ ] Phase 03: 모델과 API 키

### Part 1: LangChain Core

- [ ] Phase 04: 채팅 모델과 메시지
- [ ] Phase 05: 프롬프트 템플릿
- [ ] Phase 06: 출력 파싱
- [ ] Phase 07: LCEL과 Runnable
- [ ] Phase 08: 스트리밍과 비동기
- [ ] Phase 09: 툴 호출
- [ ] Phase 10: 구조화된 출력

### Part 2: RAG

- [ ] Phase 11: 문서 로더
- [ ] Phase 12: 텍스트 분할
- [ ] Phase 13: 임베딩
- [ ] Phase 14: 벡터 스토어
- [ ] Phase 15: 리트리버와 기본 RAG
- [ ] Phase 16: 고급 RAG
- [ ] Phase 17: RAG 평가

### Part 3: LangGraph Core

- [ ] Phase 18: LangGraph 소개와 StateGraph
- [ ] Phase 19: 상태와 리듀서
- [ ] Phase 20: 노드, 엣지, 라우팅
- [ ] Phase 21: 사이클과 반복
- [ ] Phase 22: 영속성과 체크포인터
- [ ] Phase 23: 스트리밍
- [ ] Phase 24: Human-in-the-Loop
- [ ] Phase 25: 서브그래프
- [ ] Phase 26: 메모리 스토어

### Part 4: Agents

- [ ] Phase 27: ReAct 에이전트
- [ ] Phase 28: 커스텀 에이전트 그래프
- [ ] Phase 29: 툴 설계 패턴
- [ ] Phase 30: Plan-and-Execute
- [ ] Phase 31: 반성과 자기교정
- [ ] Phase 32: 멀티에이전트 슈퍼바이저
- [ ] Phase 33: 멀티에이전트 협업
- [ ] Phase 34: 에이전틱 RAG

### Part 5: Production

- [ ] Phase 35: LangSmith 트레이싱
- [ ] Phase 36: LangSmith 평가
- [ ] Phase 37: 프롬프트 관리
- [ ] Phase 38: 테스팅 전략
- [ ] Phase 39: 비용과 캐싱 최적화
- [ ] Phase 40: 에러 처리와 복원력
- [ ] Phase 41: 보안과 가드레일
- [ ] Phase 42: FastAPI로 배포
- [ ] Phase 43: LangGraph Platform 배포
- [ ] Phase 44: 관측성과 모니터링

### Capstone

- [ ] Phase 45: 캡스톤 프로젝트

---

## 총 예상 학습 시간

| 파트 | 시간 |
|------|------|
| Part 0: Foundations | 약 2.5시간 |
| Part 1: LangChain Core | 약 6시간 |
| Part 2: RAG | 약 7시간 |
| Part 3: LangGraph Core | 약 9시간 |
| Part 4: Agents | 약 11시간 |
| Part 5: Production | 약 11시간 |
| **Phase 00~44 소계** | **약 47시간** |
| Capstone | 8~16시간 |
| **총합** | **약 55~63시간** |

> 위 시간은 개념 이해 + 실습 코드 작성을 포함한 추정치입니다.  
> 배경 지식과 학습 속도에 따라 개인차가 있습니다.
