# 학습 자료 모음 (Resources)

이 커리큘럼을 학습하는 데 유용한 공식 문서, 튜토리얼, 커뮤니티 자료를 정리했습니다.

---

## 공식 문서

### LangChain

| 자료 | URL | 설명 |
|------|-----|------|
| Python 공식 문서 | [python.langchain.com](https://python.langchain.com) | API 레퍼런스, How-to Guides, 개념 설명 |
| API 레퍼런스 | [api.python.langchain.com](https://api.python.langchain.com) | 클래스/함수 상세 레퍼런스 |
| Conceptual Guides | [python.langchain.com/docs/concepts](https://python.langchain.com/docs/concepts/) | 핵심 개념 심층 설명 |
| Integration 목록 | [python.langchain.com/docs/integrations](https://python.langchain.com/docs/integrations/providers/) | 지원되는 모든 모델/도구 목록 |
| LCEL 공식 가이드 | [python.langchain.com/docs/concepts/lcel](https://python.langchain.com/docs/concepts/lcel) | LCEL 작동 방식 심층 설명 |

### LangGraph

| 자료 | URL | 설명 |
|------|-----|------|
| 공식 문서 | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) | LangGraph 핵심 문서 |
| How-to Guides | [langchain-ai.github.io/langgraph/how-tos](https://langchain-ai.github.io/langgraph/how-tos/) | 특정 기능 구현 방법 |
| Tutorials | [langchain-ai.github.io/langgraph/tutorials](https://langchain-ai.github.io/langgraph/tutorials/) | 실전 튜토리얼 모음 |
| Conceptual Guide | [langchain-ai.github.io/langgraph/concepts](https://langchain-ai.github.io/langgraph/concepts/) | State, Node, Edge 개념 설명 |
| Cloud (Platform) | [langchain-ai.github.io/langgraph/cloud](https://langchain-ai.github.io/langgraph/cloud/) | LangGraph Platform 배포 |
| Prebuilt Components | [langgraph/prebuilt](https://langchain-ai.github.io/langgraph/reference/prebuilt/) | `create_react_agent`, `ToolNode` 등 내장 컴포넌트 |

### LangSmith

| 자료 | URL | 설명 |
|------|-----|------|
| 공식 문서 | [docs.smith.langchain.com](https://docs.smith.langchain.com) | 트레이싱, 평가, 데이터셋 관리 |
| Python SDK | [docs.smith.langchain.com/sdk/python](https://docs.smith.langchain.com/sdk/python) | Python SDK 레퍼런스 |
| 평가 가이드 | [docs.smith.langchain.com/evaluation](https://docs.smith.langchain.com/evaluation) | Evaluator 작성 방법 |

---

## 모델 제공자

### OpenRouter (채팅 모델)

| 자료 | URL | 설명 |
|------|-----|------|
| 공식 문서 | [openrouter.ai/docs](https://openrouter.ai/docs) | API 사용법, 모델 목록 |
| 모델 목록 | [openrouter.ai/models](https://openrouter.ai/models) | 가격, 컨텍스트 크기, 속도 비교 |
| API 레퍼런스 | [openrouter.ai/docs/api-reference](https://openrouter.ai/docs/api-reference) | OpenAI 호환 엔드포인트 상세 |
| Rate Limits | [openrouter.ai/docs/limits](https://openrouter.ai/docs/limits) | 요청 제한 정책 |

> OpenRouter는 OpenAI Chat Completions API와 호환됩니다.  
> `base_url="https://openrouter.ai/api/v1"`로 설정하면 `ChatOpenAI`에서 바로 사용할 수 있습니다.

### OpenAI (임베딩)

| 자료 | URL | 설명 |
|------|-----|------|
| 임베딩 가이드 | [platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings) | 임베딩 모델 선택 및 사용법 |
| 모델 비교 | [platform.openai.com/docs/models](https://platform.openai.com/docs/models) | text-embedding-3-small vs large |
| API 레퍼런스 | [platform.openai.com/docs/api-reference/embeddings](https://platform.openai.com/docs/api-reference/embeddings) | Embeddings API 상세 |

---

## GitHub 저장소

| 저장소 | URL | 설명 |
|--------|-----|------|
| langchain-ai/langchain | [github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain) | LangChain 메인 저장소 |
| langchain-ai/langgraph | [github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | LangGraph 메인 저장소 |
| langchain-ai/langsmith-sdk | [github.com/langchain-ai/langsmith-sdk](https://github.com/langchain-ai/langsmith-sdk) | LangSmith Python SDK |
| langchain-ai/langchain-academy | [github.com/langchain-ai/langchain-academy](https://github.com/langchain-ai/langchain-academy) | 공식 튜토리얼 코드 (강력 추천) |
| langchain-ai/rag-from-scratch | [github.com/langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch) | RAG 처음부터 구현 예제 |

---

## 블로그 및 튜토리얼

### LangChain 공식 블로그

| 자료 | URL |
|------|-----|
| LangChain Blog | [blog.langchain.dev](https://blog.langchain.dev) |
| LangChain Academy (강의) | [academy.langchain.com](https://academy.langchain.com) |

### 추천 튜토리얼

| 제목 | URL | 설명 |
|------|-----|------|
| LangGraph Quickstart | [langchain-ai.github.io/langgraph/tutorials/introduction](https://langchain-ai.github.io/langgraph/tutorials/introduction/) | LangGraph 공식 시작 가이드 |
| Build a RAG App | [python.langchain.com/docs/tutorials/rag](https://python.langchain.com/docs/tutorials/rag/) | LangChain 공식 RAG 튜토리얼 |
| Build an Agent | [python.langchain.com/docs/tutorials/agents](https://python.langchain.com/docs/tutorials/agents/) | LangChain 공식 에이전트 튜토리얼 |
| LangSmith Quickstart | [docs.smith.langchain.com/tutorials/Developers/quickstart](https://docs.smith.langchain.com/tutorials/Developers/quickstart) | LangSmith 시작 가이드 |

### 유튜브

| 채널 | URL | 설명 |
|------|-----|------|
| LangChain 공식 | [youtube.com/@LangChain](https://www.youtube.com/@LangChain) | 공식 데모, 웨비나, 튜토리얼 |
| Greg Kamradt | [youtube.com/@DataIndependent](https://www.youtube.com/@DataIndependent) | LangChain 심층 튜토리얼 |
| Sam Witteveen | [youtube.com/@samwitteveenai](https://www.youtube.com/@samwitteveenai) | LLM 개발 실전 예제 |

---

## 커뮤니티

| 커뮤니티 | URL | 설명 |
|----------|-----|------|
| LangChain Discord | [discord.gg/langchain](https://discord.gg/langchain) | 공식 커뮤니티, Q&A, 채널별 주제 |
| GitHub Discussions (LangChain) | [github.com/langchain-ai/langchain/discussions](https://github.com/langchain-ai/langchain/discussions) | 기능 요청, 버그 논의 |
| GitHub Discussions (LangGraph) | [github.com/langchain-ai/langgraph/discussions](https://github.com/langchain-ai/langgraph/discussions) | LangGraph 관련 질문 |
| Reddit r/LangChain | [reddit.com/r/LangChain](https://www.reddit.com/r/LangChain/) | 커뮤니티 공유, 프로젝트 소개 |

---

## 도구 및 라이브러리

### 이 커리큘럼에서 사용하는 주요 패키지

| 패키지 | PyPI | 문서 |
|--------|------|------|
| langchain | [pypi.org/project/langchain](https://pypi.org/project/langchain/) | LangChain 코어 |
| langchain-openai | [pypi.org/project/langchain-openai](https://pypi.org/project/langchain-openai/) | OpenAI, OpenRouter 연동 |
| langgraph | [pypi.org/project/langgraph](https://pypi.org/project/langgraph/) | LangGraph |
| langsmith | [pypi.org/project/langsmith](https://pypi.org/project/langsmith/) | LangSmith SDK |
| langchain-chroma | [pypi.org/project/langchain-chroma](https://pypi.org/project/langchain-chroma/) | ChromaDB 연동 |
| chromadb | [pypi.org/project/chromadb](https://pypi.org/project/chromadb/) | 로컬 벡터 스토어 |
| langchain-community | [pypi.org/project/langchain-community](https://pypi.org/project/langchain-community/) | 문서 로더, 기타 통합 |
| pydantic | [pypi.org/project/pydantic](https://pypi.org/project/pydantic/) | 데이터 검증, Structured Output |
| fastapi | [pypi.org/project/fastapi](https://pypi.org/project/fastapi/) | API 서버 배포 |
| tavily-python | [pypi.org/project/tavily-python](https://pypi.org/project/tavily-python/) | Tavily 검색 API |

### UV (패키지 관리)

| 자료 | URL |
|------|-----|
| UV 공식 문서 | [docs.astral.sh/uv](https://docs.astral.sh/uv/) |
| UV GitHub | [github.com/astral-sh/uv](https://github.com/astral-sh/uv) |
| UV 설치 | [docs.astral.sh/uv/getting-started/installation](https://docs.astral.sh/uv/getting-started/installation/) |

---

## 추천 학습 순서

처음 LangChain/LangGraph를 시작하는 분에게는 다음 순서를 권장합니다:

1. **이 커리큘럼 Phase 00~10** — 환경 설정과 LangChain 핵심 개념
2. **LangChain 공식 튜토리얼** — [Build a RAG App](https://python.langchain.com/docs/tutorials/rag/) 실습
3. **이 커리큘럼 Phase 11~26** — RAG와 LangGraph 핵심
4. **LangGraph Academy** — [langchain-ai.github.io/langgraph/tutorials/introduction](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
5. **이 커리큘럼 Phase 27~44** — 에이전트와 프로덕션
6. **langchain-academy GitHub** — 공식 강의 코드 심화 학습
7. **캡스톤 프로젝트** — 종합 응용

---

*링크가 만료되었거나 더 좋은 자료가 있다면 이슈로 알려주세요.*
