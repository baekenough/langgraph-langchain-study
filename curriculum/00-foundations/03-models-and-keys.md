# Phase 03: 모델 제공자와 API 키

> **예상 소요시간**: 45분  
> **난이도**: ★★☆☆☆  
> **선행 페이즈**: [Phase 02: VSCode 개발 환경](./02-vscode-setup.md)

---

## 🎯 학습 목표

- 이 커리큘럼의 모델 제공자 전략(OpenRouter + OpenAI)을 이해한다
- OpenRouter가 무엇이고 왜 사용하는지 설명할 수 있다
- API 키를 안전하게 관리하는 방법을 실천한다
- `python-dotenv`로 환경변수를 로드하는 패턴을 익힌다
- LangChain을 통해 첫 번째 LLM 호출("Hello World")을 성공한다
- 토큰, 비용, 모델 선택 기준을 이해한다

---

## 📚 핵심 개념

### 이 커리큘럼의 모델 전략

이 커리큘럼은 **채팅 LLM**과 **임베딩**에 서로 다른 제공자를 사용합니다:

| 용도 | 제공자 | 클래스 | 환경변수 |
|------|--------|--------|----------|
| 채팅 모델 | **OpenRouter** | `ChatOpenAI` + `base_url` | `OPENROUTER_API_KEY` |
| 임베딩 | **OpenAI** | `OpenAIEmbeddings` | `OPENAI_API_KEY` |

**왜 이 조합인가?**

- **OpenRouter**: 단일 API 키로 OpenAI, Anthropic, Google, Meta 등 수십 가지 모델에 접근. 모델을 자유롭게 교체하며 학습 가능.
- **OpenAI 임베딩**: OpenRouter는 임베딩 API를 제공하지 않음. RAG 파트(Phase 13)에서 `text-embedding-3-small` 사용.

### OpenRouter란?

[OpenRouter](https://openrouter.ai/)는 **OpenAI 호환 단일 API**로 여러 벤더의 모델을 접근할 수 있는 라우팅 서비스입니다.

```
사용자 코드
    │  ChatOpenAI(base_url="https://openrouter.ai/api/v1",
    │             model="anthropic/claude-3.5-haiku")
    ▼
OpenRouter API  ──────────────┬──── OpenAI (GPT-4o-mini)
                              ├──── Anthropic (Claude)
                              ├──── Google (Gemini)
                              └──── Meta (Llama)
```

**OpenRouter 모델 ID 형식**: `{벤더}/{모델명}` (예: `"openai/gpt-4o-mini"`, `"anthropic/claude-3.5-haiku"`, `"google/gemini-2.0-flash"`)

모델 목록과 가격은 [openrouter.ai/models](https://openrouter.ai/models)에서 확인하세요.

### 모델 선택 가이드

**이 커리큘럼의 기본 모델**: `openai/gpt-4o-mini`
- 저렴하고 빠름 (학습 예제에 적합)
- 도구 호출(Function Calling) 완벽 지원
- 필요 시 `anthropic/claude-3.5-haiku`, `google/gemini-2.0-flash`로 교체 가능

```python
# 같은 코드로 모델만 바꿔서 비교
llm_gpt = ChatOpenAI(model="openai/gpt-4o-mini", ...)
llm_claude = ChatOpenAI(model="anthropic/claude-3.5-haiku", ...)
llm_gemini = ChatOpenAI(model="google/gemini-2.0-flash", ...)
```

### 로컬 모델 (Ollama) 옵션

API 키 없이 로컬에서 실험하고 싶다면 Ollama를 사용할 수 있습니다:

```bash
# Ollama 설치 (macOS)
brew install ollama

# 모델 다운로드
ollama pull llama3.2

# 서버 시작
ollama serve
```

LangChain에서 사용:
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")
```

> **주의**: 로컬 모델은 성능이 상대적으로 낮고 일부 고급 기능(도구 호출 등)이 제한될 수 있습니다. 커리큘럼의 고급 예제는 OpenRouter 기준으로 작성되어 있습니다.

### API 키 관리

**가장 중요한 보안 규칙**: API 키는 절대로 코드에 직접 쓰거나 Git에 커밋하면 안 됩니다.

**올바른 방법 — `.env` 파일 + `python-dotenv`:**

```ini
# .env 파일 (Git에 커밋하지 않음)
OPENROUTER_API_KEY=sk-or-v1-abc123...
OPENAI_API_KEY=sk-proj-abc123...
```

```python
# Python 코드
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일을 읽어 환경변수로 등록

api_key = os.environ["OPENROUTER_API_KEY"]
```

> `.gitignore`에 `.env`가 포함되어 있는지 반드시 확인하세요.

### 토큰과 비용 개념

LLM은 **토큰(token)** 단위로 텍스트를 처리하고 과금합니다.

- 한국어 1글자 ≈ 1~2 토큰
- 영어 1단어 ≈ 1~1.5 토큰
- `gpt-4o-mini` (OpenRouter 경유): 입력 $0.15/1M 토큰, 출력 $0.60/1M 토큰

**비용 계산 예시:**
```
"안녕하세요, LangChain에 대해 설명해줘" ≈ 15 토큰
응답: "LangChain은..." ≈ 200 토큰
총 비용: ≈ $0.0001 (0.1원 미만)
```

> 학습 중 비용을 아끼려면:
> - `openai/gpt-4o-mini` 사용 (가장 저렴한 편)
> - OpenRouter 대시보드에서 월 크레딧 한도 설정

---

## 💻 코드 예제

### API 키 발급

1. **OpenRouter**: [openrouter.ai](https://openrouter.ai) → 로그인 → Keys → Create Key
2. **OpenAI**: [platform.openai.com](https://platform.openai.com) → API Keys → Create new secret key  
   (RAG 파트 이전에는 필수 아님)
3. **LangSmith**: [smith.langchain.com](https://smith.langchain.com) → Settings → API Keys

### `.env` 파일 설정

```bash
# .env.example을 복사
cp .env.example .env

# .env 파일을 열어 실제 API 키 입력
```

```ini
# .env 파일 내용 예시
OPENROUTER_API_KEY=sk-or-v1-여기에_실제_키_입력
OPENAI_API_KEY=sk-proj-여기에_실제_키_입력
LANGSMITH_API_KEY=lsv2_pt_여기에_실제_키_입력
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=langchain-langgraph-study
```

### Hello World — OpenRouter

```python
# examples/hello_openrouter.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 환경변수 로드
load_dotenv()

# OpenRouter로 채팅 모델 초기화
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",       # OpenRouter 모델 ID: {벤더}/{모델}
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,                    # 재현 가능한 결과를 위해 0으로 설정
)

# 첫 번째 호출
response = llm.invoke([HumanMessage(content="안녕하세요! 한 문장으로 자기소개해주세요.")])

print(type(response))            # <class 'langchain_core.messages.ai.AIMessage'>
print(response.content)          # AI의 응답 텍스트
print(response.usage_metadata)   # 토큰 사용량
```

실행:
```bash
uv run python examples/hello_openrouter.py
```

### 다른 벤더 모델로 교체

`model` 파라미터만 바꾸면 다른 벤더 모델을 즉시 사용할 수 있습니다:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# OpenRouter 공통 설정을 딕셔너리로 관리
openrouter_kwargs = {
    "api_key": os.environ["OPENROUTER_API_KEY"],
    "base_url": "https://openrouter.ai/api/v1",
    "temperature": 0,
}

# GPT-4o-mini (OpenAI)
llm_gpt = ChatOpenAI(model="openai/gpt-4o-mini", **openrouter_kwargs)

# Claude 3.5 Haiku (Anthropic via OpenRouter)
llm_claude = ChatOpenAI(model="anthropic/claude-3.5-haiku", **openrouter_kwargs)

# Gemini 2.0 Flash (Google via OpenRouter)
llm_gemini = ChatOpenAI(model="google/gemini-2.0-flash", **openrouter_kwargs)

question = "파이썬 리스트 컴프리헨션을 한 문장으로 설명해줘."

for name, llm in [("GPT-4o-mini", llm_gpt), ("Claude-3.5-Haiku", llm_claude)]:
    response = llm.invoke(question)
    print(f"\n[{name}]")
    print(response.content)
```

### 임베딩 미리보기 (OpenAI 직접 사용)

> 임베딩은 **Phase 13 (RAG)** 에서 본격적으로 다룹니다.  
> 여기서는 OpenAI 임베딩 클래스가 어떻게 생겼는지만 확인합니다.

```python
# RAG 파트(Phase 13)에서 사용할 임베딩 — 지금은 참고만
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# OpenAI 임베딩 (OPENAI_API_KEY 사용)
# OpenRouter는 임베딩 API를 제공하지 않으므로 OpenAI 직접 사용
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"],  # OPENROUTER_API_KEY가 아님!
)

# Phase 13에서 본격 학습
# vector = embeddings.embed_query("안녕하세요")
```

### 모델 파라미터 이해

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,    # 창의성 (0=결정적, 1=창의적)
    max_tokens=500,     # 최대 출력 토큰
    timeout=30,         # 타임아웃 (초)
    max_retries=2,      # 실패 시 재시도 횟수
)

messages = [
    SystemMessage(content="당신은 친절한 Python 튜터입니다."),
    HumanMessage(content="리스트 컴프리헨션을 한 문장으로 설명해줘."),
]

response = llm.invoke(messages)
print(response.content)
```

### 환경변수 검증 유틸리티

`utils/check_env.py`는 `.env`에 설정된 API 키를 **실제 API 호출**로 유효성까지 검증하는 스크립트입니다.

```bash
# 전체 검증 — 키 존재 여부 + 라이브 API 호출
uv run python utils/check_env.py

# 오프라인 모드 — API 호출 없이 키 존재 여부만 확인
uv run python utils/check_env.py --offline
```

출력 예시:
```
=== API 키 검증 — 라이브 (API 호출 포함) ===

[ 필수 ]
  ✓ OK      OpenRouter (채팅)         키 유효 (sk-or-...abcd) — 응답 수신 성공
  ✓ OK      OpenAI (임베딩)           키 유효 (sk-pro...wxyz) — 1536차원 벡터 수신

[ 선택 ]
  ✓ OK      LangSmith (트레이싱)      키 유효 (lsv2_p...efgh) — LangSmith 연결 성공
  – SKIP    Tavily (웹 검색)          TAVILY_API_KEY 미설정 — 웹 검색 도구 비활성 (선택 사항)

────────────────────────────────────────────────────────────
결과: 필수 키 모두 정상 — 학습을 시작할 수 있습니다.
       선택 키: 1개 정상, 1개 미설정
```

- **필수 키**(OpenRouter, OpenAI) 중 하나라도 실패하면 종료코드 1을 반환합니다.
- **선택 키**(LangSmith, Tavily)는 미설정 시 `– SKIP`으로 표시되며 종료코드에 영향을 주지 않습니다.
- 키 값은 마스킹(앞 6자 + 뒤 4자)으로만 표시됩니다.

---

## ✏️ 실습 과제

1. **API 키 설정**: `.env` 파일에 최소한 `OPENROUTER_API_KEY`를 설정하세요.

2. **Hello World 실행**: `examples/hello_openrouter.py`를 만들고 실행하여 응답을 확인하세요.

3. **모델 교체**: 같은 질문을 `openai/gpt-4o-mini`와 `anthropic/claude-3.5-haiku`에 각각 보내고 응답을 비교해보세요.

4. **비용 추적**: `response.usage_metadata`를 출력하여 입력/출력 토큰 수를 확인하세요. 10번 호출하면 얼마나 드는지 계산해보세요.

---

## ⚠️ 흔한 함정

- **`.env` 파일을 `.gitignore`에 추가 안 함**: API 키가 GitHub에 공개되면 즉시 악용됩니다. 항상 `.env`가 `.gitignore`에 있는지 확인하세요.

- **`OPENAI_API_KEY`와 `OPENROUTER_API_KEY` 혼동**: 채팅은 `OPENROUTER_API_KEY`, 임베딩은 `OPENAI_API_KEY`입니다. 잘못된 키를 전달하면 인증 오류가 납니다.

- **OpenRouter 모델 ID 형식**: `"gpt-4o-mini"` (잘못됨) vs `"openai/gpt-4o-mini"` (올바름). 벤더 접두어를 반드시 붙여야 합니다.

- **`base_url` 누락**: OpenRouter를 사용할 때 `base_url="https://openrouter.ai/api/v1"`을 명시하지 않으면 OpenAI 서버로 요청이 가서 인증 오류가 발생합니다.

- **`load_dotenv()` 호출 위치**: 환경변수를 사용하는 코드보다 **먼저** 호출해야 합니다. 파일 상단, `import` 직후가 좋습니다.

- **rate limit 오류**: `429 Too Many Requests` 오류 시 잠시 기다리거나 OpenRouter 대시보드에서 크레딧 한도를 확인하세요.

---

## ✅ 셀프 체크

- [ ] `.env` 파일이 생성되었고 `.gitignore`에 포함되어 있다
- [ ] OpenRouter가 단일 API로 여러 벤더 모델에 접근하는 서비스임을 이해했다
- [ ] `ChatOpenAI` + `base_url` 조합으로 OpenRouter를 사용하는 패턴을 익혔다
- [ ] 채팅은 `OPENROUTER_API_KEY`, 임베딩은 `OPENAI_API_KEY`를 쓰는 이유를 설명할 수 있다
- [ ] `load_dotenv()`를 사용하여 환경변수를 로드하는 패턴을 이해했다
- [ ] `llm.invoke(messages)`로 첫 번째 LLM 호출에 성공했다
- [ ] 토큰이 무엇이고 비용과 어떻게 연결되는지 이해했다

---

## 🔗 참고 자료

- [OpenRouter 공식 사이트](https://openrouter.ai/)
- [OpenRouter 모델 목록 및 가격](https://openrouter.ai/models)
- [LangChain ChatOpenAI 문서](https://python.langchain.com/docs/integrations/chat/openai/)
- [OpenAI 임베딩 문서](https://python.langchain.com/docs/integrations/text_embedding/openai/)
- [python-dotenv 문서](https://saurabh-kumar.com/python-dotenv/)
- [Ollama 공식 사이트](https://ollama.ai/)

> ⚠️ **API 변경 주의**: LangChain의 모델 클래스와 파라미터는 버전에 따라 변경될 수 있습니다. 최신 정보는 [공식 문서](https://python.langchain.com/docs/)를 확인하세요.

---

← 이전: [Phase 02: VSCode 개발 환경](./02-vscode-setup.md) | 다음: [Phase 04: 채팅 모델과 메시지](../01-langchain-core/04-chat-models-messages.md) →
