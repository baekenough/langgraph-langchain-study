# Phase 04: Chat Models와 Messages

> 예상 소요시간: 60분 | 난이도: ★★☆☆☆ | 선행 페이즈: [03-models-and-keys](../00-foundations/03-models-and-keys.md)

---

## 🎯 학습 목표

- LangChain의 Chat Model 통일 인터페이스가 무엇인지 이해합니다.
- `invoke`, `stream`, `batch` 세 가지 호출 방식의 차이를 설명할 수 있습니다.
- `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage` 타입을 구분하고 사용할 수 있습니다.
- OpenRouter를 통해 여러 LLM 제공자의 모델을 동일한 코드로 교체할 수 있습니다.
- `temperature`, `max_tokens` 파라미터가 응답에 미치는 영향을 설명할 수 있습니다.

---

## 📚 핵심 개념

### 1. Chat Model 통일 인터페이스

LangChain의 핵심 가치 중 하나는 **다양한 LLM 제공자를 동일한 인터페이스로 사용**할 수 있다는 점입니다.
OpenAI, Anthropic, Google 등 어느 모델을 쓰더라도 `.invoke()`, `.stream()`, `.batch()` 메서드 시그니처가 동일합니다.

이 커리큘럼에서는 **OpenRouter**를 프록시로 사용합니다. `langchain_openai`의 `ChatOpenAI`를 그대로 쓰되
`base_url`과 `api_key`만 OpenRouter로 지정하면 됩니다.

```python
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",        # OpenRouter 모델 ID
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)
```

### 2. 세 가지 호출 방식

| 메서드 | 설명 | 반환 타입 |
|--------|------|-----------|
| `invoke(input)` | 단건 동기 호출. 응답 완성 후 반환 | `AIMessage` |
| `stream(input)` | 토큰 단위 스트리밍. 제너레이터 반환 | `Generator[AIMessageChunk]` |
| `batch(inputs)` | 여러 입력 병렬 처리 | `list[AIMessage]` |

비동기 버전(`ainvoke`, `astream`, `abatch`)은 Phase 08에서 다룹니다.

### 3. 메시지 타입

Chat Model은 단순 문자열이 아니라 **메시지 객체 리스트**를 입력으로 받습니다.

| 메시지 타입 | 역할 | 사용 시점 |
|------------|------|----------|
| `SystemMessage` | 어시스턴트의 역할·행동 지침 | 대화 최상단 |
| `HumanMessage` | 사용자 입력 | 매 턴 사용자 발화 |
| `AIMessage` | 모델 응답 | 대화 이력 저장 시 |
| `ToolMessage` | 도구 실행 결과 전달 | 툴 호출 후 (Phase 09) |

### 4. 주요 파라미터

- **`temperature`** (0.0~2.0): 0에 가까울수록 결정론적, 높을수록 창의적
- **`max_tokens`**: 최대 출력 토큰 수. 비용·응답 길이 제어
- **`model`**: OpenRouter 모델 ID (`"openai/gpt-4o-mini"`, `"anthropic/claude-3.5-haiku"` 등)

---

## 💻 코드 예제

### 예제 1: 기본 설정과 첫 호출

```python
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# 문자열 직접 전달 (내부에서 HumanMessage로 변환됨)
response = llm.invoke("Python에서 리스트와 튜플의 차이점을 한 문장으로 설명해줘.")

print(type(response))    # <class 'langchain_core.messages.ai.AIMessage'>
print(response.content)  # 실제 응답 텍스트
print(response.usage_metadata)  # 토큰 사용량 정보
```

### 예제 2: 메시지 타입으로 역할 부여 및 멀티턴 대화

```python
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# 시스템 메시지로 역할 부여
messages = [
    SystemMessage(content="당신은 Python 전문가입니다. 간결하고 실용적인 답변만 제공합니다."),
    HumanMessage(content="리스트 컴프리헨션을 설명해주세요."),
]

response = llm.invoke(messages)
print(response.content)

# 대화 이력 유지: AIMessage를 리스트에 추가하여 문맥 전달
messages.append(response)
messages.append(HumanMessage(content="딕셔너리 컴프리헨션도 같은 방식으로 보여주세요."))

follow_up = llm.invoke(messages)
print(follow_up.content)
```

> **비용 예고**: 멀티턴 대화는 매 턴 **전체 이력**을 API에 재전송하므로, 대화가 길어질수록 입력 토큰 비용이 선형으로 증가합니다. 고정된 시스템 프롬프트나 이력 앞부분을 provider 서버에서 재사용하는 **provider prefix 캐시(cache_control)** 로 이 비용을 완화하는 방법은 [Phase 39: 비용·캐싱 최적화](../05-production/39-cost-caching-optimization.md)에서 다룹니다.

### 예제 3: stream — 토큰 단위 실시간 출력

```python
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)

print("응답: ", end="", flush=True)
for chunk in llm.stream("파이썬의 GIL이란 무엇인가요? 150자 이내로 설명해주세요."):
    print(chunk.content, end="", flush=True)
print()  # 최종 개행
```

### 예제 4: batch — 여러 질문 병렬 처리

```python
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

questions = [
    "Python의 `__init__`과 `__new__`의 차이점은?",
    "제너레이터와 이터레이터의 차이점은?",
    "데코레이터 패턴의 활용 예시는?",
]

# 세 질문을 병렬로 처리 — 순차 invoke보다 빠름
responses = llm.batch(questions)

for q, r in zip(questions, responses):
    print(f"Q: {q}")
    print(f"A: {r.content[:120]}...")
    print()
```

### 예제 5: OpenRouter로 모델 교체 — 코드 변경 없음

```python
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI

load_dotenv()

def create_llm(model_id: str, temperature: float = 0) -> ChatOpenAI:
    """OpenRouter 모델 ID를 받아 ChatOpenAI 인스턴스를 반환합니다."""
    return ChatOpenAI(
        model=model_id,
        api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
    )

# 동일한 코드로 다른 제공자 모델 사용
models = {
    "GPT-4o-mini": create_llm("openai/gpt-4o-mini"),
    "Claude 3.5 Haiku": create_llm("anthropic/claude-3.5-haiku"),
    "Gemini Flash": create_llm("google/gemini-2.0-flash-001"),
}

question = "Python의 async/await를 한 문장으로 설명해주세요."

for name, llm in models.items():
    response = llm.invoke(question)
    print(f"[{name}]: {response.content}")
    print()
```

### 예제 6: AIMessage 응답 구조 탐색

```python
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

response = llm.invoke("안녕하세요!")

# AIMessage 주요 속성
print("content:", response.content)
print("type:", response.type)              # "ai"
print("id:", response.id)                  # 메시지 고유 ID
print("usage_metadata:", response.usage_metadata)   # 입력/출력 토큰 수
print("response_metadata:", response.response_metadata)  # 제공자별 추가 정보
```

---

## ✏️ 실습 과제

### 과제 1: 역할 기반 어시스턴트

`SystemMessage`로 특정 전문가 역할(예: 코드 리뷰어, 여행 가이드)을 부여하고, `HumanMessage`/`AIMessage`를 번갈아 추가하며 3턴 대화를 구현하세요.

### 과제 2: temperature 비교 실험

동일한 창의적 질문(예: "파이썬으로 할 수 있는 재미있는 프로젝트를 하나 제안해줘.")에 대해 `temperature=0`, `0.5`, `1.2`로 각각 호출하여 응답 다양성을 비교해보세요.

### 과제 3: batch 성능 측정

`time.time()`을 사용하여 `batch()`로 5개 질문을 처리하는 시간과, `for` 루프로 `invoke()`를 5번 호출하는 시간을 비교해보세요.

### 과제 4: 모델 비교

OpenRouter의 두 모델에 동일한 코딩 문제를 주고 응답 품질(`content`)과 토큰 비용(`usage_metadata`)을 비교해보세요.

---

## ⚠️ 흔한 함정

**1. 구버전 임포트 경로 사용**

```python
# 금지 — deprecated
from langchain.chat_models import ChatOpenAI

# 올바른 최신 경로
from langchain_openai import ChatOpenAI
```

**2. response를 문자열처럼 사용**

```python
response = llm.invoke("안녕")
print(response)          # AIMessage(content='안녕하세요!', ...) — 객체 출력
print(response.content)  # "안녕하세요!" — 텍스트만 출력
```

**3. OpenRouter 모델 ID 형식 혼동**

OpenRouter는 `"provider/model-name"` 형식을 요구합니다.

```python
# 틀림 (OpenAI 직접 접근 방식)
model="gpt-4o-mini"

# 올바름 (OpenRouter 방식)
model="openai/gpt-4o-mini"
```

**4. API 변동 주의**

> LangChain은 빠르게 발전합니다. 코드가 예상대로 동작하지 않으면 [공식 문서](https://python.langchain.com/docs/integrations/chat/)를 확인하세요.

---

## ✅ 셀프 체크

- [ ] `ChatOpenAI`를 OpenRouter `base_url`과 함께 초기화할 수 있다.
- [ ] `invoke`, `stream`, `batch`의 차이를 동료에게 설명할 수 있다.
- [ ] `SystemMessage`, `HumanMessage`, `AIMessage`를 사용하여 멀티턴 대화를 구현할 수 있다.
- [ ] `response.content`로 텍스트를 추출할 수 있다.
- [ ] 모델 ID만 바꿔서 Anthropic, Google 모델로 교체할 수 있다.

---

## 🔗 참고 자료

- [LangChain Chat Models 개요](https://python.langchain.com/docs/concepts/chat_models/)
- [LangChain Messages 타입](https://python.langchain.com/docs/concepts/messages/)
- [ChatOpenAI API 레퍼런스](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)
- [OpenRouter 지원 모델 목록](https://openrouter.ai/models)

---

← [Phase 03: 모델과 API 키 관리](../00-foundations/03-models-and-keys.md) | [Phase 05: 프롬프트 템플릿](05-prompt-templates.md) →
