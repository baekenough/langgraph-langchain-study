# Phase 37: 프롬프트 관리

| 항목 | 내용 |
|------|------|
| 소요 시간 | 약 75분 |
| 난이도 | ★★★☆☆ |
| 선행 학습 | Phase 35 (LangSmith 트레이싱) |

---

## 🎯 학습 목표

- LangSmith Prompt Hub의 역할과 이점을 설명할 수 있습니다.
- `client.pull_prompt()`로 허브에서 프롬프트를 불러와 체인에 사용합니다.
- `client.push_prompt()`로 프롬프트를 허브에 저장하고 버전을 관리합니다.
- 커밋 해시 또는 태그로 특정 프롬프트 버전을 고정합니다.
- 코드와 프롬프트를 분리하는 이유와 팀 협업 이점을 이해합니다.
- 공개 허브 프롬프트를 활용하는 방법을 익힙니다.

---

## 📚 핵심 개념

### 1. 왜 프롬프트를 코드와 분리하나요?

프롬프트를 코드에 하드코딩하면 다음 문제가 발생합니다:

```
문제점
├── 프롬프트 수정 = 코드 변경 = 배포 필요
├── PM/도메인 전문가가 직접 수정 불가
├── 버전 관리가 git에 묻혀 추적 어려움
└── A/B 테스트 시 코드 분기 필요
```

LangSmith Prompt Hub를 사용하면:

```
장점
├── 코드 배포 없이 프롬프트 업데이트 가능
├── 비개발자도 Prompt Hub UI에서 수정 가능
├── 커밋 단위 버전 관리 + 태그
├── 평가와 연계해 프롬프트 성능 추적
└── 팀 전체가 공유하는 프롬프트 레지스트리
```

### 2. Prompt Hub 구조

```
LangSmith Prompt Hub
├── 내 조직 (private)
│   ├── customer-support-v2 (커밋: abc123)
│   ├── rag-answer (태그: production, latest)
│   └── classification (커밋: def456)
└── 공개 허브 (langchain-ai/...)
    ├── langchain-ai/rag-prompt
    └── langchain-ai/openai-functions-agent
```

### 3. 버전 관리 패턴

| 방법 | 문법 | 적합한 용도 |
|------|------|------------|
| 최신 버전 | `"my-prompt"` | 개발 환경 |
| 커밋 해시 고정 | `"my-prompt:abc12345"` | 프로덕션 |
| 태그 | `"my-prompt:production"` | 환경별 분리 |

> **권장 패턴**: 프로덕션에서는 반드시 커밋 해시나 안정 태그로 고정하세요.
> `latest`는 언제든 변경될 수 있어 예기치 않은 동작을 유발합니다.

### 4. pull_prompt 내부 동작

```python
client.pull_prompt("my-prompt:production")
# ↓ 반환 타입: ChatPromptTemplate, PromptTemplate, 또는 RunnableSequence
# ↓ 직접 chain에 연결 가능
```

---

## 💻 코드 예제

### 예제 1: 프롬프트 허브에 업로드

```python
# push_prompt.py
import os
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

client = Client()

# 업로드할 프롬프트 정의
customer_support_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 {company_name}의 친절한 고객 지원 담당자입니다.
다음 원칙을 항상 따르세요:
1. 고객의 문제를 공감하며 경청합니다.
2. 명확하고 실행 가능한 해결책을 제시합니다.
3. 전문 용어는 피하고 쉬운 언어를 사용합니다.
4. 필요하면 상위 부서로 에스컬레이션을 제안합니다.

현재 날짜: {current_date}
담당 부서: {department}"""),
    ("placeholder", "{conversation_history}"),
    ("human", "{customer_message}"),
])

# 허브에 업로드
# 프롬프트 이름은 "계정/이름" 형식 (공개) 또는 "이름" 형식 (개인)
prompt_name = "customer-support-assistant"

url = client.push_prompt(
    prompt_name,
    object=customer_support_prompt,
    description="다국어 지원 가능한 고객 지원 담당자 프롬프트",
    is_public=False,  # 팀 내부용
)
print(f"프롬프트 업로드 완료: {url}")


# 두 번째 버전 업로드 (에스컬레이션 로직 추가)
customer_support_v2 = ChatPromptTemplate.from_messages([
    ("system", """당신은 {company_name}의 전문 고객 지원 담당자입니다.
다음 원칙을 항상 따르세요:
1. 고객의 문제를 공감하며 경청합니다.
2. 명확하고 실행 가능한 해결책을 제시합니다.
3. 전문 용어는 피하고 쉬운 언어를 사용합니다.
4. 필요하면 상위 부서로 에스컬레이션을 제안합니다.
5. 민감한 정보(카드번호, 비밀번호)는 절대 요청하지 않습니다.

[에스컬레이션 기준]
- 기술적 환불: 결제팀 연결
- 제품 결함: 품질팀 연결
- 법적 이슈: 법무팀 연결

현재 날짜: {current_date}
담당 부서: {department}"""),
    ("placeholder", "{conversation_history}"),
    ("human", "{customer_message}"),
])

url_v2 = client.push_prompt(
    prompt_name,
    object=customer_support_v2,
    description="에스컬레이션 가이드라인이 추가된 v2",
    is_public=False,
)
print(f"버전 2 업로드 완료: {url_v2}")
```

### 예제 2: 허브에서 프롬프트 불러오기

```python
# pull_prompt.py
import os
from dotenv import load_dotenv
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from datetime import date
from pydantic import SecretStr

load_dotenv()

client = Client()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# 최신 버전 불러오기 (개발 환경)
prompt_latest = client.pull_prompt("customer-support-assistant")
print(f"불러온 프롬프트 타입: {type(prompt_latest)}")
print(f"입력 변수: {prompt_latest.input_variables}")

# 체인 구성
chain = prompt_latest | llm | StrOutputParser()

# 실행
result = chain.invoke({
    "company_name": "테크코리아",
    "current_date": date.today().isoformat(),
    "department": "기술지원팀",
    "conversation_history": [],  # 빈 대화 이력
    "customer_message": "주문한 제품이 일주일이 지나도 배송이 안 됩니다.",
})

print("\n응답:")
print(result)


# 특정 커밋 버전으로 고정 (프로덕션 환경 권장)
# 커밋 해시는 LangSmith UI 또는 push_prompt 반환값에서 확인
# prompt_prod = client.pull_prompt("customer-support-assistant:abc12345")
```

### 예제 3: 공개 허브 프롬프트 활용

```python
# public_hub.py
import os
from dotenv import load_dotenv
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

load_dotenv()

client = Client()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# LangChain 팀이 공개한 RAG 프롬프트 활용
# https://smith.langchain.com/hub/langchain-ai/rag-prompt
try:
    rag_prompt = client.pull_prompt("langchain-ai/rag-prompt")
    print("공개 RAG 프롬프트 불러오기 성공")
    print(f"입력 변수: {rag_prompt.input_variables}")

    # RAG 체인에 바로 활용
    rag_chain = rag_prompt | llm | StrOutputParser()

    result = rag_chain.invoke({
        "context": """LangSmith는 LangChain 팀이 만든 LLM 애플리케이션 관측 플랫폼입니다.
        개발자가 LLM 앱의 성능을 추적하고 평가할 수 있게 돕습니다.""",
        "question": "LangSmith는 무엇인가요?",
    })
    print(f"\n응답: {result}")

except Exception as e:
    print(f"공개 프롬프트 로드 실패: {e}")
    print("LangSmith 계정으로 로그인이 필요할 수 있습니다.")
```

### 예제 4: 프롬프트 버전 관리 전략

```python
# version_management.py
import os
from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

client = Client()


def deploy_prompt_version(
    prompt_name: str,
    prompt: ChatPromptTemplate,
    environment: str,
    description: str,
) -> str:
    """프롬프트를 업로드하고 환경별 태그를 설정합니다."""
    # 1. 새 버전 업로드
    url = client.push_prompt(
        prompt_name,
        object=prompt,
        description=f"[{environment}] {description}",
        is_public=False,
    )

    print(f"프롬프트 '{prompt_name}' 업로드 완료")
    print(f"환경: {environment} | URL: {url}")
    return url


def get_prompt_for_env(prompt_name: str, environment: str) -> ChatPromptTemplate:
    """환경에 맞는 프롬프트 버전을 불러옵니다."""
    env_mapping = {
        "development": "latest",
        "staging": "staging",
        "production": "production",
    }

    version_tag = env_mapping.get(environment, "latest")

    try:
        # 태그로 특정 버전 불러오기
        # 태그가 없으면 latest 사용
        prompt = client.pull_prompt(f"{prompt_name}:{version_tag}")
    except Exception:
        prompt = client.pull_prompt(prompt_name)

    print(f"프롬프트 '{prompt_name}:{version_tag}' 로드 완료")
    return prompt


# 현재 환경 결정
environment = os.environ.get("APP_ENV", "development")
print(f"현재 환경: {environment}")

# 환경에 맞는 프롬프트 로드
try:
    prompt = get_prompt_for_env("customer-support-assistant", environment)
    print(f"입력 변수: {prompt.input_variables}")
except Exception as e:
    # 허브에 없는 경우 로컬 fallback
    print(f"허브 로드 실패, 로컬 프롬프트 사용: {e}")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절한 고객 지원 담당자입니다."),
        ("human", "{customer_message}"),
    ])
```

### 예제 5: 프롬프트와 평가 연계

```python
# prompt_with_eval.py
import os
from dotenv import load_dotenv
from langsmith import Client, evaluate
from langsmith.schemas import Run, Example
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import SecretStr

load_dotenv()

client = Client()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


def make_chain_from_hub(prompt_name: str):
    """허브에서 프롬프트를 불러와 체인을 생성합니다."""
    prompt = client.pull_prompt(prompt_name)
    return prompt | llm | StrOutputParser()


def length_evaluator(run: Run, example: Example) -> dict:
    """답변 길이가 적절한지 평가합니다."""
    answer = run.outputs.get("answer", "")
    return {
        "key": "answer_length",
        "score": 1.0 if 50 <= len(answer) <= 500 else 0.0,
    }


# 허브 프롬프트 두 버전 비교 실험
# (실제로는 push_prompt로 두 버전을 먼저 업로드해야 합니다)
try:
    chain_v1 = make_chain_from_hub("customer-support-assistant")

    def target_v1(inputs: dict) -> dict:
        return {"answer": chain_v1.invoke(inputs)}

    results = evaluate(
        target_v1,
        data="파이썬-QA-기본",
        evaluators=[length_evaluator],
        experiment_prefix="hub-prompt-eval",
        description="허브 프롬프트 평가",
    )

    df = results.to_pandas()
    print(f"평균 길이 점수: {df['feedback.answer_length'].mean():.3f}")

except Exception as e:
    print(f"실험 실행 실패: {e}")
    print("데이터셋과 프롬프트가 존재하는지 확인하세요.")
```

---

## ✏️ 실습 과제

### 과제 1: 나만의 프롬프트 허브 구축 (필수)

다음 세 가지 프롬프트를 Prompt Hub에 업로드합니다:
1. **요약 프롬프트**: 긴 텍스트를 3줄로 요약
2. **번역 프롬프트**: 한국어 ↔ 영어 번역
3. **분류 프롬프트**: 텍스트를 카테고리로 분류

각각 두 버전을 만들고 차이점을 문서화하세요.

### 과제 2: 환경별 프롬프트 관리 (중급)

`APP_ENV` 환경변수(`development`, `production`)에 따라 다른 버전의 프롬프트를 로드하는 시스템을 구현합니다:
- `development`: 디버깅 정보를 포함한 상세 프롬프트
- `production`: 간결하고 최적화된 프롬프트

### 과제 3: 공개 허브 프롬프트 커스터마이징 (심화)

LangSmith 공개 허브에서 에이전트 관련 프롬프트를 찾아 내 조직의 요구사항에 맞게 수정하고 내 허브에 재업로드합니다.

---

## ⚠️ 흔한 함정

### 1. 프롬프트 입력 변수 불일치

```python
# 오류 발생 예시
prompt = client.pull_prompt("my-prompt")
# 프롬프트가 {company_name}을 필요로 한다면:
result = chain.invoke({"question": "..."})
# KeyError: 'company_name'

# 해결: 입력 변수 확인 후 모두 제공
print(prompt.input_variables)  # 먼저 확인
result = chain.invoke({
    "company_name": "테크코리아",
    "question": "...",
})
```

### 2. 네트워크 장애 대비

```python
def load_prompt_with_fallback(prompt_name: str, fallback_template: str):
    """허브 접근 실패 시 로컬 폴백을 사용합니다."""
    from langchain_core.prompts import ChatPromptTemplate
    try:
        return client.pull_prompt(prompt_name)
    except Exception as e:
        print(f"허브 로드 실패: {e}, 로컬 폴백 사용")
        return ChatPromptTemplate.from_messages([
            ("human", fallback_template),
        ])
```

### 3. 프롬프트 캐싱

매 요청마다 허브에서 프롬프트를 불러오면 레이턴시가 증가합니다.
애플리케이션 시작 시 한 번만 불러와 캐싱하세요:

```python
# 앱 시작 시 한 번만 로드
from functools import lru_cache

@lru_cache(maxsize=10)
def get_cached_prompt(prompt_name: str):
    """프롬프트를 캐싱합니다 (같은 이름/버전은 재사용)."""
    return client.pull_prompt(prompt_name)

# 이후 호출은 캐시에서 반환
prompt = get_cached_prompt("my-prompt:production")
```

---

## ✅ 셀프 체크

- [ ] `client.push_prompt()`로 ChatPromptTemplate을 허브에 업로드했습니다.
- [ ] `client.pull_prompt()`로 허브에서 프롬프트를 불러와 체인에 사용했습니다.
- [ ] 동일 프롬프트의 두 번째 버전을 업로드하고 커밋이 달라짐을 확인했습니다.
- [ ] 커밋 해시나 태그로 특정 버전을 고정하는 방법을 이해합니다.
- [ ] 코드와 프롬프트 분리의 장점을 3가지 이상 설명할 수 있습니다.
- [ ] 허브 로드 실패 시 로컬 폴백 로직을 구현했습니다.
- [ ] `lru_cache`를 활용해 프롬프트 반복 로드를 방지했습니다.

---

## 🔗 참고 자료

- [LangSmith Prompt Hub](https://docs.smith.langchain.com/prompt_hub)
- [pull_prompt / push_prompt API](https://docs.smith.langchain.com/reference/python/client)
- [LangChain 공개 허브](https://smith.langchain.com/hub)
- [ChatPromptTemplate 문서](https://python.langchain.com/docs/concepts/prompt_templates/)

> **API 변동 안내**: Prompt Hub API는 LangSmith 버전에 따라 달라질 수 있습니다. `hub.pull()` 방식(구 API)에서 `client.pull_prompt()` 방식으로 변경된 적이 있으니, 사용 버전에 맞는 [공식 문서](https://docs.smith.langchain.com/prompt_hub)를 확인하세요.

---

← [Phase 36: LangSmith 평가](36-langsmith-evaluation.md) | [Phase 38: 테스트 전략](38-testing-strategies.md) →
