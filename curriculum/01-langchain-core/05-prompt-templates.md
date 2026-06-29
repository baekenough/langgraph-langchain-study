# Phase 05: 프롬프트 템플릿

> 예상 소요시간: 60분 | 난이도: ★★☆☆☆ | 선행 페이즈: [04-chat-models-messages](04-chat-models-messages.md)

---

## 🎯 학습 목표

- `ChatPromptTemplate.from_messages()`로 재사용 가능한 프롬프트를 만들 수 있습니다.
- `{변수명}` 문법으로 동적 변수를 주입할 수 있습니다.
- `MessagesPlaceholder`를 사용하여 대화 이력을 삽입할 수 있습니다.
- Few-shot 예제를 프롬프트에 포함하는 방법을 이해합니다.
- `partial()`로 일부 변수를 미리 고정할 수 있습니다.

---

## 📚 핵심 개념

### 1. 왜 프롬프트 템플릿인가

하드코딩된 문자열 대신 템플릿을 사용하면 다음을 얻습니다.

- **재사용성**: 동일 구조의 프롬프트를 여러 곳에서 활용
- **유지보수성**: 프롬프트 변경이 한 곳에서 이루어짐
- **안전성**: 변수 치환 시 이스케이프 처리 자동화
- **LCEL 통합**: `|` 연산자로 모델과 연결 (Phase 07)

### 2. ChatPromptTemplate 구조

`ChatPromptTemplate.from_messages()`는 `(role, template_string)` 튜플 리스트를 받습니다.

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role}입니다."),
    ("human", "{question}"),
])
```

역할 문자열: `"system"`, `"human"`, `"ai"` 세 가지를 지원합니다.

### 3. MessagesPlaceholder

대화 이력처럼 **동적으로 결정되는 메시지 리스트**를 삽입할 때 사용합니다.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "당신은 도움이 되는 어시스턴트입니다."),
    MessagesPlaceholder("history"),   # 대화 이력이 여기에 삽입됨
    ("human", "{question}"),
])
```

### 4. partial — 변수 미리 고정

`partial()`을 사용하면 일부 변수를 미리 채운 새 템플릿을 반환합니다.

```python
review_template = template.partial(role="코드 리뷰어")
# 이제 role은 고정, question만 채우면 됨
```

---

## 💻 코드 예제

### 예제 1: 기본 ChatPromptTemplate

```python
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# 템플릿 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {language} 전문가입니다. 명확하고 간결하게 답변합니다."),
    ("human", "{question}"),
])

# 변수를 채워 메시지 생성
messages = prompt.invoke({
    "language": "Python",
    "question": "리스트 컴프리헨션과 map()의 성능 차이는?",
})
print(messages)  # ChatPromptValue 객체

# 모델에 전달
response = llm.invoke(messages)
print(response.content)
```

### 예제 2: LCEL로 연결 (프리뷰)

```python
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {domain} 전문가입니다."),
    ("human", "{question}"),
])

# prompt → llm → parser 체인 구성 (LCEL, Phase 07에서 상세 설명)
chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "domain": "머신러닝",
    "question": "과적합을 방지하는 방법 세 가지를 알려주세요.",
})
print(result)  # 순수 문자열 반환
```

### 예제 3: MessagesPlaceholder로 대화 이력 관리

```python
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 친절한 Python 튜터입니다."),
    MessagesPlaceholder("history"),  # 대화 이력 삽입 위치
    ("human", "{question}"),
])

# 대화 이력 누적
history = []

def chat(question: str) -> str:
    """대화 이력을 유지하며 모델에 질문합니다."""
    messages = prompt.invoke({"history": history, "question": question})
    response = llm.invoke(messages)
    # 이력 업데이트
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=response.content))
    return response.content

print(chat("제너레이터란 무엇인가요?"))
print("---")
print(chat("방금 설명한 것과 이터레이터의 차이점은?"))  # 이전 맥락 참조 가능
```

### 예제 4: Few-shot 프롬프트

```python
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# few-shot 예시를 messages 리스트에 직접 포함
prompt = ChatPromptTemplate.from_messages([
    ("system", "코드 오류를 분석하고 수정 방법을 제시합니다."),
    # Few-shot 예시 1
    ("human", "오류: TypeError: unsupported operand type(s) for +: 'int' and 'str'"),
    ("ai", "원인: 정수와 문자열을 + 연산자로 직접 더했습니다.\n수정: str(num) + text 또는 f'{num}{text}' 형태로 변환하세요."),
    # Few-shot 예시 2
    ("human", "오류: IndexError: list index out of range"),
    ("ai", "원인: 리스트 범위를 초과한 인덱스에 접근했습니다.\n수정: 접근 전 len(lst)로 크기를 확인하거나 try-except로 처리하세요."),
    # 실제 사용자 입력
    ("human", "{error_message}"),
])

result = llm.invoke(
    prompt.invoke({"error_message": "오류: KeyError: 'username'"})
)
print(result.content)
```

### 예제 5: partial로 변수 미리 고정

```python
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=SecretStr(os.environ["OPENROUTER_API_KEY"]),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

# 공통 템플릿
base_prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {language} {role}입니다. {style}로 답변합니다."),
    ("human", "{question}"),
])

# partial로 특화된 프롬프트 파생
python_reviewer = base_prompt.partial(
    language="Python",
    role="코드 리뷰어",
    style="건설적이고 구체적인 피드백",
)

python_tutor = base_prompt.partial(
    language="Python",
    role="초보자 튜터",
    style="쉬운 비유와 예제 위주",
)

reviewer_chain = python_reviewer | llm | StrOutputParser()
tutor_chain = python_tutor | llm | StrOutputParser()

question = "이 코드를 어떻게 개선할 수 있나요? `result = []; [result.append(x*2) for x in range(10)]`"

print("=== 리뷰어 관점 ===")
print(reviewer_chain.invoke({"question": question}))

print("\n=== 튜터 관점 ===")
print(tutor_chain.invoke({"question": question}))
```

### 예제 6: 템플릿 변수 목록 확인

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 {role}입니다. 언어: {language}."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

# 필요한 변수 목록 확인 — 디버깅 시 유용
print("입력 변수:", prompt.input_variables)
# ['role', 'language', 'question']

print("모든 변수 (placeholder 포함):", list(prompt.input_schema.model_fields.keys()))
```

---

## ✏️ 실습 과제

### 과제 1: 다국어 번역 템플릿

`source_lang`, `target_lang`, `text` 세 변수를 받는 번역 템플릿을 만들고, `partial()`로 `target_lang="Korean"`을 고정한 번역기를 구현하세요.

### 과제 2: 대화형 학습 도우미

`MessagesPlaceholder`를 사용하여 3턴 이상의 대화 이력을 유지하는 학습 도우미를 구현하세요. 각 턴이 이전 맥락을 참조하는지 확인하세요.

### 과제 3: Few-shot 감정 분석기

긍정/부정 분류 few-shot 예시 3쌍을 포함한 프롬프트를 만들어 새 문장의 감정을 분류하세요.

### 과제 4: 도메인별 전문가 팩토리

`partial()`을 활용하여 Python, JavaScript, SQL 각각의 전문가 체인을 생성하는 함수를 작성하세요.

---

## ⚠️ 흔한 함정

**1. 변수명 오타 — 즉시 KeyError 발생**

```python
prompt = ChatPromptTemplate.from_messages([("human", "{quesiton}")])  # 오타!
prompt.invoke({"question": "..."})  # KeyError: 'quesiton'
```

`prompt.input_variables`로 변수 목록을 먼저 확인하세요.

**2. MessagesPlaceholder 변수명 불일치**

```python
# 선언
MessagesPlaceholder("chat_history")

# 호출 시 — 같은 이름이어야 함
prompt.invoke({"chat_history": [...], ...})
# "history"로 전달하면 KeyError 발생
```

**3. PromptTemplate과 ChatPromptTemplate 혼동**

`PromptTemplate`은 문자열 → 문자열이고, `ChatPromptTemplate`은 문자열 → 메시지 리스트입니다.
Chat Model에는 반드시 `ChatPromptTemplate`을 사용하세요.

**4. API 변동 주의**

> LangChain은 빠르게 발전합니다. 프롬프트 관련 최신 사양은 [공식 문서](https://python.langchain.com/docs/concepts/prompt_templates/)를 확인하세요.

---

## ✅ 셀프 체크

- [ ] `ChatPromptTemplate.from_messages()`로 시스템+사용자 메시지 템플릿을 만들 수 있다.
- [ ] `{변수명}` 문법으로 동적 변수를 주입할 수 있다.
- [ ] `MessagesPlaceholder`로 대화 이력을 삽입할 수 있다.
- [ ] `partial()`로 특화된 프롬프트를 파생시킬 수 있다.
- [ ] Few-shot 예제를 messages 리스트에 포함할 수 있다.

---

## 🔗 참고 자료

- [LangChain Prompt Templates 개요](https://python.langchain.com/docs/concepts/prompt_templates/)
- [ChatPromptTemplate API 레퍼런스](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html)
- [MessagesPlaceholder API 레퍼런스](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.MessagesPlaceholder.html)

---

← [Phase 04: Chat Models와 Messages](04-chat-models-messages.md) | [Phase 06: 출력 파싱](06-output-parsing.md) →
