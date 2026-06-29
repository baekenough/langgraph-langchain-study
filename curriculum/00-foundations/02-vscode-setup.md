# Phase 02: VSCode 개발 환경

> **예상 소요시간**: 30분
> **난이도**: ★☆☆☆☆
> **선행 페이즈**: [Phase 01: UV로 Python 환경 구성](./01-uv-python-setup.md)

---

## 🎯 학습 목표

- LangChain/LangGraph 개발에 최적화된 VSCode 환경을 구성한다
- Python 인터프리터를 UV가 만든 `.venv`로 정확히 설정한다
- Ruff로 코드 품질을 자동으로 유지한다
- Jupyter Notebook으로 인터랙티브하게 실험하고, 디버거로 코드를 추적한다

---

## 📚 핵심 개념

### 왜 VSCode인가?

VSCode는 Python 개발에서 가장 널리 쓰이는 에디터입니다. LangChain/LangGraph 개발에서 VSCode가 특히 유용한 이유:

- **Jupyter Notebook 지원**: 코드 셀을 하나씩 실행하며 LLM 응답을 확인 가능
- **디버거**: LangGraph 노드 실행 흐름을 중단점으로 추적 가능
- **Ruff 통합**: 빠른 Python 린터/포맷터로 코드 품질 자동화
- **`.env` 파일 지원**: API 키 파일에 자동 색상 강조

### 필수 확장 프로그램

| 확장 | ID | 역할 |
|------|----|------|
| Python | `ms-python.python` | Python 언어 지원, 인터프리터 선택 |
| Pylance | `ms-python.vscode-pylance` | 타입 검사, 자동완성 |
| Jupyter | `ms-toolsai.jupyter` | `.ipynb` 파일 실행 |
| Ruff | `charliermarsh.ruff` | 빠른 린터 + 포맷터 |

### 핵심 설정 파일

**`.vscode/settings.json`**: 프로젝트별 VSCode 설정. 팀원 모두가 동일한 환경을 쓰도록 Git에 커밋합니다.

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll.ruff": "explicit",
      "source.organizeImports.ruff": "explicit"
    }
  },
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"]
}
```

**`.vscode/extensions.json`**: 팀원에게 설치를 권장하는 확장 목록.

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-toolsai.jupyter",
    "charliermarsh.ruff"
  ]
}
```

**`.vscode/launch.json`**: 디버그 구성.

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "justMyCode": true,
      "envFile": "${workspaceFolder}/.env"
    }
  ]
}
```

---

## 💻 코드 예제

### 인터프리터 설정하기

1. VSCode에서 프로젝트 폴더를 엽니다: `code /Users/sangyi/workspace/study/langraph_langchain`

2. Python 인터프리터를 설정합니다:
   - `Cmd+Shift+P` (macOS) / `Ctrl+Shift+P` (Windows)
   - `Python: Select Interpreter` 입력
   - `.venv/bin/python (Recommended)` 선택

3. 오른쪽 하단에 Python 버전이 표시되는지 확인합니다.

### Jupyter Notebook으로 첫 실험

`experiments/` 폴더에 새 파일 `hello_langchain.ipynb`를 만들어 실험해봅니다:

```python
# Cell 1: 환경 확인
import sys
print(f"Python: {sys.version}")

import langchain
import langgraph
print(f"LangChain: {langchain.__version__}")
print(f"LangGraph: {langgraph.version.__version__}")
```

```python
# Cell 2: 환경변수 로드 확인
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일 로드

# 키가 설정되었는지 확인 (값은 출력하지 않음)
openai_key = os.getenv("OPENAI_API_KEY", "")
print("OPENAI_API_KEY 설정됨:", bool(openai_key))
```

### 디버거 사용하기

아래 코드를 `debug_example.py`로 저장하고 중단점을 설정해봅니다:

```python
# debug_example.py
from langchain_core.messages import HumanMessage, AIMessage


def process_messages(messages: list) -> str:
    """메시지 목록을 처리하는 예제 함수."""
    results = []

    for msg in messages:  # <- 이 줄에 중단점 설정 (F9)
        if isinstance(msg, HumanMessage):
            results.append(f"Human: {msg.content}")
        elif isinstance(msg, AIMessage):
            results.append(f"AI: {msg.content}")

    return "\n".join(results)


if __name__ == "__main__":
    messages = [
        HumanMessage(content="안녕하세요!"),
        AIMessage(content="안녕하세요! 무엇을 도와드릴까요?"),
        HumanMessage(content="LangChain이 뭔가요?"),
    ]

    result = process_messages(messages)
    print(result)
```

**디버깅 순서:**
1. `for msg in messages:` 줄 왼쪽 번호 클릭 → 빨간 점(중단점) 설정
2. `F5` 또는 Run → Start Debugging
3. `F10`(다음 줄), `F11`(함수 진입), `F5`(계속) 로 추적
4. 왼쪽 "Variables" 패널에서 `msg` 값 확인

### 통합 터미널에서 UV 사용

VSCode 통합 터미널(`` Ctrl+` ``)에서 UV 명령어를 직접 실행합니다:

```bash
# 스크립트 실행
uv run python debug_example.py

# Jupyter 서버 시작
uv run jupyter notebook

# 테스트 실행
uv run pytest tests/ -v

# 코드 포맷
uv run ruff format .
uv run ruff check .
```

---

## ✏️ 실습 과제

1. **확장 설치**: `.vscode/extensions.json`에 있는 4개 확장을 모두 설치하세요.
   (`Cmd+Shift+X`에서 각 ID로 검색)

2. **인터프리터 확인**: 오른쪽 하단 Python 버전이 `.venv` 경로를 가리키는지 확인하세요.

3. **Jupyter 실험**: `experiments/hello_langchain.ipynb`를 만들고 위 코드를 실행해보세요.

4. **디버거 실습**: `debug_example.py`를 만들고 중단점을 걸어 디버거로 실행해보세요.
   `msg` 변수의 타입이 단계마다 어떻게 바뀌는지 관찰하세요.

---

## ⚠️ 흔한 함정

- **인터프리터가 `.venv`가 아닌 시스템 Python**: 오른쪽 하단에서 반드시 `.venv/bin/python`을 선택하세요. 시스템 Python에는 설치된 패키지가 없습니다.

- **Pylance 타입 오류 무시**: LangChain의 동적 타입 특성상 Pylance가 경고를 많이 냅니다. `# type: ignore` 주석이나 `pyrightconfig.json`으로 조정할 수 있습니다.

- **Jupyter `.ipynb` 파일 커밋**: 노트북 파일은 출력 데이터(API 응답, 임베딩 등)를 포함할 수 있습니다. `.gitignore`에 `.ipynb_checkpoints`를 반드시 추가하세요.

- **`.env` 파일 공유 금지**: `.env`에는 API 키가 있으므로 절대 Git에 커밋하지 마세요. `.env.example`만 커밋합니다.

---

## ✅ 셀프 체크

- [ ] 4개 필수 확장이 모두 설치되어 있다
- [ ] Python 인터프리터가 `.venv/bin/python`으로 설정되어 있다
- [ ] 저장 시 Ruff가 자동으로 코드를 포맷한다
- [ ] Jupyter Notebook에서 Python 셀을 실행할 수 있다
- [ ] 디버거로 중단점을 설정하고 변수를 검사할 수 있다

---

## 🔗 참고 자료

- [VSCode Python 확장 공식 문서](https://code.visualstudio.com/docs/python/python-tutorial)
- [Ruff 공식 문서](https://docs.astral.sh/ruff/)
- [VSCode Jupyter 사용 가이드](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- [VSCode 디버거 가이드](https://code.visualstudio.com/docs/python/debugging)

---

← 이전: [Phase 01: UV로 Python 환경 구성](./01-uv-python-setup.md) | 다음: [Phase 03: 모델 제공자와 API 키](./03-models-and-keys.md) →
