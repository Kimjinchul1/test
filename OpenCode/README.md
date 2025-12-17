# OpenCode 오프라인 사용 테스트 결과

## 📋 테스트 개요

OpenCode를 설치하고 완전한 오프라인 환경(사내망)에서 사용 가능한지 테스트한 결과입니다.

## ⚠️ 핵심 결론

**OpenCode는 현재 완전한 오프라인 사용이 불가능합니다.**

### 가능한 것 ✅
- 바이너리 파일 복사 및 실행
- `--version`, `--help` 등 기본 명령어
- 파일 탐색 등 로컬 기능

### 불가능한 것 ❌
- AI 코딩 어시스턴트 기능 (핵심 기능!)
- 모델 목록 조회
- 코드 생성/완성/분석
- 실시간 코드 제안

## 📊 설치 정보

```
버전: 1.0.164
전체 크기: 525MB
바이너리 크기: 128-135MB (플랫폼별)
설치 경로: /opt/node22/lib/node_modules/opencode-ai/
```

## 🔍 오프라인 사용 불가 이유

### 1. 런타임 패키지 의존성
OpenCode 실행 시 npm 레지스트리에서 추가 패키지를 동적으로 설치합니다:
- `@opencode-ai/plugin@1.0.164`
- `opencode-copilot-auth@0.0.9`

```
ERROR: GET https://registry.npmjs.org/@opencode-ai%2fplugin - 401
ERROR: GET https://registry.npmjs.org/opencode-copilot-auth - 401
```

### 2. AI API 의존성
OpenCode는 AI 코딩 어시스턴트이므로 외부 AI 서비스 연결이 필수입니다:
- Anthropic Claude API
- OpenAI API
- 기타 AI provider API

### 3. 인증 및 라이선싱
- 일부 npm 패키지는 비공개 (401 Unauthorized)
- 정식 인증/라이선스 필요 가능성

## 📁 생성된 파일

```
OpenCode/
├── README.md                          # 이 파일
├── OFFLINE_SETUP_GUIDE.md             # 상세 오프라인 구성 가이드
├── package-structure.txt              # 패키지 구조 설명
├── test-offline.sh                    # 오프라인 테스트 스크립트
└── opencode-ai-offline/               # 복사된 OpenCode (525MB)
    ├── bin/opencode -> (심볼릭 링크)
    ├── node_modules/
    │   ├── opencode-linux-x64/        # 135MB
    │   ├── opencode-linux-x64-baseline/
    │   ├── opencode-linux-x64-musl/
    │   └── opencode-linux-x64-baseline-musl/
    ├── package.json
    └── postinstall.mjs
```

## 🛠️ 오프라인 환경 구성 옵션

### 옵션 1: 로컬 npm 레지스트리 + 로컬 AI 서버
**난이도: 높음**
- Verdaccio 등 로컬 npm 레지스트리 구축
- Ollama, LM Studio 등 로컬 LLM 서버 구축
- OpenCode 설정을 로컬 엔드포인트로 변경

**문제점:**
- 비공개 패키지 접근 불가 (401 에러)
- OpenCode가 로컬 모델을 지원하는지 불명확
- 상당한 기술적 노력 필요

### 옵션 2: 제한적 네트워크 허용
**난이도: 중간**
- 필수 도메인만 화이트리스트 허용:
  - `registry.npmjs.org` (npm 패키지)
  - AI provider 도메인 (예: `api.anthropic.com`)
  - `models.dev` (모델 정보)

**장점:**
- 가장 현실적인 방법
- OpenCode 본래 기능 사용 가능

**단점:**
- 완전한 오프라인은 아님
- 외부 통신 필요

### 옵션 3: 대안 도구 검토 (권장)
**난이도: 낮음**

완전 오프라인이 필수라면 다음 도구를 고려하세요:

1. **Continue.dev**
   - 로컬 LLM 지원
   - Ollama, LM Studio 연동 가능
   - 완전 오프라인 가능

2. **Tabby**
   - 완전 로컬 실행 가능
   - 자체 모델 또는 로컬 LLM 사용
   - 오픈소스

3. **Cody (Sourcegraph)**
   - 자체 호스팅 가능
   - 엔터프라이즈 버전 제공

## 🧪 테스트 실행 방법

```bash
# 테스트 스크립트 실행
cd OpenCode
./test-offline.sh

# 직접 테스트
./opencode-ai-offline/node_modules/opencode-linux-x64/bin/opencode --version
./opencode-ai-offline/node_modules/opencode-linux-x64/bin/opencode --help
./opencode-ai-offline/node_modules/opencode-linux-x64/bin/opencode models  # 실패 예상
```

## 📝 로그 예시

```
INFO  service=default version=1.0.164 args=["models"]
INFO  service=bun installing package opencode-copilot-auth@0.0.9
ERROR service=bun GET https://registry.npmjs.org/opencode-copilot-auth - 401
ERROR BunInstallFailedError: Command failed with exit code 1
```

## 💡 권장사항

### 사내망에서 AI 코딩 도구가 필요한 경우:

1. **OpenCode 팀에 문의**
   - 엔터프라이즈/온프레미스 버전 문의
   - 오프라인 라이선스 확인
   - https://opencode.ai 참조

2. **완전 로컬 도구 사용**
   - Continue.dev + Ollama
   - Tabby (자체 호스팅)
   - 로컬 LLM 기반 솔루션

3. **제한적 네트워크 허용**
   - 필수 도메인만 화이트리스트
   - 프록시 서버 구성
   - 보안 정책 조정

## 🔗 참고 링크

- OpenCode 공식: https://opencode.ai
- Continue.dev: https://continue.dev
- Tabby: https://github.com/TabbyML/tabby
- Ollama: https://ollama.ai

## 📞 추가 지원

더 자세한 내용은 `OFFLINE_SETUP_GUIDE.md` 파일을 참조하세요.
