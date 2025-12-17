# OpenCode 의존성 상세 분석 결과

## 테스트 날짜: 2025-12-17

## 📌 핵심 발견사항

### Custom Provider 사용 가능 ✅
- 사용자 확인: 내부망 AI API는 Custom Provider로 사용 가능
- **AI 모델 API 연결은 문제가 아님**

### 남은 의존성 문제 ❌

OpenCode는 실행 시 다음 npm 패키지들을 **동적으로 설치**하려고 시도합니다:

## 1. 필수 npm 패키지 목록

### 공통 패키지
```
@opencode-ai/plugin@1.0.164
  위치: /root/.config/opencode/node_modules/
  용도: OpenCode 플러그인 시스템
  상태: npm으로 다운로드 가능, Bun으로 설치 시 401 에러
```

### Provider별 인증 패키지 (동적 로드)
```
opencode-copilot-auth@0.0.9
  위치: /root/.cache/opencode/node_modules/
  용도: GitHub Copilot 인증
  의존성: zod
  상태: npm으로 설치 가능

opencode-anthropic-auth@0.0.5
  위치: /root/.cache/opencode/node_modules/
  용도: Anthropic Claude 인증
  상태: npm으로 설치 가능

[기타 provider 인증 패키지들]
  - opencode-openai-codex-auth
  - opencode-gemini-auth
  - opencode-google-antigravity-auth
  - 등등...
```

## 2. 문제의 핵심: Bun vs npm

### 테스트 결과
```bash
# npm으로 설치 → 성공 ✅
$ npm install @opencode-ai/plugin@1.0.164
$ npm install opencode-copilot-auth@0.0.9
$ npm install opencode-anthropic-auth@0.0.5
→ 모두 정상 설치됨

# OpenCode 내장 Bun으로 설치 → 실패 ❌
OpenCode 실행 시 내부적으로:
$ /path/to/opencode add --force --exact opencode-anthropic-auth@0.0.5
→ GET https://registry.npmjs.org/opencode-anthropic-auth - 401
```

### 401 에러 원인 분석

**가설 1: 인증 차이**
- npm: 일반 공개 레지스트리 접근 (성공)
- OpenCode 내장 Bun: 다른 인증 메커니즘 사용? (401 에러)

**가설 2: 네트워크 환경**
- 이 테스트 환경에서는 일부 호스트 차단됨 (models.dev → 403)
- Bun이 다른 엔드포인트나 프로토콜 사용 가능성

**가설 3: Bun 레지스트리 설정**
- OpenCode가 내부적으로 특정 레지스트리 설정 사용
- .bunfig.toml 생성해도 효과 없음

## 3. 실행 흐름 분석

```
OpenCode 시작
  ↓
1. 설정 파일 로드
   - /root/.config/opencode/config.json
   - /root/.config/opencode/opencode.json
   - /root/.config/opencode/opencode.jsonc
  ↓
2. @opencode-ai/plugin 설치 시도
   - 위치: /root/.config/opencode
   - 상태: 401 에러 (하지만 진행은 계속됨)
  ↓
3. models.dev 갱신 시도
   - 모델 목록 조회
   - 실패해도 진행 가능
  ↓
4. Provider 초기화
   - 사용 가능한 provider들 스캔
   - 각 provider의 auth 플러그인 로드 시도
  ↓
5. Provider별 Auth 플러그인 로드
   - opencode-copilot-auth@0.0.9 로드 시도
   - opencode-anthropic-auth@0.0.5 로드 시도
   - 설치 안 되어 있으면 Bun으로 설치 시도
   - 설치 실패 시 전체 프로세스 중단 ❌
```

## 4. 외부 연결 의존성

### npm 레지스트리
```
https://registry.npmjs.org
  - 패키지 다운로드 (필수)
  - 사내망에서 차단 시 대응 방법:
    * Verdaccio 등 로컬 미러 구축
    * npm 프록시 설정
    * 패키지 사전 다운로드 후 로컬 설치
```

### models.dev (선택적)
```
https://models.dev
  - 모델 정보 조회
  - 실패해도 기본 동작은 가능할 것으로 추정
  - 이 테스트 환경에서 403 Forbidden
```

## 5. 오프라인 구성 전략

### 전략 A: 로컬 npm 레지스트리 (권장)

#### 1단계: 온라인 환경에서 준비
```bash
# 모든 필요한 패키지 다운로드
npm install -g opencode-ai
npm install @opencode-ai/plugin@1.0.164
npm install opencode-copilot-auth@0.0.9
npm install opencode-anthropic-auth@0.0.5
npm install opencode-gemini-auth
# ... 기타 provider auth 패키지들

# 패키지들을 tarball로 추출
npm pack @opencode-ai/plugin
npm pack opencode-copilot-auth
npm pack opencode-anthropic-auth
```

#### 2단계: Verdaccio 로컬 레지스트리 구축
```bash
# 사내망 서버에 Verdaccio 설치
npm install -g verdaccio

# Verdaccio 실행
verdaccio &

# 패키지들 업로드
npm publish opencode-ai-plugin-1.0.164.tgz --registry http://내부서버:4873/
npm publish opencode-copilot-auth-0.0.9.tgz --registry http://내부서버:4873/
npm publish opencode-anthropic-auth-0.0.5.tgz --registry http://내부서버:4873/
```

#### 3단계: OpenCode에서 로컬 레지스트리 사용
```bash
# npm 레지스트리 변경
npm config set registry http://내부서버:4873/

# Bun 레지스트리 변경 (시도)
cat > ~/.bunfig.toml << EOF
[install]
registry = "http://내부서버:4873/"
EOF
```

**문제점:**
- OpenCode 내장 Bun이 .bunfig.toml을 읽는지 확인 필요
- OpenCode 바이너리 내부 설정 변경 불가능

### 전략 B: 패키지 사전 설치 (부분적 해결)

```bash
# 온라인 환경에서
npm install -g opencode-ai
npm install @opencode-ai/plugin@1.0.164
npm install opencode-copilot-auth@0.0.9
npm install opencode-anthropic-auth@0.0.5

# 필요한 디렉토리 복사
tar czf opencode-offline.tar.gz \
  /usr/lib/node_modules/opencode-ai/ \
  ~/.config/opencode/ \
  ~/.cache/opencode/ \
  ~/.local/share/opencode/

# 오프라인 환경으로 전송 및 압축 해제
# ... 설치 ...
```

**테스트 결과:**
- 패키지를 수동으로 node_modules에 복사해도
- OpenCode가 `--force` 플래그로 재설치 시도
- Bun의 401 에러로 실패

### 전략 C: Custom Provider만 사용 (이상적)

**개념:**
사용자가 Custom Provider를 사용한다면, 기본 provider(Anthropic, OpenAI 등)의 auth 패키지가 불필요할 수 있음.

**필요한 것:**
1. OpenCode 설정에서 기본 provider들을 비활성화하는 방법
2. Custom Provider만 등록하고 사용

**현재 상태:**
- config.json에 `providers` 키는 인식되지 않음
- OpenCode 문서나 예제 확인 필요
- 기본 provider 비활성화 방법 불명확

## 6. 검증된 해결책 (현재까지)

### ❌ 실패한 방법들

1. **패키지 사전 복사**
   - node_modules에 직접 복사
   - package.json 업데이트
   - 결과: OpenCode가 --force로 재설치 시도 → 실패

2. **Bun 설정 변경**
   - ~/.bunfig.toml 생성
   - 레지스트리 URL 지정
   - 결과: OpenCode 내장 Bun이 무시하는 것으로 보임

3. **Provider 비활성화**
   - config.json에 providers: [] 설정
   - 결과: "Unrecognized key" 에러

### ⚠️ 부분적 성공

- npm으로 패키지 설치는 가능
- 문제는 OpenCode 런타임의 Bun이 재설치를 시도한다는 것

## 7. 결론 및 권장사항

### 현재 상황
**OpenCode는 Custom Provider를 지원하지만, 실행 시 여전히 기본 provider들의 auth 패키지를 로드하려고 시도합니다.**

### 필요한 추가 조사

1. **OpenCode 설정 문서 확인**
   - Custom Provider 전용 모드가 있는지
   - 기본 provider 비활성화 방법
   - opencode.json/opencode.jsonc 설정 옵션

2. **OpenCode 소스코드 분석** (가능하다면)
   - Bun 레지스트리 설정 위치
   - Provider 로딩 메커니즘
   - --force 플래그를 사용하는 이유

3. **로컬 레지스트리 테스트**
   - Verdaccio 구축
   - OpenCode가 로컬 레지스트리 사용 가능한지 검증
   - Bun의 레지스트리 설정 우선순위

### 사내망 사용을 위한 최선의 방법

#### 옵션 1: 제한적 네트워크 허용 (가장 현실적)
```
화이트리스트 도메인:
- registry.npmjs.org (필수)
- 내부 AI API 엔드포인트 (Custom Provider용)
```

#### 옵션 2: 로컬 npm 레지스트리 + OpenCode 설정 수정
```
1. Verdaccio 구축
2. 모든 필요한 패키지 업로드
3. OpenCode 소스/바이너리에서 레지스트리 URL 변경 (고급)
4. 또는 네트워크 레벨 리다이렉션 (registry.npmjs.org → 내부 서버)
```

#### 옵션 3: OpenCode 개발팀 문의
```
- 엔터프라이즈/오프라인 버전 존재 여부
- Custom Provider 전용 모드 설정 방법
- 오프라인 구성 가이드 요청
```

#### 옵션 4: 대안 도구 (완전 오프라인 필요 시)
```
- Continue.dev + Ollama
- Tabby (자체 호스팅)
- Aider (로컬 LLM 지원)
```

## 8. 다음 단계

### 즉시 시도 가능한 것
1. OpenCode 공식 문서/GitHub에서 오프라인 구성 가이드 찾기
2. opencode.json/opencode.jsonc 설정 옵션 조사
3. Verdaccio 로컬 레지스트리 테스트

### 확인이 필요한 것
1. Custom Provider 사용 시 기본 provider auth가 정말 필요한지
2. OpenCode의 `--no-plugins` 같은 플래그가 있는지
3. 환경 변수로 레지스트리 URL 오버라이드 가능한지

### 장기적 해결책
1. OpenCode 팀과 협의하여 오프라인 버전/설정 확보
2. 또는 사내망 정책 조정하여 npm 레지스트리 접근 허용
3. 완전 오프라인이 필수라면 대안 도구로 전환

## 요약

**AI API는 해결됨 (Custom Provider) ✅**

**남은 문제: npm 패키지 의존성 ❌**
- @opencode-ai/plugin
- provider별 auth 패키지들 (copilot, anthropic, 등)
- 이 패키지들은 공개되어 있고 npm으로 다운로드 가능
- 하지만 OpenCode 내장 Bun이 설치 시도 시 401 에러
- 로컬에 미리 설치해도 --force로 재설치 시도

**현실적 해결책:**
1. npm 레지스트리 접근 허용 (가장 간단)
2. Verdaccio 로컬 미러 + 네트워크 리다이렉션 (기술적)
3. OpenCode 팀 지원 요청 (공식적)
4. 대안 도구 검토 (완전 오프라인 필수 시)
