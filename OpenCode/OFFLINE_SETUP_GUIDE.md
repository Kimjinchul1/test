# OpenCode 오프라인 환경 구성 가이드

## 테스트 결과 요약

### ✅ 설치 확인
- OpenCode 버전: 1.0.164
- 설치 크기: 약 525MB
- 바이너리 크기: 128-135MB (플랫폼별)

### ❌ 오프라인 사용 제한사항
**현재 상태로는 완전한 오프라인 사용이 불가능합니다.**

#### 문제점:
1. **런타임 패키지 설치 필요**
   - OpenCode 실행 시 다음 패키지를 npm 레지스트리에서 다운로드 시도:
     - `@opencode-ai/plugin@1.0.164`
     - `opencode-copilot-auth@0.0.9`

2. **네트워크 의존성**
   - npm 레지스트리 연결 필요 (일부 패키지는 비공개 - 401 에러)
   - AI 모델 제공업체(provider)에 대한 인터넷 연결 필요
   - models.dev에서 모델 목록 갱신 필요

3. **AI 서비스 의존성**
   - OpenCode는 AI 코딩 어시스턴트 도구
   - 외부 AI API (Anthropic, OpenAI 등) 연결 필요
   - API 키 및 인증 필요

## 오프라인 환경 구성 방법

### 방법 1: 로컬 npm 레지스트리 구성 (권장)

#### 1단계: 온라인 환경에서 준비
```bash
# 1. OpenCode 설치
npm i -g opencode-ai

# 2. 한 번 실행하여 필요한 모든 패키지 설치
opencode --version
opencode models
opencode auth list

# 3. npm 캐시 확인
npm cache verify
```

#### 2단계: 필요한 파일/디렉토리 수집
```bash
# OpenCode 설치 디렉토리
/opt/node22/lib/node_modules/opencode-ai/  (약 525MB)

# npm 캐시
~/.npm/  (설치된 모든 패키지 캐시)

# OpenCode 설정 및 캐시
~/.config/opencode/
~/.cache/opencode/
~/.local/share/opencode/
```

#### 3단계: 사내망 환경 구성
```bash
# Verdaccio 등 로컬 npm 레지스트리 설치
npm i -g verdaccio

# Verdaccio 실행
verdaccio

# npm 레지스트리 설정 변경
npm set registry http://your-local-registry:4873/
```

#### 4단계: 패키지 업로드
```bash
# 수집한 패키지들을 로컬 레지스트리에 업로드
npm publish --registry http://your-local-registry:4873/
```

### 방법 2: 직접 파일 복사 (제한적)

#### 필수 파일 목록:
```
사내망_서버/
├── opencode-ai/                      # 메인 패키지 (525MB)
│   ├── bin/
│   ├── node_modules/
│   │   ├── opencode-linux-x64/      # 실제 바이너리
│   │   ├── opencode-linux-x64-baseline/
│   │   ├── opencode-linux-x64-musl/
│   │   └── opencode-linux-x64-baseline-musl/
│   ├── package.json
│   └── postinstall.mjs
├── .npm/                             # npm 캐시
├── .config/opencode/                 # 설정 파일
├── .cache/opencode/                  # 런타임 캐시
└── .local/share/opencode/            # 데이터 저장소
```

#### 설치 스크립트:
```bash
#!/bin/bash
# install-opencode-offline.sh

# 1. 디렉토리 생성
mkdir -p /usr/local/lib/node_modules/opencode-ai
mkdir -p /usr/local/bin

# 2. 파일 복사
cp -r opencode-ai/* /usr/local/lib/node_modules/opencode-ai/

# 3. 실행 파일 링크 생성
ln -sf /usr/local/lib/node_modules/opencode-ai/node_modules/opencode-linux-x64/bin/opencode \
       /usr/local/bin/opencode

# 4. 실행 권한 부여
chmod +x /usr/local/bin/opencode

# 5. 사용자 설정 복사 (선택사항)
# cp -r .config/opencode ~/.config/
# cp -r .cache/opencode ~/.cache/
# cp -r .local/share/opencode ~/.local/share/
```

### 방법 3: 로컬 AI 모델 사용 (대안)

OpenCode는 외부 AI API에 의존하므로, 사내망에서 사용하려면:

1. **로컬 LLM 서버 구축**
   ```bash
   # Ollama, LM Studio 등 로컬 LLM 서버 설치
   # OpenAI 호환 API 엔드포인트 제공
   ```

2. **OpenCode 설정 수정**
   - 커스텀 API 엔드포인트 설정
   - 로컬 모델 사용 설정

## 핵심 제약사항

### 🚨 완전한 오프라인 사용은 다음 이유로 어렵습니다:

1. **런타임 의존성**
   - OpenCode는 실행 시 npm 레지스트리에서 추가 패키지 다운로드
   - 이는 설계상의 제약

2. **AI API 의존성**
   - OpenCode는 AI 코딩 어시스턴트
   - Anthropic Claude, OpenAI 등 외부 AI 서비스 필요
   - 로컬 모델 지원 여부 불명확

3. **인증 및 라이선싱**
   - 일부 패키지는 비공개 (401 에러)
   - 정식 인증/라이선스 필요 가능

## 권장사항

### 완전 오프라인이 필수인 경우:
1. **대안 도구 검토**
   - Continue.dev (로컬 LLM 지원)
   - Cody (자체 호스팅 가능)
   - Tabby (완전 로컬 실행 가능)

2. **OpenCode 개발팀 문의**
   - 오프라인/온프레미스 버전 문의
   - 엔터프라이즈 라이선스 확인

### 제한적 네트워크 환경인 경우:
1. **프록시 서버 구성**
   - npm 레지스트리 프록시
   - AI API 프록시

2. **화이트리스트 방식**
   - 필수 도메인만 허용
   - registry.npmjs.org
   - 사용할 AI API 도메인

## 테스트 로그 분석

```
ERROR 2025-12-17T07:01:21 service=bun
GET https://registry.npmjs.org/@opencode-ai%2fplugin - 401

ERROR 2025-12-17T07:01:21 service=bun
GET https://registry.npmjs.org/opencode-copilot-auth - 401
```

→ OpenCode가 비공개 npm 패키지에 접근 시도
→ 정식 인증 없이는 완전한 기능 사용 불가

## 결론

**OpenCode는 현재 구조상 완전한 오프라인 사용이 어렵습니다.**

- 바이너리 자체는 복사 가능 (525MB)
- 런타임 의존성으로 인해 네트워크 필요
- AI 서비스 연결 필수
- 로컬 npm 레지스트리 + 로컬 AI 서버 구성 시 가능성 있음

사내망에서 AI 코딩 도구가 필요하다면:
1. OpenCode 개발팀에 엔터프라이즈/온프레미스 버전 문의
2. 완전 로컬 실행 가능한 대안 도구 검토
3. 제한적 네트워크 허용 (필수 도메인만)
