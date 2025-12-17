# OpenCode 사내망 사용 최종 해결책

## 핵심 발견

### 1. Custom Provider는 이미 해결됨 ✅
- **AI API 문제 없음**: 내부망 AI API를 Custom Provider로 사용 가능
- 모델 API는 신경 쓸 필요 없음

### 2. 남은 문제: 불필요한 provider auth 패키지들 ❌

사용자 질문이 정확합니다:
> "어차피 custom provider 사용할건데, auth 나 다른 provider는 필요없을 것 같은데?"

**맞습니다!** Custom Provider만 사용한다면 이런 패키지들이 필요 없어야 합니다:
- `opencode-copilot-auth` (GitHub Copilot용)
- `opencode-anthropic-auth` (Anthropic Claude용)
- `opencode-gemini-auth` (Google Gemini용)
- 기타 provider별 auth 패키지들

**하지만 문제는:**
OpenCode는 실행 시 자동으로 이 모든 provider auth 플러그인들을 로드하려고 시도합니다.

```
로그에서 확인된 동작:
INFO service=provider init
INFO service=plugin path=opencode-copilot-auth@0.0.9 loading plugin
INFO service=plugin path=opencode-anthropic-auth@0.0.5 loading plugin
→ 설치 안 되어 있으면 Bun으로 자동 설치 시도
→ 401 에러 → OpenCode 종료
```

OpenCode는 **컴파일된 바이너리**라서 이 동작을 수정할 수 없습니다.

## npm 의존성 로컬 복사에 대해

### 질문
> "npm 의존성 파일들도 그냥 내가 로컬에서 싹다 다운로드 받아놓은 다음에, 그대로 올리는건 안돼?"

### 답변: 아쉽게도 직접 복사만으로는 안 됩니다

**이미 테스트한 결과:**

1. **npm으로 패키지 설치** ✅
   ```bash
   npm install @opencode-ai/plugin opencode-copilot-auth opencode-anthropic-auth
   ```
   → 정상 설치됨

2. **node_modules + package-lock.json 복사** ✅
   ```bash
   cp -r node_modules /root/.cache/opencode/
   cp package-lock.json /root/.cache/opencode/
   ```
   → 정상 복사됨

3. **OpenCode 실행** ❌
   ```
   INFO service=bun cmd=["opencode","add","--force","--exact","opencode-copilot-auth@0.0.9"]
   ERROR GET https://registry.npmjs.org/opencode-copilot-auth - 401
   ```
   → OpenCode가 `--force` 플래그로 **재설치를 시도**
   → npm 레지스트리에 접근 시도
   → 실패

**왜 안 될까?**
- OpenCode 내장 Bun이 `--force` (강제 재설치) 플래그 사용
- 기존에 설치되어 있어도 무시하고 새로 다운로드 시도
- 이건 OpenCode의 설계 방식 (바이너리라서 수정 불가)

## 실제 작동하는 해결책

### 방법 1: npm 레지스트리만 허용 (가장 간단) ⭐⭐⭐

사내망 방화벽 설정:
```
허용 도메인:
✅ registry.npmjs.org     (OpenCode 패키지 다운로드)
✅ 내부 AI API 엔드포인트 (Custom Provider용)

차단:
❌ models.dev           (모델 정보, 없어도 동작 가능)
❌ 기타 모든 외부 도메인
```

**장점:**
- 가장 간단함
- OpenCode 정상 동작
- AI는 내부망만 사용
- 외부로 나가는 트래픽: npm 패키지 다운로드만 (최초 실행 시)

**단점:**
- 완전한 오프라인은 아님
- npm 레지스트리 의존성 있음

### 방법 2: 로컬 npm 미러 구축 (기술적) ⭐⭐

**온라인 환경에서 준비:**
```bash
# 1. 모든 필요한 패키지 다운로드
mkdir opencode-packages && cd opencode-packages

npm install -g opencode-ai
npm pack @opencode-ai/plugin@1.0.164
npm pack opencode-copilot-auth@0.0.9
npm pack opencode-anthropic-auth@0.0.5
npm pack opencode-gemini-auth@latest
npm pack opencode-openai-codex-auth@latest
# ... 기타 provider auth 패키지들

# 2. 의존성들도 다운로드
cd node_modules && for dir in */; do npm pack "$dir"; done
cd ..

# 전체를 tarball로 압축
tar czf opencode-packages.tar.gz *.tgz
```

**사내망 서버에 구축:**
```bash
# 1. Verdaccio 설치 (로컬 npm 레지스트리)
npm install -g verdaccio

# 2. Verdaccio 실행
verdaccio &
# 기본 포트: 4873

# 3. 다운로드한 패키지들 업로드
for tgz in *.tgz; do
  npm publish "$tgz" --registry http://localhost:4873/
done
```

**네트워크 리다이렉션 설정:**
```bash
# 방법 A: DNS 리다이렉션
# registry.npmjs.org → 내부서버IP

# 방법 B: 프록시 설정 (nginx/squid)
# registry.npmjs.org 요청을 내부 Verdaccio로 리다이렉션

# 방법 C: /etc/hosts (간단 테스트용)
echo "192.168.1.100 registry.npmjs.org" >> /etc/hosts
```

**장점:**
- 완전 오프라인 가능
- npm 레지스트리 독립적

**단점:**
- 설정 복잡함
- Verdaccio 서버 관리 필요
- 네트워크 리다이렉션 설정 필요
- OpenCode 내장 Bun이 인증서 검증 문제 있을 수 있음

### 방법 3: OpenCode 개발팀 문의 ⭐

```
문의 내용:
1. Custom Provider 전용 모드 존재 여부
2. 불필요한 provider auth 플러그인 비활성화 방법
3. 엔터프라이즈/오프라인 버전 존재 여부
4. 오프라인 구성 공식 가이드

연락처:
- GitHub: https://github.com/opencodeco/opencode (추정)
- 웹사이트: https://opencode.ai
```

## 권장사항

### Custom Provider만 사용하는 경우

**1순위: npm 레지스트리만 허용**
```
이유:
- 가장 간단하고 확실함
- AI는 이미 내부망 사용 중
- npm 패키지는 공개 소프트웨어 (보안 문제 적음)
- 최초 실행 후에는 캐시됨
```

**2순위: 로컬 npm 미러**
```
이유:
- 완전 오프라인 필수인 경우
- 네트워크 인프라 팀 지원 필요
- 유지보수 부담 있음
```

**3순위: OpenCode 개발팀 지원 요청**
```
이유:
- 공식 오프라인 버전이 있을 수 있음
- Custom Provider 전용 설정이 있을 수 있음
```

## 핵심 정리

### 질문1: Custom Provider 사용하는데 다른 auth 필요한가?
**답:** 필요 없어야 하지만, OpenCode가 자동으로 로드 시도합니다.
- 이건 OpenCode의 설계 방식
- 바이너리라서 수정 불가
- 설정으로 비활성화하는 방법을 찾지 못했음

### 질문2: npm 파일들 다운로드해서 그대로 올리면 안 되나?
**답:** 직접 복사만으로는 안 됩니다. 이유는:
1. OpenCode가 `--force`로 재설치 시도
2. npm 레지스트리 접근 필요
3. **해결책:** 로컬 npm 레지스트리 + 네트워크 리다이렉션

### 최선의 현실적 방법

```
registry.npmjs.org만 허용
  ↓
Custom Provider로 내부 AI 사용
  ↓
OpenCode 정상 동작 ✅
```

## 구현 예시

### registry.npmjs.org 허용 설정 (방화벽)

```bash
# iptables 예시
iptables -A OUTPUT -d registry.npmjs.org -j ACCEPT
iptables -A OUTPUT -d <내부AI서버IP> -j ACCEPT
iptables -A OUTPUT -j REJECT

# 또는 프록시 화이트리스트
# squid.conf:
acl allowed_domains dstdomain registry.npmjs.org
acl allowed_domains dstdomain <내부AI도메인>
http_access allow allowed_domains
http_access deny all
```

### OpenCode 사용

```bash
# 1. 최초 실행 (패키지 다운로드)
opencode --version
# → npm 레지스트리에서 필요한 패키지 자동 다운로드

# 2. Custom Provider 설정
opencode auth list
# → Custom Provider 추가 방법 확인

# 3. 이후 사용
opencode
# → 캐시된 패키지 사용, npm 접근 최소화
```

## 요약

| 항목 | 필요성 | 해결 방법 |
|------|--------|----------|
| AI API | ✅ 해결됨 | Custom Provider 사용 |
| npm 레지스트리 | ❌ 필요 | `registry.npmjs.org` 허용 (권장) |
| | | 또는 로컬 npm 미러 구축 |
| provider auth 패키지 | ❓ 불필요하지만 | OpenCode가 자동 로드 시도 |
| | | (비활성화 방법 찾지 못함) |

**결론: `registry.npmjs.org` 하나만 허용하면 됩니다.** 🎯
