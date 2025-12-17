# OpenCode 사내망 네트워크 테스트 가이드

## 목적

이 스크립트는 사내망 환경에서 OpenCode를 사용할 수 있는지 미리 확인하기 위한 도구입니다.

## 테스트 항목

### 1. DNS 해석
- `registry.npmjs.org` 도메인을 IP 주소로 변환할 수 있는지 확인
- 차단 시: DNS 레벨에서 차단됨

### 2. 기본 네트워크 연결
- Ping 테스트 (ICMP)
- 일부 환경에서는 차단되어 있어도 정상

### 3. HTTP/HTTPS 연결
- 포트 80 (HTTP) 접근 가능 여부
- 포트 443 (HTTPS) 접근 가능 여부
- **OpenCode는 HTTPS 필수**

### 4. npm 레지스트리 접근
- `https://registry.npmjs.org/` 루트 접근
- 메타데이터 조회 가능 여부

### 5. OpenCode 필수 패키지 접근
- `opencode-ai`
- `@opencode-ai/plugin`
- `opencode-copilot-auth`
- `opencode-anthropic-auth`

### 6. 실제 패키지 다운로드
- npm을 사용한 실제 패키지 설치 테스트
- 가장 확실한 테스트

### 7. 프록시 설정 확인
- 환경 변수 프록시 설정
- npm 프록시 설정

### 8. 추가 네트워크 정보
- 네트워크 인터페이스
- 라우팅 테이블
- DNS 서버

## 사용 방법

### 1. 스크립트 전송

**온라인 환경에서 다운로드:**
```bash
git clone <이 저장소>
cd OpenCode
```

**사내망으로 전송:**
```bash
# USB, 이메일, 내부 공유 폴더 등을 통해 전송
cp test-network-connectivity.sh /사내망/경로/
```

### 2. 실행

```bash
cd /사내망/경로/
chmod +x test-network-connectivity.sh
./test-network-connectivity.sh
```

### 3. 결과 확인

스크립트 실행 중 실시간 결과가 출력되며, 완료 후:
- 화면에 최종 판정 표시
- `network-test-result.txt` 파일에 상세 결과 저장

## 결과 해석

### 점수: 4/4 - OpenCode 사용 가능 ✓

```
✓ DNS 해석 가능
✓ HTTPS 연결 가능
✓ npm 레지스트리 접근 가능
✓ npm 패키지 설치 가능

결과: registry.npmjs.org에 완전히 접근 가능합니다.
→ OpenCode를 바로 설치하고 사용할 수 있습니다!
```

**다음 단계:**
1. OpenCode 설치: `npm install -g opencode-ai`
2. Custom Provider 설정
3. 사용 시작

### 점수: 2-3/4 - OpenCode 사용 가능 (제한적) ⚠

```
일부 테스트 실패

예:
✓ DNS 해석 가능
✓ HTTPS 연결 가능
✗ npm 레지스트리 접근 불가
✗ npm 패키지 설치 불가

결과: 제한적으로 접근 가능하거나 프록시 설정 필요
```

**가능한 원인:**
- 프록시 서버 미설정
- 인증서 검증 실패
- 일부 URL 필터링

**해결 방법:**
1. **프록시 설정 확인**
   ```bash
   # 환경 변수 설정
   export HTTP_PROXY=http://프록시서버:포트
   export HTTPS_PROXY=http://프록시서버:포트
   export NO_PROXY=localhost,127.0.0.1

   # npm 프록시 설정
   npm config set proxy http://프록시서버:포트
   npm config set https-proxy http://프록시서버:포트
   ```

2. **인증서 문제 (임시)**
   ```bash
   # 주의: 보안 위험이 있으므로 테스트 용도로만 사용
   npm config set strict-ssl false
   ```

3. **다시 테스트**
   ```bash
   ./test-network-connectivity.sh
   ```

### 점수: 0-1/4 - OpenCode 사용 불가 ✗

```
대부분의 테스트 실패

예:
✗ DNS 해석 불가
✗ HTTPS 연결 불가
✗ npm 레지스트리 접근 불가
✗ npm 패키지 설치 불가

결과: registry.npmjs.org 완전 차단
```

**필요한 조치:**

#### 옵션 1: 네트워크 관리자에게 요청 (권장)

이메일 예시:
```
제목: OpenCode 사용을 위한 npm 레지스트리 접근 권한 요청

안녕하세요,

AI 코딩 도구인 OpenCode를 사용하려고 합니다.
다음 도메인에 대한 HTTPS(443 포트) 접근 권한이 필요합니다:

- registry.npmjs.org (npm 패키지 레지스트리)
  → 용도: OpenCode 설치 및 업데이트
  → 프로토콜: HTTPS (포트 443)
  → 공식 사이트: https://www.npmjs.com

AI 모델은 내부망 API를 사용하므로,
외부 AI 서비스 연결은 필요하지 않습니다.

테스트 결과 파일을 첨부합니다.
(network-test-result.txt)

감사합니다.
```

#### 옵션 2: 로컬 npm 미러 구축

`FINAL_SOLUTION.md` 파일의 "방법 2: 로컬 npm 미러 구축" 참조

1. Verdaccio 서버 구축
2. 필요한 패키지 미리 다운로드
3. 내부 서버에 업로드
4. 네트워크 리다이렉션 설정

#### 옵션 3: 대안 도구 검토

완전 오프라인 가능한 도구들:
- **Continue.dev** + Ollama (로컬 LLM)
- **Tabby** (자체 호스팅)
- **Aider** (로컬 LLM 지원)

`README.md` 파일 참조

## 상세 로그 확인

```bash
# 전체 결과 파일 확인
cat network-test-result.txt

# 특정 섹션만 확인
grep "✓\|✗" network-test-result.txt

# 에러 메시지만 확인
grep "에러\|실패" network-test-result.txt
```

## 문제 해결

### DNS 해석 실패

**증상:**
```
✗ DNS 해석 실패 - DNS 차단됨
```

**원인:**
- DNS 레벨 차단
- 내부 DNS 서버 설정

**해결:**
1. `/etc/resolv.conf` 확인
2. 네트워크 관리자에게 DNS 설정 확인 요청

### HTTPS 연결 실패

**증상:**
```
✓ DNS 해석 가능
✗ HTTPS (포트 443) 연결 불가
```

**원인:**
- 방화벽에서 443 포트 차단
- 프록시 미설정

**해결:**
1. 프록시 설정 확인
2. 방화벽 규칙 확인 요청

### npm 레지스트리 접근 불가

**증상:**
```
✓ HTTPS 연결 가능
✗ npm 레지스트리 루트 접근 불가
```

**원인:**
- URL 필터링
- 인증서 검증 실패
- 프록시 인증 필요

**해결:**
1. 프록시 인증 정보 설정
2. SSL 인증서 확인
3. 네트워크 팀에 상세 로그 전달

### npm 패키지 설치 실패

**증상:**
```
✓ npm 레지스트리 루트 접근 가능
✗ npm install 테스트 실패
```

**원인:**
- 패키지 다운로드 중 연결 끊김
- 대용량 파일 전송 제한
- 타임아웃

**해결:**
```bash
# 타임아웃 늘리기
npm config set timeout 60000

# 재시도 횟수 늘리기
npm config set fetch-retry-maxtimeout 120000
npm config set fetch-retries 5

# 다시 테스트
./test-network-connectivity.sh
```

## 추가 정보

### 필요한 네트워크 대역폭

OpenCode 패키지 크기:
- opencode-ai: ~525MB (최초 설치)
- 플러그인 패키지들: ~5-10MB

**대역폭 요구사항:**
- 최초 설치: ~530MB 다운로드
- 이후 업데이트: ~10-50MB
- 실시간 사용: 내부망 AI API만 사용 (외부 연결 불필요)

### 보안 고려사항

**외부 연결:**
- registry.npmjs.org만 필요
- 공개 npm 패키지 다운로드
- AI 모델 API는 내부망 사용

**데이터 전송:**
- 다운로드만 발생 (업로드 없음)
- 코드는 내부망에서만 처리
- 외부로 코드 전송되지 않음

### 지원

문제 발생 시:
1. `network-test-result.txt` 파일 확인
2. 네트워크 관리자에게 전달
3. 이 저장소의 다른 문서 참조:
   - `FINAL_SOLUTION.md`: 최종 해결책
   - `DEPENDENCY_ANALYSIS.md`: 의존성 분석
   - `README.md`: 전체 개요

## 예상 실행 시간

- 정상 환경: ~30초
- 타임아웃 발생 시: ~2-3분

## 스크립트 요구사항

필수 명령어:
- `curl` (HTTP 요청)
- `host` 또는 `dig` (DNS 조회)
- `ping` (선택적)
- `npm` (패키지 설치 테스트용)

대부분의 Linux 환경에 기본 설치되어 있습니다.
