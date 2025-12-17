#!/bin/bash

# OpenCode 사내망 네트워크 접근성 테스트
# 이 스크립트를 사내망 환경에서 실행하여 OpenCode 사용 가능 여부를 확인하세요.

echo "========================================="
echo "OpenCode 사내망 네트워크 접근성 테스트"
echo "========================================="
echo ""
echo "실행 시간: $(date)"
echo ""

# 결과 저장
RESULT_FILE="network-test-result.txt"
echo "OpenCode 네트워크 테스트 결과 - $(date)" > "$RESULT_FILE"
echo "=========================================" >> "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

# 테스트 결과 변수
DNS_OK=0
HTTP_OK=0
HTTPS_OK=0
NPM_REGISTRY_OK=0
NPM_INSTALL_OK=0

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 결과 출력 함수
print_result() {
    local test_name="$1"
    local result="$2"
    local details="$3"

    if [ "$result" -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $test_name"
        echo "✓ $test_name" >> "$RESULT_FILE"
    else
        echo -e "${RED}✗${NC} $test_name"
        echo "✗ $test_name" >> "$RESULT_FILE"
    fi

    if [ -n "$details" ]; then
        echo "  → $details"
        echo "  → $details" >> "$RESULT_FILE"
    fi
}

print_section() {
    echo ""
    echo "========================================="
    echo "$1"
    echo "========================================="
    echo "" >> "$RESULT_FILE"
    echo "=========================================" >> "$RESULT_FILE"
    echo "$1" >> "$RESULT_FILE"
    echo "=========================================" >> "$RESULT_FILE"
}

# 1. DNS 해석 테스트
print_section "1. DNS 해석 테스트"

echo "registry.npmjs.org DNS 조회 중..."
if host registry.npmjs.org > /dev/null 2>&1; then
    IP=$(host registry.npmjs.org | grep "has address" | head -1 | awk '{print $4}')
    print_result "DNS 해석" 0 "IP: $IP"
    DNS_OK=1
    echo "  IP 주소: $IP" >> "$RESULT_FILE"
else
    print_result "DNS 해석" 1 "실패 - DNS 차단됨"
fi

# 2. 기본 네트워크 연결 테스트
print_section "2. 기본 네트워크 연결 테스트"

echo "Ping 테스트 중... (3회)"
if ping -c 3 registry.npmjs.org > /dev/null 2>&1; then
    print_result "Ping" 0 "성공"
else
    print_result "Ping" 1 "실패 - ICMP 차단 가능 (정상일 수 있음)"
fi

# 3. HTTP/HTTPS 연결 테스트
print_section "3. HTTP/HTTPS 연결 테스트"

echo "HTTP 연결 테스트 중..."
if curl -I -m 10 http://registry.npmjs.org > /dev/null 2>&1; then
    print_result "HTTP (포트 80)" 0 "연결 가능"
    HTTP_OK=1
else
    print_result "HTTP (포트 80)" 1 "연결 불가"
fi

echo "HTTPS 연결 테스트 중..."
HTTPS_RESULT=$(curl -I -m 10 https://registry.npmjs.org 2>&1)
HTTPS_STATUS=$?
if [ $HTTPS_STATUS -eq 0 ]; then
    print_result "HTTPS (포트 443)" 0 "연결 가능"
    HTTPS_OK=1
    echo "  응답 헤더:" >> "$RESULT_FILE"
    echo "$HTTPS_RESULT" | head -5 >> "$RESULT_FILE"
else
    print_result "HTTPS (포트 443)" 1 "연결 불가"
    echo "  에러: $HTTPS_RESULT" >> "$RESULT_FILE"
fi

# 4. npm 레지스트리 접근 테스트
print_section "4. npm 레지스트리 접근 테스트"

echo "npm 레지스트리 메타데이터 조회 중..."
NPM_META=$(curl -s -m 10 https://registry.npmjs.org/ 2>&1)
if echo "$NPM_META" | grep -q "db_name"; then
    print_result "npm 레지스트리 루트" 0 "접근 가능"
    NPM_REGISTRY_OK=1
else
    print_result "npm 레지스트리 루트" 1 "접근 불가"
    echo "  응답: $NPM_META" >> "$RESULT_FILE"
fi

# 5. OpenCode 필수 패키지 접근 테스트
print_section "5. OpenCode 필수 패키지 접근 테스트"

test_package() {
    local pkg_name="$1"
    local pkg_url="$2"

    echo "패키지 확인 중: $pkg_name"
    PKG_RESULT=$(curl -s -m 10 "$pkg_url" 2>&1)
    PKG_STATUS=$?

    if [ $PKG_STATUS -eq 0 ] && echo "$PKG_RESULT" | grep -q "\"name\""; then
        VERSION=$(echo "$PKG_RESULT" | grep -o '"version":"[^"]*"' | head -1 | cut -d'"' -f4)
        print_result "$pkg_name" 0 "버전: $VERSION"
        return 0
    else
        print_result "$pkg_name" 1 "접근 불가"
        echo "  응답: $PKG_RESULT" | head -3 >> "$RESULT_FILE"
        return 1
    fi
}

test_package "opencode-ai" "https://registry.npmjs.org/opencode-ai"
test_package "@opencode-ai/plugin" "https://registry.npmjs.org/@opencode-ai/plugin"
test_package "opencode-copilot-auth" "https://registry.npmjs.org/opencode-copilot-auth"
test_package "opencode-anthropic-auth" "https://registry.npmjs.org/opencode-anthropic-auth"

# 6. 실제 패키지 다운로드 테스트
print_section "6. 실제 패키지 다운로드 테스트 (npm)"

if command -v npm > /dev/null 2>&1; then
    echo "npm을 사용하여 실제 패키지 설치 테스트 중..."
    echo "테스트 디렉토리: /tmp/opencode-network-test"

    rm -rf /tmp/opencode-network-test
    mkdir -p /tmp/opencode-network-test
    cd /tmp/opencode-network-test

    npm init -y > /dev/null 2>&1

    echo "  - @opencode-ai/plugin@1.0.164 설치 시도..."
    NPM_INSTALL_OUTPUT=$(npm install @opencode-ai/plugin@1.0.164 --verbose 2>&1)
    NPM_INSTALL_STATUS=$?

    if [ $NPM_INSTALL_STATUS -eq 0 ]; then
        print_result "npm install 테스트" 0 "성공"
        NPM_INSTALL_OK=1
        echo "  설치된 패키지:" >> "$RESULT_FILE"
        ls -la node_modules/@opencode-ai/ >> "$RESULT_FILE" 2>&1
    else
        print_result "npm install 테스트" 1 "실패"
        echo "  에러 로그:" >> "$RESULT_FILE"
        echo "$NPM_INSTALL_OUTPUT" | tail -20 >> "$RESULT_FILE"
    fi

    cd - > /dev/null
else
    print_result "npm 설치 확인" 1 "npm이 설치되지 않음"
fi

# 7. 프록시 설정 확인
print_section "7. 프록시 설정 확인"

echo "환경 변수 프록시 설정:"
if [ -n "$HTTP_PROXY" ]; then
    print_result "HTTP_PROXY" 0 "$HTTP_PROXY"
else
    print_result "HTTP_PROXY" 1 "미설정"
fi

if [ -n "$HTTPS_PROXY" ]; then
    print_result "HTTPS_PROXY" 0 "$HTTPS_PROXY"
else
    print_result "HTTPS_PROXY" 1 "미설정"
fi

if [ -n "$NO_PROXY" ]; then
    echo "  NO_PROXY: $NO_PROXY"
    echo "  NO_PROXY: $NO_PROXY" >> "$RESULT_FILE"
fi

# npm 프록시 설정
if command -v npm > /dev/null 2>&1; then
    NPM_PROXY=$(npm config get proxy)
    NPM_HTTPS_PROXY=$(npm config get https-proxy)

    if [ "$NPM_PROXY" != "null" ]; then
        echo "  npm proxy: $NPM_PROXY"
        echo "  npm proxy: $NPM_PROXY" >> "$RESULT_FILE"
    fi

    if [ "$NPM_HTTPS_PROXY" != "null" ]; then
        echo "  npm https-proxy: $NPM_HTTPS_PROXY"
        echo "  npm https-proxy: $NPM_HTTPS_PROXY" >> "$RESULT_FILE"
    fi
fi

# 8. 추가 네트워크 정보
print_section "8. 추가 네트워크 정보"

echo "활성 네트워크 인터페이스:"
ip addr show | grep -E "^[0-9]+:|inet " | head -10
ip addr show | grep -E "^[0-9]+:|inet " | head -10 >> "$RESULT_FILE"

echo ""
echo "라우팅 테이블:"
ip route | head -5
ip route | head -5 >> "$RESULT_FILE"

echo ""
echo "DNS 서버:"
if [ -f /etc/resolv.conf ]; then
    grep nameserver /etc/resolv.conf
    grep nameserver /etc/resolv.conf >> "$RESULT_FILE"
fi

# 9. 최종 결과 요약
print_section "최종 결과 요약"

echo ""
echo "========================================="
echo "테스트 결과 요약"
echo "========================================="
echo "" | tee -a "$RESULT_FILE"

TOTAL_SCORE=0

if [ $DNS_OK -eq 1 ]; then
    echo -e "${GREEN}✓${NC} DNS 해석 가능"
    echo "✓ DNS 해석 가능" >> "$RESULT_FILE"
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
else
    echo -e "${RED}✗${NC} DNS 해석 불가"
    echo "✗ DNS 해석 불가" >> "$RESULT_FILE"
fi

if [ $HTTPS_OK -eq 1 ]; then
    echo -e "${GREEN}✓${NC} HTTPS 연결 가능"
    echo "✓ HTTPS 연결 가능" >> "$RESULT_FILE"
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
else
    echo -e "${RED}✗${NC} HTTPS 연결 불가"
    echo "✗ HTTPS 연결 불가" >> "$RESULT_FILE"
fi

if [ $NPM_REGISTRY_OK -eq 1 ]; then
    echo -e "${GREEN}✓${NC} npm 레지스트리 접근 가능"
    echo "✓ npm 레지스트리 접근 가능" >> "$RESULT_FILE"
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
else
    echo -e "${RED}✗${NC} npm 레지스트리 접근 불가"
    echo "✗ npm 레지스트리 접근 불가" >> "$RESULT_FILE"
fi

if [ $NPM_INSTALL_OK -eq 1 ]; then
    echo -e "${GREEN}✓${NC} npm 패키지 설치 가능"
    echo "✓ npm 패키지 설치 가능" >> "$RESULT_FILE"
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
else
    echo -e "${RED}✗${NC} npm 패키지 설치 불가"
    echo "✗ npm 패키지 설치 불가" >> "$RESULT_FILE"
fi

echo ""
echo "점수: $TOTAL_SCORE / 4" | tee -a "$RESULT_FILE"
echo "" | tee -a "$RESULT_FILE"

# 최종 판정
if [ $TOTAL_SCORE -eq 4 ]; then
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}결과: OpenCode 사용 가능 ✓${NC}"
    echo -e "${GREEN}=========================================${NC}"
    echo "결과: OpenCode 사용 가능 ✓" >> "$RESULT_FILE"
    echo "" | tee -a "$RESULT_FILE"
    echo "registry.npmjs.org에 정상적으로 접근할 수 있습니다." | tee -a "$RESULT_FILE"
    echo "OpenCode를 설치하고 사용할 수 있습니다." | tee -a "$RESULT_FILE"
    exit 0

elif [ $TOTAL_SCORE -ge 2 ]; then
    echo -e "${YELLOW}=========================================${NC}"
    echo -e "${YELLOW}결과: OpenCode 사용 가능 (제한적) ⚠${NC}"
    echo -e "${YELLOW}=========================================${NC}"
    echo "결과: OpenCode 사용 가능 (제한적) ⚠" >> "$RESULT_FILE"
    echo "" | tee -a "$RESULT_FILE"
    echo "일부 네트워크 기능이 제한되어 있습니다." | tee -a "$RESULT_FILE"
    echo "OpenCode 사용 중 일부 문제가 발생할 수 있습니다." | tee -a "$RESULT_FILE"
    echo "" | tee -a "$RESULT_FILE"
    echo "권장 조치:" | tee -a "$RESULT_FILE"

    if [ $NPM_INSTALL_OK -eq 0 ]; then
        echo "  - npm 패키지 설치가 실패했습니다." | tee -a "$RESULT_FILE"
        echo "  - 프록시 설정을 확인하세요." | tee -a "$RESULT_FILE"
        echo "  - 네트워크 관리자에게 registry.npmjs.org 접근 권한을 요청하세요." | tee -a "$RESULT_FILE"
    fi
    exit 1

else
    echo -e "${RED}=========================================${NC}"
    echo -e "${RED}결과: OpenCode 사용 불가 ✗${NC}"
    echo -e "${RED}=========================================${NC}"
    echo "결과: OpenCode 사용 불가 ✗" >> "$RESULT_FILE"
    echo "" | tee -a "$RESULT_FILE"
    echo "registry.npmjs.org에 접근할 수 없습니다." | tee -a "$RESULT_FILE"
    echo "OpenCode를 사용하려면 네트워크 설정이 필요합니다." | tee -a "$RESULT_FILE"
    echo "" | tee -a "$RESULT_FILE"
    echo "필요한 조치:" | tee -a "$RESULT_FILE"
    echo "  1. 네트워크 관리자에게 다음 도메인 접근 허용 요청:" | tee -a "$RESULT_FILE"
    echo "     - registry.npmjs.org (HTTPS, 포트 443)" | tee -a "$RESULT_FILE"
    echo "  2. 또는 로컬 npm 미러 서버 구축 (FINAL_SOLUTION.md 참고)" | tee -a "$RESULT_FILE"
    echo "  3. 또는 대안 도구 검토 (Continue.dev, Tabby 등)" | tee -a "$RESULT_FILE"
    exit 2
fi

echo ""
echo "========================================="
echo "상세 결과가 $RESULT_FILE 에 저장되었습니다."
echo "========================================="
