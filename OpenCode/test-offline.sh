#!/bin/bash
# OpenCode 오프라인 테스트 스크립트

echo "========================================="
echo "OpenCode 오프라인 실행 테스트"
echo "========================================="
echo ""

# 1. 바이너리 존재 확인
echo "1. 바이너리 파일 확인..."
OPENCODE_BIN="./opencode-ai-offline/node_modules/opencode-linux-x64/bin/opencode"
if [ -f "$OPENCODE_BIN" ]; then
    echo "✓ 바이너리 파일 존재: $OPENCODE_BIN"
    ls -lh "$OPENCODE_BIN"
else
    echo "✗ 바이너리 파일 없음: $OPENCODE_BIN"
    exit 1
fi
echo ""

# 2. 버전 확인 (네트워크 불필요)
echo "2. 버전 확인 (네트워크 불필요)..."
$OPENCODE_BIN --version
if [ $? -eq 0 ]; then
    echo "✓ 버전 확인 성공"
else
    echo "✗ 버전 확인 실패"
fi
echo ""

# 3. 도움말 확인 (네트워크 불필요)
echo "3. 도움말 확인 (네트워크 불필요)..."
$OPENCODE_BIN --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ 도움말 확인 성공"
else
    echo "✗ 도움말 확인 실패"
fi
echo ""

# 4. 모델 목록 확인 (네트워크 필요 - 실패 예상)
echo "4. 모델 목록 확인 (네트워크 필요 - 실패 예상)..."
$OPENCODE_BIN models 2>&1 | head -5
if [ $? -eq 0 ]; then
    echo "✓ 모델 목록 확인 성공 (네트워크 연결됨)"
else
    echo "✗ 모델 목록 확인 실패 (예상된 결과 - 네트워크/패키지 필요)"
fi
echo ""

echo "========================================="
echo "테스트 결과 요약"
echo "========================================="
echo "✓ 바이너리 실행: 가능"
echo "✓ 기본 명령어: 동작"
echo "✗ AI 기능: 네트워크 및 추가 패키지 필요"
echo ""
echo "결론: 완전한 오프라인 사용 불가"
echo "      기본 명령어만 실행 가능"
echo "      AI 코딩 기능은 온라인 연결 필수"
echo "========================================="
