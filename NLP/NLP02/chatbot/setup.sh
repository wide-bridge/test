#!/bin/bash

# 한국어 챗봇 프로젝트 환경 설정 스크립트
# 지원: Ubuntu/Debian (WSL 포함), macOS

set -e

echo "=========================================="
echo "환경 설정 시작"
echo "=========================================="

# OS 감지
OS="$(uname -s)"
case "$OS" in
    Linux*)  PLATFORM="linux" ;;
    Darwin*) PLATFORM="mac" ;;
    *)       echo "지원하지 않는 OS: $OS"; exit 1 ;;
esac
echo "플랫폼: $PLATFORM"

# ──────────────────────────────────────────────
# 1. Java 설치 (KoNLPy 의존성)
# ──────────────────────────────────────────────
echo ""
echo "=========================================="
echo "[1/4] Java 설치 확인"
echo "=========================================="

if java -version 2>/dev/null; then
    echo "Java 이미 설치됨 — 건너뜁니다."
else
    echo "Java를 설치합니다..."
    if [ "$PLATFORM" = "linux" ]; then
        sudo apt-get update -qq
        sudo apt-get install -y default-jdk
    elif [ "$PLATFORM" = "mac" ]; then
        if command -v brew >/dev/null 2>&1; then
            brew install --cask temurin
        else
            echo "Homebrew가 없습니다. https://adoptium.net 에서 Java를 수동으로 설치하세요."
            exit 1
        fi
    fi
    echo "Java 설치 완료"
fi

# JAVA_HOME 설정
if [ -z "$JAVA_HOME" ]; then
    if [ "$PLATFORM" = "linux" ]; then
        JAVA_PATH="$(update-alternatives --query java 2>/dev/null | grep 'Value:' | awk '{print $2}' | sed 's|/bin/java||')"
        if [ -n "$JAVA_PATH" ]; then
            export JAVA_HOME="$JAVA_PATH"
            echo "export JAVA_HOME=$JAVA_HOME" >> ~/.bashrc
            echo "JAVA_HOME 설정: $JAVA_HOME"
        fi
    elif [ "$PLATFORM" = "mac" ]; then
        export JAVA_HOME="$(/usr/libexec/java_home 2>/dev/null || true)"
        echo "export JAVA_HOME=$JAVA_HOME" >> ~/.zshrc
        echo "JAVA_HOME 설정: $JAVA_HOME"
    fi
fi

# ──────────────────────────────────────────────
# 2. MeCab 설치
# ──────────────────────────────────────────────
echo ""
echo "=========================================="
echo "[2/4] MeCab 설치"
echo "=========================================="

if command -v mecab >/dev/null 2>&1; then
    echo "MeCab 이미 설치됨 — 건너뜁니다."
else
    echo "MeCab을 설치합니다..."
    if [ "$PLATFORM" = "linux" ]; then
        sudo apt-get update -qq
        sudo apt-get install -y mecab libmecab-dev mecab-ipadic-utf8

        # mecab-ko (한국어 사전) 설치
        echo "MeCab 한국어 사전(mecab-ko-dic) 설치 중..."
        TMP_DIR=$(mktemp -d)
        cd "$TMP_DIR"

        # mecab-ko
        wget -q https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
        tar -zxf mecab-0.996-ko-0.9.2.tar.gz
        cd mecab-0.996-ko-0.9.2
        ./configure
        make
        sudo make install
        cd ..

        # mecab-ko-dic
        wget -q https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
        tar -zxf mecab-ko-dic-2.1.1-20180720.tar.gz
        cd mecab-ko-dic-2.1.1-20180720
        ./autogen.sh
        ./configure
        make
        sudo make install
        cd ~
        rm -rf "$TMP_DIR"

    elif [ "$PLATFORM" = "mac" ]; then
        brew install mecab mecab-ko mecab-ko-dic
    fi
    echo "MeCab 설치 완료"
fi

# ──────────────────────────────────────────────
# 3. Python 패키지 설치
# ──────────────────────────────────────────────
echo ""
echo "=========================================="
echo "[3/4] Python 패키지 설치"
echo "=========================================="

# pip 업그레이드
python -m pip install --upgrade pip -q

# mecab-python3 (Python 바인딩)
echo "mecab-python3 설치 중..."
pip install mecab-python3

# konlpy
echo "konlpy 설치 중..."
pip install konlpy

# gensim
echo "gensim 설치 중..."
pip install gensim

# 프로젝트 공통 의존성
echo "기타 패키지 설치 중..."
pip install torch pandas numpy tqdm scikit-learn

echo "Python 패키지 설치 완료"

# ──────────────────────────────────────────────
# 4. 설치 검증
# ──────────────────────────────────────────────
echo ""
echo "=========================================="
echo "[4/4] 설치 검증"
echo "=========================================="

python - <<'PYEOF'
import sys

results = []

# Java (JPype 경유)
try:
    import jpype
    results.append(("Java/JPype", True, ""))
except Exception as e:
    results.append(("Java/JPype", False, str(e)))

# KoNLPy
try:
    import konlpy
    results.append(("konlpy", True, konlpy.__version__))
except Exception as e:
    results.append(("konlpy", False, str(e)))

# Gensim
try:
    import gensim
    results.append(("gensim", True, gensim.__version__))
except Exception as e:
    results.append(("gensim", False, str(e)))

# MeCab (python binding)
try:
    import MeCab
    results.append(("mecab-python3", True, ""))
except Exception as e:
    results.append(("mecab-python3", False, str(e)))

# MeCab via konlpy
try:
    from konlpy.tag import Mecab
    m = Mecab()
    m.pos("테스트")
    results.append(("KoNLPy Mecab", True, ""))
except Exception as e:
    results.append(("KoNLPy Mecab", False, str(e)))

print()
all_ok = True
for name, ok, info in results:
    status = "OK" if ok else "FAIL"
    detail = f" ({info})" if info else ""
    print(f"  [{status}] {name}{detail}")
    if not ok:
        all_ok = False

print()
if all_ok:
    print("모든 패키지가 정상적으로 설치되었습니다.")
else:
    print("일부 패키지에 문제가 있습니다. 위 FAIL 항목을 확인하세요.")
    sys.exit(1)
PYEOF

echo ""
echo "=========================================="
echo "환경 설정 완료!"
echo "새 터미널을 열거나 'source ~/.bashrc' 를 실행하세요."
echo "=========================================="
