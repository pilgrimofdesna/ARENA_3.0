#!/usr/bin/env bash
# =============================================================================
# Vast.ai 인스턴스를 [1.3.1] Linear Probes 한국어판을 바로 돌릴 수 있는 상태로 만든다.
#
# 새 인스턴스에서 한 줄:
#   curl -fsSL https://raw.githubusercontent.com/pilgrimofdesna/ARENA_3.0/refs/heads/study/linear-probes-ko/vast/setup.sh | bash
#
# 새 인스턴스마다 손 안 대려면 (한 번만):
#   - Vast 계정 설정의 환경 변수에 HF_TOKEN을 넣는다. 그러면 새 인스턴스에서 hf 로그인이 필요 없다.
#     (인스턴스 안에서 'hf auth login'으로 넣은 토큰은 /workspace와 함께 파괴 때 사라진다.)
#   - 쓰는 템플릿의 On-start script에 위 한 줄을 넣으면 인스턴스가 켜지면서 알아서 세팅된다.
#     진행 상황: tail -f /workspace/arena_setup.log (On-start에서는 '| bash > /workspace/arena_setup.log 2>&1')
#
# 하는 일 (여러 번 실행해도 안전하다. 끝난 단계는 건너뛴다):
#   1. 저장소 clone (이미 있으면 pull)
#   2. uv로 Python 3.12 venv 만들고 torch(cu128) + requirements.txt 설치
#   3. Jupyter kernel 등록, VS Code Remote가 이 venv를 기본 interpreter로 쓰게 설정
#   4. HF_TOKEN이 있으면 Llama-3.1-8B-Instruct 접근 확인 후 weight 미리 받기
#   5. GPU / 디스크 점검 후 요약 출력
#
# 설정 (환경 변수로 바꿀 수 있다):
#   HF_TOKEN       Hugging Face 토큰. Vast 템플릿/계정의 환경 변수로 넣는다. 저장소에 쓰지 않는다.
#   ARENA_DIR      clone 위치            (기본: /workspace/ARENA_3.0)
#   ARENA_BRANCH   branch                (기본: study/linear-probes-ko)
#   ARENA_VENV     venv 위치. 비워 두면 이미지의 /venv/main에 같은 torch가 있을 때 그걸 쓰고,
#                  없으면 /workspace/venv-arena를 새로 만든다.
#   BASE_VENV      이미지에 들어 있는 venv (기본: /venv/main, Vast PyTorch 이미지)
#   SKIP_MODEL=1   model weight 미리 받기를 건너뛴다
#   SKIP_TORCH=1   torch 설치를 건너뛴다 (테스트용)
# =============================================================================
set -euo pipefail

REPO_URL="https://github.com/pilgrimofdesna/ARENA_3.0.git"
ARENA_BRANCH="${ARENA_BRANCH:-study/linear-probes-ko}"
WORKSPACE="${WORKSPACE:-/workspace}"
ARENA_DIR="${ARENA_DIR:-$WORKSPACE/ARENA_3.0}"
ARENA_VENV="${ARENA_VENV:-}"
BASE_VENV="${BASE_VENV:-/venv/main}"
PART_REL="chapter1_transformer_interp/exercises/part31_linear_probes_ko"
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
TORCH_VERSION="2.11.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
# uv 기본 타임아웃(30초)이면 Vast에서 torch 의존 패키지(수백 MB) 받다가 끊긴다.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

step() { printf '\n\033[1;36m=== [%s] %s ===\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[1;33m[주의] %s\033[0m\n' "$*"; }

mkdir -p "$WORKSPACE"
START=$(date +%s)

# ---------------------------------------------------------------------------
step "1/5 저장소 ($ARENA_BRANCH)"
if ! command -v git >/dev/null 2>&1; then
    SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
    $SUDO apt-get update -qq && $SUDO apt-get install -y -qq git
fi
if [ -d "$ARENA_DIR/.git" ]; then
    git -C "$ARENA_DIR" fetch -q origin "$ARENA_BRANCH"
    if [ -n "$(git -C "$ARENA_DIR" status --porcelain --untracked-files=no)" ]; then
        warn "$ARENA_DIR 에 커밋 안 된 수정이 있어서 pull을 건너뛴다."
    else
        git -C "$ARENA_DIR" checkout -q "$ARENA_BRANCH"
        git -C "$ARENA_DIR" merge -q --ff-only "origin/$ARENA_BRANCH" || warn "fast-forward가 안 돼서 pull을 건너뛴다."
    fi
else
    # 전체 기록은 필요 없다. 이 branch의 최신 상태만 받는다.
    git clone -q --depth 1 --branch "$ARENA_BRANCH" "$REPO_URL" "$ARENA_DIR"
fi
PART_DIR="$ARENA_DIR/$PART_REL"
[ -f "$PART_DIR/requirements.txt" ] || { echo "requirements.txt를 못 찾음: $PART_DIR"; exit 1; }
echo "HEAD: $(git -C "$ARENA_DIR" log -1 --format='%h %s')"

# ---------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# 이미지에 같은 버전의 CUDA torch가 들어 있으면 그 venv를 그대로 쓴다. torch를 다시 받는 몇 분이 사라진다.
if [ -z "$ARENA_VENV" ]; then
    if [ -x "$BASE_VENV/bin/python" ] && "$BASE_VENV/bin/python" -c "
import sys, torch
sys.exit(0 if torch.__version__.split('+')[0] == '$TORCH_VERSION' and torch.version.cuda else 1)
" 2>/dev/null; then
        ARENA_VENV="$BASE_VENV"
    else
        ARENA_VENV="$WORKSPACE/venv-arena"
    fi
fi
step "2/5 Python 환경 ($ARENA_VENV)"
if [ ! -x "$ARENA_VENV/bin/python" ]; then
    uv venv --python 3.12 "$ARENA_VENV"
fi
PY="$ARENA_VENV/bin/python"
# run_notebook.sh가 같은 venv를 찾도록 남겨 둔다.
echo "$ARENA_VENV" > "$WORKSPACE/.arena_venv"
# requirements.txt가 바뀌었을 때만 다시 설치한다.
REQ_HASH=$(sha256sum "$PART_DIR/requirements.txt" | cut -c1-16)
STAMP="$ARENA_VENV/.arena-req-$REQ_HASH"
if [ ! -f "$STAMP" ]; then
    if [ "${SKIP_TORCH:-0}" != "1" ]; then
        if ! "$PY" -c "import sys, torch; sys.exit(0 if torch.__version__.split('+')[0] == '$TORCH_VERSION' and torch.version.cuda else 1)" 2>/dev/null; then
            uv pip install --python "$PY" "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
        else
            echo "torch $TORCH_VERSION (CUDA) 이미 있음"
        fi
        uv pip install --python "$PY" -r "$PART_DIR/requirements.txt" nbconvert
    else
        grep -v '^torch==' "$PART_DIR/requirements.txt" > "$ARENA_VENV/req-notorch.txt"
        uv pip install --python "$PY" -r "$ARENA_VENV/req-notorch.txt" nbconvert
    fi
    rm -f "$ARENA_VENV"/.arena-req-*
    touch "$STAMP"
else
    echo "requirements 변경 없음, 설치 건너뜀"
fi

# ---------------------------------------------------------------------------
step "3/5 Jupyter kernel, VS Code 설정"
KERNEL_DISPLAY="ARENA ($ARENA_VENV)"
"$PY" -m ipykernel install --user --name arena --display-name "$KERNEL_DISPLAY" >/dev/null
echo "kernel '$KERNEL_DISPLAY' 등록"
# VS Code Remote-SSH로 붙었을 때 어느 폴더를 열어도 이 venv를 기본으로 쓰게 한다.
VSC_MACHINE="$HOME/.vscode-server/data/Machine"
mkdir -p "$VSC_MACHINE"
# 이 스크립트가 만든 파일이면(표시 줄이 있으면) 새 venv 경로로 다시 쓴다. 사람이 만든 파일은 건드리지 않는다.
if [ ! -f "$VSC_MACHINE/settings.json" ] || grep -q '"arena.setupManaged": true' "$VSC_MACHINE/settings.json"; then
    cat > "$VSC_MACHINE/settings.json" <<JSON
{
    "arena.setupManaged": true,
    "python.defaultInterpreterPath": "$PY",
    "jupyter.notebookFileRoot": "\${fileDirname}"
}
JSON
    echo "VS Code 기본 interpreter: $PY"
else
    echo "VS Code Machine 설정이 이미 있어서 그대로 둠 ($VSC_MACHINE/settings.json)"
fi

# ---------------------------------------------------------------------------
step "4/5 Hugging Face ($MODEL_ID)"
MODEL_OK=0
if [ -z "${HF_TOKEN:-}" ]; then
    warn "HF_TOKEN이 없다. Vast 템플릿 환경 변수에 넣거나, 지금 'hf auth login'을 한 번 실행한다."
    warn "(이미 'hf auth login'을 해 둔 인스턴스면 무시해도 된다.)"
fi
if "$PY" - "$MODEL_ID" <<'PYEOF'
import sys
from huggingface_hub import model_info, whoami
mid = sys.argv[1]
try:
    print("HF 계정:", whoami()["name"])
    model_info(mid)
    print("모델 접근 OK:", mid)
except Exception as e:
    print("모델 접근 실패:", type(e).__name__, str(e).splitlines()[0][:200])
    sys.exit(1)
PYEOF
then
    MODEL_OK=1
    if [ "${SKIP_MODEL:-0}" != "1" ]; then
        echo "weight 받는 중 (약 16 GB, 처음 한 번만)..."
        "$ARENA_VENV/bin/hf" download "$MODEL_ID" --exclude "original/*" >/dev/null
        echo "weight 준비 완료"
    fi
else
    warn "모델 접근이 안 된다. https://huggingface.co/$MODEL_ID 에서 승인받은 계정의 토큰인지 확인한다."
fi

# ---------------------------------------------------------------------------
step "5/5 점검"
"$PY" - <<'PYEOF'
import shutil, torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    gib = p.total_memory / 2**30
    print(f"GPU: {p.name} {gib:.0f} GiB")
    if gib < 24:
        print("[주의] GPU 메모리 24 GiB 미만. bf16 8B 모델(약 15 GiB)+activation이 빠듯하다.")
else:
    print("[주의] CUDA GPU가 안 보인다.")
free = shutil.disk_usage("/workspace" if __import__("os").path.isdir("/workspace") else "/").free / 2**30
print(f"남은 디스크: {free:.0f} GiB" + ("  [주의] activation cache까지 생각하면 10 GiB 이상 남겨 둔다." if free < 10 else ""))
print("torch", torch.__version__)
PYEOF

ELAPSED=$(( $(date +%s) - START ))
cat <<MSG

==================== 준비 끝 (${ELAPSED}초) ====================
폴더:    $PART_DIR
kernel:  $KERNEL_DISPLAY
모델:    $( [ "$MODEL_OK" = 1 ] && echo "접근 OK" || echo "접근 안 됨 (위 주의 참고)" )

다음:
  - VS Code Remote-SSH로 붙어서 위 폴더의 notebook을 열고 kernel '$KERNEL_DISPLAY' 선택
  - 또는 notebook 하나를 통째로 실행해 결과 저장:
      bash $ARENA_DIR/vast/run_notebook.sh 01
==============================================================
MSG
