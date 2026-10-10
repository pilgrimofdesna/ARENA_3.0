#!/usr/bin/env bash
# =============================================================================
# [1.3.1] 한국어판 notebook 하나를 처음부터 끝까지 실행하고 결과를 남긴다.
#
#   bash vast/run_notebook.sh 01        # ① probe가_묻는_질문
#   bash vast/run_notebook.sh 04        # ④ 인과_개입
#
# 남는 것:
#   <part>/runs/<날짜-시각>_<번호>/  실행된 notebook + 그 notebook이 쓴 results/*.json 사본
#   <part>/results/*.json            notebook이 덮어쓴다. 'git diff results/'로 저장소의 기준 출력과 비교할 수 있다.
# =============================================================================
set -euo pipefail

NUM="${1:?사용법: bash vast/run_notebook.sh <번호, 예: 01>}"
NUM=$(printf '%02d' "$((10#$NUM))")
ARENA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# setup.sh가 고른 venv를 쓴다 (/venv/main 또는 /workspace/venv-arena).
ARENA_VENV="${ARENA_VENV:-$(cat "${WORKSPACE:-/workspace}/.arena_venv" 2>/dev/null || echo /workspace/venv-arena)}"
PART_DIR="$ARENA_DIR/chapter1_transformer_interp/exercises/part31_linear_probes_ko"

shopt -s nullglob
matches=("$PART_DIR/${NUM}_"*.ipynb)
[ "${#matches[@]}" -eq 1 ] || { echo "${NUM}번 notebook을 못 찾음 (또는 여러 개): $PART_DIR"; exit 1; }
NB="${matches[0]}"

[ -x "$ARENA_VENV/bin/jupyter" ] || { echo "venv가 없다. 먼저 vast/setup.sh를 실행한다."; exit 1; }

RUN_DIR="$PART_DIR/runs/$(date +%Y%m%d-%H%M)_$NUM"
mkdir -p "$RUN_DIR"
MARK="$RUN_DIR/.start"; touch "$MARK"

echo "실행: $(basename "$NB")"
echo "로그: $RUN_DIR/run.log"
START=$(date +%s)
set +e
"$ARENA_VENV/bin/jupyter" nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=arena \
    --output-dir "$RUN_DIR" "$NB" 2>&1 | tee "$RUN_DIR/run.log"
STATUS=${PIPESTATUS[0]}
set -e
ELAPSED=$(( $(date +%s) - START ))

# 이번 실행이 새로 쓴 results 파일만 사본으로 남긴다.
for f in "$PART_DIR"/results/*; do
    [ "$f" -nt "$MARK" ] && cp "$f" "$RUN_DIR/"
done
rm -f "$MARK"

echo
if [ "$STATUS" -eq 0 ]; then
    echo "완료 (${ELAPSED}초). 결과: $RUN_DIR"
    ls -1 "$RUN_DIR"
    echo
    echo "저장소 기준 출력과 비교:  git -C $ARENA_DIR diff --stat -- $PART_DIR/results"
    echo "GitHub에 남기기:          git -C $ARENA_DIR add $RUN_DIR && git -C $ARENA_DIR commit -m 'run $NUM' && git -C $ARENA_DIR push"
else
    echo "실패 (${ELAPSED}초, exit $STATUS). 에러는 $RUN_DIR/run.log 끝부분에 있다."
    tail -n 20 "$RUN_DIR/run.log"
fi
exit "$STATUS"
