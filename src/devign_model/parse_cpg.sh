#!/usr/bin/env bash
set -euo pipefail

# ---- Defaults ----
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../ && pwd)"
IN_DIR="${ROOT_DIR}/data/raw/devign_c"
OUT_DIR="${ROOT_DIR}/data/cpg"
REPR="cpg14"
JOBS=1
LIMIT=0

# ---- Args ----
while getopts ":j:n:" opt; do
  case "$opt" in
    j) JOBS=${OPTARG} ;;
    n) LIMIT=${OPTARG} ;;
  esac
done

mkdir -p "${OUT_DIR}" "${OUT_DIR}/logs"

echo "ROOT_DIR=${ROOT_DIR}"
echo "IN_DIR=${IN_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "REPR=${REPR}"
echo "JOBS=${JOBS}"
echo "LIMIT=${LIMIT}"

count=0
running=0

# ❗ Dùng process substitution để while chạy trong **cùng shell** (không bị subshell)
while IFS= read -r -d '' f; do
  (( LIMIT>0 && count>=LIMIT )) && break

  base="$(basename "$f")"
  stem="${base%.*}"
  bin="${OUT_DIR}/${stem}.bin"
  json="${OUT_DIR}/${stem}.json"
  log="${OUT_DIR}/logs/${stem}.log"

  # Nếu lỡ có THƯ MỤC trùng tên file json -> xoá để export ra file
  [ -d "${json}" ] && rm -rf "${json}"

  ( joern-parse "$f" --out "$bin" \
    && joern-export "$bin" --repr "$REPR" --out "$json" ) >"$log" 2>&1 &

  ((running++))
  ((count++))
  if (( running >= JOBS )); then
    wait -n
    ((running--))
  fi
done < <(find "${IN_DIR}" -type f \( -name '*.c' -o -name '*.cpp' \) -print0)

# Đợi nốt job còn lại
wait
echo "✅ DONE: parsed ${count} files to ${OUT_DIR}"