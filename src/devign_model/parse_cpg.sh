#!/usr/bin/env bash
set -euo pipefail

# ---- Paths (auto) ----
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../ && pwd)"
IN_DIR="${ROOT_DIR}/data/raw/devign_c"
OUT_DIR="${ROOT_DIR}/data/cpg"
REPR="cpg14"
JOBS=1
LIMIT=0

mkdir -p "${OUT_DIR}" "${OUT_DIR}/logs"

count=0; running=0
find "${IN_DIR}" -type f \( -name '*.c' -o -name '*.cpp' \) -print0 | \
while IFS= read -r -d '' f; do
  (( LIMIT>0 && count>=LIMIT )) && break
  base="$(basename "${f}")"; stem="${base%.*}"
  bin="${OUT_DIR}/${stem}.bin"; json="${OUT_DIR}/${stem}.json"; log="${OUT_DIR}/logs/${stem}.log"

  ( joern-parse "${f}" --out "${bin}" && joern-export "${bin}" --repr "${REPR}" --out "${json}" ) \
    >"${log}" 2>&1 &

  ((running++)); ((count++))
  if (( running >= JOBS )); then wait -n; ((running--)); fi
done
wait
echo "DONE: parsed ${count} files to ${OUT_DIR}"

while getopts ":i:o:r:j:n:h" opt; do
  case $opt in
    i) IN_DIR="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    r) REPR="$OPTARG" ;;
    j) JOBS="$OPTARG" ;;
    n) LIMIT="$OPTARG" ;;
    h|*) usage ;;
  esac
done

mkdir -p "$OUT_DIR"

# --- Hàm xử lý 1 file ---
process_one() {
  local f="$1"
  local base="$(basename "$f")"
  local stem="${base%.*}"
  local bin="$OUT_DIR/$stem.bin"
  local json="$OUT_DIR/$stem.json"
  local log="$OUT_DIR/$stem.log"

  if [[ -s "$json" ]]; then
    echo "✓ Skip $stem (đã có JSON)" ; return 0
  fi

  {
    echo "→ Parse $base"
    # Tạo .bin
    joern-parse "$f" --out "$bin"
    # Export JSON (v1.1.230 KHÔNG dùng --input; truyền .bin trực tiếp)
    joern-export "$bin" --repr "$REPR" --out "$json"
    echo "✔ Done $stem"
  } >"$log" 2>&1 || {
    echo "✗ Fail $stem (xem log: $log)"
    rm -f "$bin"
    return 1
  }
}

# --- Duyệt file an toàn với tên có khoảng trắng ---
count=0
running=0

while IFS= read -r -d '' f; do
  ((count++))
  # Giới hạn số file nếu cần
  if [[ "$LIMIT" -gt 0 && "$count" -gt "$LIMIT" ]]; then break; fi

  if [[ "$JOBS" -gt 1 ]]; then
    process_one "$f" &
    ((running++))
    # Giữ số job nền tối đa = JOBS
    if [[ "$running" -ge "$JOBS" ]]; then
      wait -n
      ((running--))
    fi
  else
    process_one "$f"
  fi
done < <(find "$IN_DIR" -type f \( -name "*.c" -o -name "*.cpp" \) -print0)

# Chờ job nền (nếu có)
wait || true
echo "✅ Done. Output ở: $OUT_DIR"
