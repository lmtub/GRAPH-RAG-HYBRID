#!/usr/bin/env bash
set -euo pipefail

# ========= Config =========
IN_DIR="${IN_DIR:-/app/data/raw/devign_c}"
OUT_DIR="${OUT_DIR:-/app/data/cpg}"

RESUME="${RESUME:-1}"       # 1: skip mẫu đã có JSON
JOBS="${JOBS:-8}"           # số job song song
OVERWRITE="${OVERWRITE:-0}" # 1: xóa thư mục .cpg14 cũ trước khi export

CONVERTER="${CONVERTER:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/convert_dot_to_cpg14_json.py}"
export CONVERTER

export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Xms512m -Xmx6g}"

# ========= Func =========
parse_one() {
  local sample="$1"
  local src="${IN_DIR}/${sample}.c"
  local bin="${OUT_DIR}/${sample}.bin"
  local outd="${OUT_DIR}/${sample}.cpg14"

  echo "==== ${sample} ===="

  [[ -f "${src}" ]] || { echo "[SKIP] Không thấy source: ${src}"; return 0; }

  if [[ "${RESUME}" == "1" && -s "${outd}/nodes.json" && -s "${outd}/edges.json" ]]; then
    echo "[SKIP] ${sample} (đã có JSON)"; return 0
  fi

  # joern-export yêu cầu outd CHƯA tồn tại
  if [[ "${OVERWRITE}" == "1" && -d "${outd}" ]]; then
    rm -rf "${outd}"
  fi
  mkdir -p "${OUT_DIR}"
  # KHÔNG tạo sẵn ${outd} trước export (để tránh “already exists”)
  [[ -d "${outd}" ]] && rm -rf "${outd}"

  echo "[parse] ${src} -> ${bin}"
  set +o pipefail
  joern-parse --language c "${src}" -o "${bin}" 2>&1 | tee "${OUT_DIR}/${sample}.parse.log"
  local rc_parse=${PIPESTATUS[0]}
  set -o pipefail
  [[ ${rc_parse} -eq 0 && -s "${bin}" ]] || { echo "[ERR] Parse lỗi"; return 1; }

  echo "[export:cpg14 DOT] ${bin} -> ${outd}"
  set +o pipefail
  joern-export --repr cpg14 --out "${outd}" "${bin}" 2>&1 | tee "${outd}.export.log"
  local rc_export=${PIPESTATUS[0]}
  set -o pipefail
  [[ ${rc_export} -eq 0 ]] || { echo "[ERR] Export lỗi"; return 2; }

  # Convert DOT -> JSON
  [[ -f "${CONVERTER}" ]] || { echo "[ERR] Thiếu converter: ${CONVERTER}"; return 3; }
  set +o pipefail
  python3 "${CONVERTER}" "${outd}" 2>&1 | tee "${outd}.convert.log"
  local rc_conv=${PIPESTATUS[0]}
  set -o pipefail
  [[ ${rc_conv} -eq 0 ]] || { echo "[ERR] Converter lỗi"; return 4; }

  if [[ -s "${outd}/nodes.json" && -s "${outd}/edges.json" ]]; then
    echo "[OK] ${sample}"
  else
    echo "[ERR] Thiếu nodes.json/edges.json trong ${outd}"; return 5
  fi
}

export -f parse_one
export IN_DIR OUT_DIR RESUME JOBS OVERWRITE JAVA_TOOL_OPTIONS CONVERTER

# ========= Run =========
if [[ $# -ge 1 ]]; then
  printf "%s\n" "$@" | xargs -I{} -P "${JOBS}" bash -c 'parse_one "$@"' _ {}
else
  find "${IN_DIR}" -maxdepth 1 -type f -name "*.c" -printf "%f\n" \
    | sed 's/\.c$//' \
    | xargs -I{} -P "${JOBS}" bash -c 'parse_one "$@"' _ {}
fi
