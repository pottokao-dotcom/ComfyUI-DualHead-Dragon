#!/usr/bin/env bash
# Quantize the Qwen-Image-2.1 text encoder (heretic by default) to a mixed-precision NVFP4 GGUF for the dual-head dragon.
# Every step uses the pinned vendor/llama.cpp (same commit the node runs). See docs/QUANTIZATION.md for the why.
#
#   TIER=t2 tools/quantize_te.sh
#   TIER=t3 PROTECT=<layers from the sensitivity sweep> tools/quantize_te.sh
#
# Env (defaults in brackets):
#   HF_REPO   [pottokao/Qwen-Image-2.1-Text-Encoder-Heretic]  source: HF transformers bf16 shards
#   NAME      [qwen3vl_8b_heretic]                            output file prefix
#   WORK      [$HOME/dhd_quant]                               downloads, BF16 GGUF, imatrix
#   OUT       [$WORK/out]                                     final GGUFs
#   TIER      [t2]   t1: all attn+ffn NVFP4 | t2: attn Q8_0, ffn NVFP4 | t3: t2 + PROTECT layers' ffn in Q8_0
#   PROTECT   []     comma list of layer ids whose ffn stays Q8_0 (t3), taken from the sensitivity sweep
#   RADIUS    [3]    nvfp4q scale-search radius (0 = byte-identical to ggml's quantizer)
#   CALIB_EXTRA []   extra text files appended to the calibration set
#   REF_CHECK [1]    run the precision check against the BF16 GGUF at the end
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
HF_REPO="${HF_REPO:-pottokao/Qwen-Image-2.1-Text-Encoder-Heretic}"
NAME="${NAME:-qwen3vl_8b_heretic}"
WORK="${WORK:-$HOME/dhd_quant}"; OUT="${OUT:-$WORK/out}"
TIER="${TIER:-t2}"; PROTECT="${PROTECT:-}"; RADIUS="${RADIUS:-3}"; REF_CHECK="${REF_CHECK:-1}"
PY="${PY:-python3}"
V="$HERE/vendor/llama.cpp"
BIN="$(ls -d "$HERE"/build/*/bin 2>/dev/null | head -1)"
[ -x "$BIN/llama-quantize" ] && [ -x "$BIN/llama-imatrix" ] || { echo "run: $PY build.py --with-tools"; exit 1; }
mkdir -p "$WORK" "$OUT"
export PYTHONPATH="$V/gguf-py:${PYTHONPATH:-}"
SRC="$WORK/hf_src"; BF16="$WORK/${NAME}_BF16.gguf"; IMAT="$WORK/${NAME}.imatrix.gguf"

echo "== 1/6 download $HF_REPO (bf16 shards only)"
$PY - <<EOF
from huggingface_hub import snapshot_download
snapshot_download("$HF_REPO", local_dir="$SRC", allow_patterns=["model-0000*", "*.json", "*.jinja", "LICENSE", "NOTICE", "README.md"])
EOF
( cd "$SRC" && sha256sum model-0000*.safetensors > "$WORK/source.sha256" )

echo "== 2/6 BF16 GGUF + f32 mmproj (vendor converter)"
[ -f "$BF16" ] || $PY "$V/convert_hf_to_gguf.py" "$SRC" --outtype bf16 --outfile "$BF16"
MMPROJ="$OUT/mmproj-${NAME}-f32.gguf"
[ -f "$MMPROJ" ] || $PY "$V/convert_hf_to_gguf.py" "$SRC" --mmproj --outtype f32 --outfile "$MMPROJ"

echo "== 3/6 calibration text + imatrix"
CALIB="$WORK/calib.txt"
cat "$HERE/prompts/pe_t2i.txt" "$HERE/prompts/pe_i2i.txt" ${CALIB_EXTRA:-} > "$CALIB"
[ -f "$IMAT" ] || "$BIN/llama-imatrix" -m "$BF16" -f "$CALIB" -o "$IMAT" -ngl 999 -c 512 2>&1 | tail -2

echo "== 4/6 layout with llama-quantize ($TIER)"
RULES=()
case "$TIER" in
  t1) RULES+=(--tensor-type 'blk\..*\.(attn_(q|k|v|output)|ffn_(gate|up|down))\.weight=nvfp4') ;;
  t2) RULES+=(--tensor-type 'blk\..*\.ffn_(gate|up|down)\.weight=nvfp4') ;;
  t3) [ -n "$PROTECT" ] || { echo "t3 needs PROTECT=..."; exit 1; }
      # first matching rule wins (llama-quant.cpp), so the protected layers must come first
      RULES+=(--tensor-type "blk\\.(${PROTECT//,/|})\\.ffn_(gate|up|down)\\.weight=q8_0")
      RULES+=(--tensor-type 'blk\..*\.ffn_(gate|up|down)\.weight=nvfp4') ;;
  *) echo "TIER must be t1/t2/t3"; exit 1 ;;
esac
TAG="$TIER${PROTECT:+-p${PROTECT//,/_}}"
DST="$OUT/${NAME}-NVFP4-${TAG}.gguf"
# base type Q8_0: attention (t2/t3), token_embd and output stay 8-bit
"$BIN/llama-quantize" --imatrix "$IMAT" "${RULES[@]}" "$BF16" "$DST" Q8_0 2>&1 | tail -1

echo "== 5/6 replace NVFP4 payloads with nvfp4q (imatrix-weighted scale search, radius $RADIUS)"
$PY "$HERE/tools/patch_nvfp4.py" "$BF16" "$DST" "$IMAT" "$RADIUS"

echo "== 6/6 record + check"
sha256sum "$DST" "$MMPROJ" | tee "$DST.sha256"
$PY - "$DST" <<'EOF'
import sys, collections, gguf
r = gguf.GGUFReader(sys.argv[1]); print("tensor types:", dict(collections.Counter(t.tensor_type.name for t in r.tensors)))
EOF
if [ "$REF_CHECK" = 1 ]; then
  $PY "$HERE/tools/bench_heretic.py" "$BF16" "$DST" 2>&1 | grep -E "^\{" || true
fi
echo "done -> $DST"
