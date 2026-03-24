#!/bin/bash
set -e

MODEL_SRC="/mnt/elice/datahub/models/progen2/progen2-medium"
MODEL_OUT="/mnt/elice/datahub/results/progen2-medium"
FURIOSA_CONFIGS="/home/elicer/.local/lib/python3.10/site-packages/furiosa_llm/optimum/configs/llama"
LOCAL_FALLBACK="/home/elicer/progen2/progen2-medium"
BUILD_OUT="/home/elicer/output-progen2"

echo "=== Step 1: 재변환 (vocab_size=128) ==="
# convert_progen2_to_llama.py를 먼저 서버에 업로드해야 함
# 스크립트 위치는 환경에 맞게 수정
SCRIPT_DIR="${SCRIPT_DIR:-/home/elicer}"
python "$SCRIPT_DIR/convert_progen2_to_llama.py" \
  --model "$MODEL_SRC" \
  --output "$MODEL_OUT" \
  --no-verify

echo ""
echo "=== Step 2: furiosa config JSON 업데이트 ==="
# furiosa optimum configs 디렉토리에 우리 모델 config 복사 (vocab_size=128 반영)
python3 -c "
import json

# 모델 config 읽기
with open('$MODEL_OUT/config.json') as f:
    config = json.load(f)

# _progen2_source 제거 (furiosa matcher에서 불필요)
furiosa_config = {k: v for k, v in config.items() if not k.startswith('_')}

# furiosa optimum config로 저장
with open('$FURIOSA_CONFIGS/progen2_progen2-medium.json', 'w') as f:
    json.dump(furiosa_config, f, indent=2)
print(f'furiosa config 저장: $FURIOSA_CONFIGS/progen2_progen2-medium.json')
print(f'  vocab_size = {furiosa_config[\"vocab_size\"]}')

# 로컬 폴백 config도 업데이트
import os
os.makedirs('$LOCAL_FALLBACK', exist_ok=True)
with open('$LOCAL_FALLBACK/config.json', 'w') as f:
    json.dump(config, f, indent=2)
print(f'로컬 폴백 config 저장: $LOCAL_FALLBACK/config.json')
"

echo ""
echo "=== Step 3: furiosa-llm build ==="
cd /home/elicer
rm -rf "$BUILD_OUT"
furiosa-llm build "$MODEL_OUT" "$BUILD_OUT" -tp 8

echo ""
echo "=== 완료 ==="
echo "빌드 결과: $BUILD_OUT"
ls -la "$BUILD_OUT"
