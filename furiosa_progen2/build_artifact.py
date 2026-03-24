"""
ProGen2 모델을 RNGD NPU용 아티팩트로 빌드합니다.

furiosa-llm CLI(furiosa-llm build)는 별도 프로세스라서 register_progen2()가
적용되지 않으므로, Python에서 직접 ArtifactBuilder를 호출합니다.

사용법:
    python -m furiosa_progen2.build_artifact \
        --model-path /mnt/elice/datahub/models/progen2/results/ \
        --output /mnt/elice/datahub/models/progen2/artifact/
"""

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ProGen2 NPU 아티팩트 빌드")
    parser.add_argument("--model-path", type=str, required=True,
                        help="변환된 체크포인트 경로 (safetensors + config.json)")
    parser.add_argument("--output", type=str, required=True,
                        help="아티팩트 출력 경로")
    parser.add_argument("--tp", type=int, default=8,
                        help="텐서 병렬 크기 (기본: 8)")
    parser.add_argument("--prefill-buckets", type=str, default="1,128",
                        help="Prefill 버킷 (batch,seq_len 형식, 기본: 1,128)")
    parser.add_argument("--decode-buckets", type=str, default="1,2048",
                        help="Decode 버킷 (batch,seq_len 형식, 기본: 1,2048)")
    args = parser.parse_args()

    # 1. ProGen2 등록
    print("[1/3] Registering ProGen2 with furiosa ecosystem...")
    from furiosa_progen2.register import register_progen2
    register_progen2()

    # 2. ArtifactBuilder 로드
    print("[2/3] Initializing ArtifactBuilder...")
    from furiosa_llm.artifact.builder import ArtifactBuilder

    # 버킷 파싱
    def parse_bucket(s):
        parts = s.split(",")
        return (int(parts[0]), int(parts[1]))

    prefill = [parse_bucket(b) for b in args.prefill_buckets.split(";")]
    decode = [parse_bucket(b) for b in args.decode_buckets.split(";")]

    os.makedirs(args.output, exist_ok=True)

    # 3. 빌드 실행
    print("[3/3] Building artifact...")
    print(f"  Model: {args.model_path}")
    print(f"  Output: {args.output}")
    print(f"  TP: {args.tp}")
    print(f"  Prefill buckets: {prefill}")
    print(f"  Decode buckets: {decode}")

    try:
        builder = ArtifactBuilder(
            model=args.model_path,
            output_path=args.output,
            tensor_parallel_size=args.tp,
            prefill_buckets=prefill,
            decode_buckets=decode,
        )
        builder.build()
        print(f"\nBuild complete! Artifact saved to: {args.output}")
    except Exception as e:
        logger.error(f"Build failed: {e}")
        print(f"\n빌드 실패. 에러 내용을 공유해주세요.")
        raise


if __name__ == "__main__":
    main()
