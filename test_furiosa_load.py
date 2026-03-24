import sys
import os
import json
import shutil
import argparse
import tempfile
import traceback


def test_direct_load(model_path):
    """1-1. 원본 config로 furiosa-llm LLM 직접 로드"""
    print("=" * 60)
    print("[Test 1-1] furiosa-llm LLM 직접 로드 (원본 config)")
    print("=" * 60)

    try:
        from furiosa_llm import LLM
        llm = LLM(model_path)
        print("[SUCCESS] 로드 성공!")
        llm.shutdown()
        return True
    except ImportError:
        print("[SKIP] furiosa_llm 패키지 미설치")
        return None
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_load_with_llama_config(model_path):
    """1-2. architectures를 LlamaForCausalLM으로 변경 후 로드"""
    print()
    print("=" * 60)
    print("[Test 1-2] architectures → LlamaForCausalLM 변경 후 로드")
    print("=" * 60)

    tmpdir = None
    try:
        from furiosa_llm import LLM

        tmpdir = tempfile.mkdtemp(prefix="progen2_llama_")
        print(f"임시 디렉토리: {tmpdir}")

        for fname in os.listdir(model_path):
            src = os.path.join(model_path, fname)
            dst = os.path.join(tmpdir, fname)
            if fname == "config.json":
                with open(src, "r") as f:
                    config = json.load(f)
                original_arch = config.get("architectures", [])
                config["architectures"] = ["LlamaForCausalLM"]
                with open(dst, "w") as f:
                    json.dump(config, f, indent=2)
                print(f"  architectures 변경: {original_arch} → ['LlamaForCausalLM']")
            else:
                os.symlink(src, dst)

        llm = LLM(tmpdir)
        print("[SUCCESS] 로드 성공!")
        llm.shutdown()
        return True
    except ImportError:
        print("[SKIP] furiosa_llm 패키지 미설치")
        return None
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    finally:
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


def test_artifact_builder(model_path):
    """1-3. ArtifactBuilder로 빌드 시도"""
    print()
    print("=" * 60)
    print("[Test 1-3] ArtifactBuilder 빌드 시도")
    print("=" * 60)

    tmpdir = None
    try:
        from furiosa_llm.artifact.builder import ArtifactBuilder

        tmpdir = tempfile.mkdtemp(prefix="progen2_artifact_")
        builder = ArtifactBuilder(
            model_path,
            tensor_parallel_size=8,
            max_seq_len_to_capture=1024,
        )
        builder.build(tmpdir)
        print(f"[SUCCESS] 아티팩트 빌드 성공! → {tmpdir}")
        return True
    except ImportError:
        print("[SKIP] furiosa_llm.artifact.builder 패키지 미설치")
        return None
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    finally:
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


def print_model_info(model_path):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        print(f"[ERROR] config.json 없음: {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    print("모델 정보:")
    print(f"  architectures: {config.get('architectures', 'N/A')}")
    print(f"  model_type: {config.get('model_type', 'N/A')}")
    print(f"  n_embd: {config.get('n_embd', 'N/A')}")
    print(f"  n_layer: {config.get('n_layer', 'N/A')}")
    print(f"  n_head: {config.get('n_head', 'N/A')}")
    print(f"  vocab_size: {config.get('vocab_size', 'N/A')}")
    print()


def main():
    parser = argparse.ArgumentParser(description="furiosa-llm ProGen2 로드 테스트")
    parser.add_argument("--model", type=str, default="./progen2-medium",
                        help="ProGen2 체크포인트 경로")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        print(f"[ERROR] 모델 경로가 존재하지 않음: {model_path}")
        sys.exit(1)

    print_model_info(model_path)

    results = {}
    results["direct_load"] = test_direct_load(model_path)
    results["llama_config"] = test_load_with_llama_config(model_path)
    results["artifact_builder"] = test_artifact_builder(model_path)

    print()
    print("=" * 60)
    print("결과 요약")
    print("=" * 60)
    status_map = {True: "SUCCESS", False: "FAIL", None: "SKIP"}
    for name, result in results.items():
        print(f"  {name}: {status_map[result]}")

    if any(v is True for v in results.values()):
        print("\n→ furiosa-llm 로드 가능! 추가 변환 불필요.")
    elif all(v is None for v in results.values()):
        print("\n→ furiosa_llm 패키지 미설치. RNGD 서버에서 재실행 필요.")
    else:
        print("\n→ furiosa-llm 직접 로드 불가. Step 2 (torch.compiler) 또는 Step 3 (Llama 변환) 진행.")


if __name__ == "__main__":
    main()
