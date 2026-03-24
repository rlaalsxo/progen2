import sys
import os
import argparse
import traceback

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from progen.modeling_progen import ProGenForCausalLM, ProGenBlock, ProGenAttention, ProGenMLP
from progen.configuration_progen import ProGenConfig


def get_compile_gm():
    from furiosa.torch.compiler import compile_gm
    return compile_gm


def test_linear(compile_gm):
    """가장 기본적인 nn.Linear 컴파일 테스트"""
    print("=" * 60)
    print("[Test 0] nn.Linear 단독 컴파일")
    print("=" * 60)

    try:
        model = nn.Linear(256, 256, bias=False).eval()
        dummy = torch.randn(1, 128, 256)

        exported = torch.export.export(model, (dummy,))
        compiled = compile_gm(exported.graph_module, (dummy,))
        print("[SUCCESS] nn.Linear 컴파일 성공")
        return True
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_mlp(compile_gm, config):
    """ProGenMLP 단독 컴파일 테스트"""
    print()
    print("=" * 60)
    print("[Test 1] ProGenMLP 컴파일")
    print("=" * 60)

    try:
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        mlp = ProGenMLP(inner_dim, config).eval()
        dummy = torch.randn(1, 32, config.n_embd)

        exported = torch.export.export(mlp, (dummy,))
        compiled = compile_gm(exported.graph_module, (dummy,))
        print("[SUCCESS] ProGenMLP 컴파일 성공")
        return True
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_attention(compile_gm, config):
    """ProGenAttention 단독 컴파일 테스트"""
    print()
    print("=" * 60)
    print("[Test 2] ProGenAttention 컴파일")
    print("=" * 60)

    try:
        attn = ProGenAttention(config).eval()
        dummy = torch.randn(1, 32, config.n_embd)

        exported = torch.export.export(attn, (dummy,))
        compiled = compile_gm(exported.graph_module, (dummy,))
        print("[SUCCESS] ProGenAttention 컴파일 성공")
        return True
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_block(compile_gm, config):
    """ProGenBlock 단일 레이어 컴파일 테스트"""
    print()
    print("=" * 60)
    print("[Test 3] ProGenBlock 단일 레이어 컴파일")
    print("=" * 60)

    try:
        block = ProGenBlock(config).eval()
        dummy = torch.randn(1, 32, config.n_embd)

        exported = torch.export.export(block, (dummy,))
        compiled = compile_gm(exported.graph_module, (dummy,))
        print("[SUCCESS] ProGenBlock 컴파일 성공")
        return True
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_full_model(compile_gm, model_path):
    """전체 ProGenForCausalLM 컴파일 테스트"""
    print()
    print("=" * 60)
    print("[Test 4] ProGenForCausalLM 전체 모델 컴파일")
    print("=" * 60)

    try:
        if os.path.exists(model_path):
            model = ProGenForCausalLM.from_pretrained(model_path).eval()
            print(f"  체크포인트에서 로드: {model_path}")
        else:
            print(f"  체크포인트 없음 ({model_path}), 랜덤 가중치로 테스트")
            config = ProGenConfig()
            config.n_layer = 2
            model = ProGenForCausalLM(config).eval()

        dummy = torch.randint(0, model.config.vocab_size, (1, 32))

        with torch.no_grad():
            exported = torch.export.export(model, (dummy,))
        compiled = compile_gm(exported.graph_module, (dummy,))
        print("[SUCCESS] 전체 모델 컴파일 성공")
        return True
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def test_torch_compile_backend(model_path):
    """torch.compile의 furiosa 백엔드 테스트"""
    print()
    print("=" * 60)
    print("[Test 5] torch.compile(backend='furiosa') 테스트")
    print("=" * 60)

    try:
        from furiosa.torch.backend import backend as furiosa_backend

        config = ProGenConfig()
        config.n_layer = 2
        config.vocab_size = 32
        model = ProGenForCausalLM(config).eval()

        compiled_model = torch.compile(model, backend=furiosa_backend)
        dummy = torch.randint(0, config.vocab_size, (1, 16))

        with torch.no_grad():
            output = compiled_model(dummy)
        print(f"[SUCCESS] torch.compile(backend='furiosa') 성공")
        print(f"  output logits shape: {output.logits.shape}")
        return True
    except ImportError:
        print("[SKIP] furiosa.torch.backend 미설치")
        return None
    except Exception as e:
        print(f"[FAIL] {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="furiosa.torch.compiler ProGen2 컴파일 테스트")
    parser.add_argument("--model", type=str, default="./progen2-medium",
                        help="ProGen2 체크포인트 경로")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)

    # furiosa.torch.compiler 임포트 확인
    try:
        compile_gm = get_compile_gm()
        print("furiosa.torch.compiler.compile_gm 로드 성공\n")
    except ImportError:
        print("furiosa.torch.compiler 패키지 미설치.")
        print("RNGD 서버에서 재실행 필요.\n")
        compile_gm = None

    # config 로드
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        config = ProGenConfig.from_pretrained(model_path)
        print(f"Config 로드: n_embd={config.n_embd}, n_layer={config.n_layer}, "
              f"n_head={config.n_head}, vocab_size={config.vocab_size}\n")
    else:
        print(f"Config 없음 ({config_path}), 기본값 사용\n")
        config = ProGenConfig()

    results = {}

    if compile_gm is not None:
        # 작은 단위부터 큰 단위로 테스트 (실패 지점 파악)
        results["linear"] = test_linear(compile_gm)
        results["mlp"] = test_mlp(compile_gm, config)
        results["attention"] = test_attention(compile_gm, config)
        results["block"] = test_block(compile_gm, config)
        results["full_model"] = test_full_model(compile_gm, model_path)
    else:
        for name in ["linear", "mlp", "attention", "block", "full_model"]:
            results[name] = None

    results["torch_compile"] = test_torch_compile_backend(model_path)

    # 결과 요약
    print()
    print("=" * 60)
    print("결과 요약")
    print("=" * 60)
    status_map = {True: "SUCCESS", False: "FAIL", None: "SKIP"}
    for name, result in results.items():
        print(f"  {name}: {status_map[result]}")

    if results.get("full_model") is True or results.get("torch_compile") is True:
        print("\n→ NPU 컴파일 가능! compile_gm 기반 추론 파이프라인 구축 진행.")
    elif any(v is True for v in results.values()):
        failed = [k for k, v in results.items() if v is False]
        passed = [k for k, v in results.items() if v is True]
        print(f"\n→ 부분 성공. 통과: {passed}, 실패: {failed}")
        print("  실패 연산자 확인 후 우회 방법 검토 필요.")
    elif all(v is None for v in results.values()):
        print("\n→ furiosa 패키지 미설치. RNGD 서버에서 재실행 필요.")
    else:
        print("\n→ NPU 컴파일 불가. Step 3 (Llama 호환 가중치 변환) 진행.")


if __name__ == "__main__":
    main()
