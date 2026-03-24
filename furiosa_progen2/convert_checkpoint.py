"""
ProGen2 체크포인트를 furiosa-models 호환 safetensors 포맷으로 변환합니다.

주요 변환:
1. PyTorch .bin → safetensors
2. 가중치 이름 매핑 (ProGen2 → furiosa 패턴)
3. QKV 결합 프로젝션 분리 (인터리브 해제 포함)
4. config.json 생성

사용법:
    python -m furiosa_progen2.convert_checkpoint \
        --ckpt ./checkpoints/progen2-medium/ \
        --output ./checkpoints/progen2-medium-furiosa/
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


# progen2-medium 기본 설정
MODEL_CONFIGS = {
    "progen2-small": {
        "vocab_size": 50400,
        "n_embd": 1024,
        "n_layer": 24,
        "n_head": 16,
        "n_inner": 4096,
        "n_positions": 2048,
        "rotary_dim": 64,
    },
    "progen2-medium": {
        "vocab_size": 50400,
        "n_embd": 1536,
        "n_layer": 27,
        "n_head": 16,
        "n_inner": 6144,
        "n_positions": 2048,
        "rotary_dim": 64,
    },
    "progen2-base": {
        "vocab_size": 50400,
        "n_embd": 1536,
        "n_layer": 27,
        "n_head": 16,
        "n_inner": 6144,
        "n_positions": 2048,
        "rotary_dim": 64,
    },
    "progen2-oas": {
        "vocab_size": 50400,
        "n_embd": 1536,
        "n_layer": 27,
        "n_head": 16,
        "n_inner": 6144,
        "n_positions": 2048,
        "rotary_dim": 64,
    },
    "progen2-large": {
        "vocab_size": 50400,
        "n_embd": 4096,
        "n_layer": 28,
        "n_head": 16,
        "n_inner": 16384,
        "n_positions": 2048,
        "rotary_dim": 64,
    },
    "progen2-BFD90": {
        "vocab_size": 50400,
        "n_embd": 4096,
        "n_layer": 28,
        "n_head": 16,
        "n_inner": 16384,
        "n_positions": 2048,
        "rotary_dim": 64,
    },
    "progen2-xlarge": {
        "vocab_size": 50400,
        "n_embd": 6144,
        "n_layer": 44,
        "n_head": 16,
        "n_inner": 24576,
        "n_positions": 2048,
        "rotary_dim": 64,
    },
}


def split_qkv_weight(qkv_weight: torch.Tensor, n_head: int, mp_num: int = 8):
    """
    ProGen2의 인터리브된 QKV 결합 가중치를 분리합니다.

    ProGen2 QKV 레이아웃 (mp_num=8 인터리브):
        qkv_proj.weight: [3*H, H]
        reshape → [mp_num, 3*local_dim, H]
        split → Q[mp_num, local_dim, H], V[mp_num, local_dim, H], K[mp_num, local_dim, H]

    주의: ProGen2의 split 순서는 Q, V, K (Q, K, V가 아님!)

    Args:
        qkv_weight: [3*hidden_size, hidden_size] 형태의 가중치
        n_head: 어텐션 헤드 수
        mp_num: 모델 병렬화 분할 수 (기본 8, TPU용)

    Returns:
        (q_weight, k_weight, v_weight): 각각 [hidden_size, hidden_size] 형태
    """
    total_dim = qkv_weight.shape[0]  # 3 * hidden_size
    hidden_size = qkv_weight.shape[1]
    head_dim = hidden_size // n_head
    local_dim = head_dim * n_head // mp_num  # hidden_size // mp_num

    # [3*H, H] → [mp_num, 3*local_dim, H]
    qkv_reshaped = qkv_weight.reshape(mp_num, 3 * local_dim, hidden_size)

    # split along dim=1: Q, V, K (ProGen2 순서!)
    q_chunks = qkv_reshaped[:, :local_dim, :]               # [mp_num, local_dim, H]
    v_chunks = qkv_reshaped[:, local_dim:2*local_dim, :]     # [mp_num, local_dim, H]
    k_chunks = qkv_reshaped[:, 2*local_dim:3*local_dim, :]   # [mp_num, local_dim, H]

    # [mp_num, local_dim, H] → [H, H]
    q_weight = q_chunks.reshape(hidden_size, hidden_size)
    k_weight = k_chunks.reshape(hidden_size, hidden_size)
    v_weight = v_chunks.reshape(hidden_size, hidden_size)

    return q_weight, k_weight, v_weight


FURIOSA_VOCAB_PADDING = 64  # furiosa-models DEFAULT_VOCAB_PADDING_SIZE


def pad_vocab_size(vocab_size: int) -> int:
    """furiosa-models의 VocabEmbeddingLayer 패딩 규칙에 맞게 vocab_size를 올림합니다."""
    return ((vocab_size + FURIOSA_VOCAB_PADDING - 1) // FURIOSA_VOCAB_PADDING) * FURIOSA_VOCAB_PADDING


def pad_embedding_weight(weight: torch.Tensor, padded_vocab_size: int) -> torch.Tensor:
    """임베딩/lm_head 가중치를 padded_vocab_size로 제로패딩합니다."""
    current_size = weight.shape[0]
    if current_size >= padded_vocab_size:
        return weight
    pad = torch.zeros(padded_vocab_size - current_size, weight.shape[1], dtype=weight.dtype)
    return torch.cat([weight, pad], dim=0)


def convert_state_dict(state_dict: dict, n_head: int, n_layer: int) -> dict:
    """
    ProGen2 state_dict를 furiosa-models 호환 이름으로 변환합니다.

    변환 규칙:
        transformer.wte.weight              → model.wte.weight
        transformer.h.{i}.ln_1.weight/bias  → model.layers.{i}.ln_1.weight/bias
        transformer.h.{i}.attn.qkv_proj.weight → model.layers.{i}.self_attn.{q,k,v}_proj.weight
        transformer.h.{i}.attn.out_proj.weight → model.layers.{i}.self_attn.out_proj.weight
        transformer.h.{i}.mlp.fc_in.*       → model.layers.{i}.mlp.fc_in.*
        transformer.h.{i}.mlp.fc_out.*      → model.layers.{i}.mlp.fc_out.*
        transformer.ln_f.weight/bias        → model.ln_f.weight/bias
        lm_head.weight                      → lm_head.weight
    """
    new_state_dict = {}

    for key, value in state_dict.items():
        # 버퍼는 건너뛰기 (causal mask, masked_bias)
        if "attn.bias" in key or "attn.masked_bias" in key:
            continue

        new_key = None

        # 토큰 임베딩 (vocab_size 패딩 적용)
        if key == "transformer.wte.weight":
            new_key = "model.wte.weight"
            padded_size = pad_vocab_size(value.shape[0])
            if padded_size != value.shape[0]:
                print(f"  Padding wte.weight: [{value.shape[0]}, {value.shape[1]}] → [{padded_size}, {value.shape[1]}]")
                value = pad_embedding_weight(value, padded_size)

        # 최종 LayerNorm
        elif key == "transformer.ln_f.weight":
            new_key = "model.ln_f.weight"
        elif key == "transformer.ln_f.bias":
            new_key = "model.ln_f.bias"

        # LM Head (vocab_size 패딩 적용)
        elif key == "lm_head.weight":
            new_key = "lm_head.weight"
            padded_size = pad_vocab_size(value.shape[0])
            if padded_size != value.shape[0]:
                print(f"  Padding lm_head.weight: [{value.shape[0]}, {value.shape[1]}] → [{padded_size}, {value.shape[1]}]")
                value = pad_embedding_weight(value, padded_size)

        # 레이어별 가중치
        elif key.startswith("transformer.h."):
            parts = key.split(".")
            layer_idx = parts[2]
            rest = ".".join(parts[3:])

            if rest == "ln_1.weight":
                new_key = f"model.layers.{layer_idx}.ln_1.weight"
            elif rest == "ln_1.bias":
                new_key = f"model.layers.{layer_idx}.ln_1.bias"
            elif rest == "attn.qkv_proj.weight":
                # QKV 분리
                q, k, v = split_qkv_weight(value, n_head)
                new_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = q
                new_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = k
                new_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = v
                continue  # 이미 추가했으므로 아래 할당 건너뛰기
            elif rest == "attn.out_proj.weight":
                new_key = f"model.layers.{layer_idx}.self_attn.out_proj.weight"
            elif rest == "mlp.fc_in.weight":
                new_key = f"model.layers.{layer_idx}.mlp.fc_in.weight"
            elif rest == "mlp.fc_in.bias":
                new_key = f"model.layers.{layer_idx}.mlp.fc_in.bias"
            elif rest == "mlp.fc_out.weight":
                new_key = f"model.layers.{layer_idx}.mlp.fc_out.weight"
            elif rest == "mlp.fc_out.bias":
                new_key = f"model.layers.{layer_idx}.mlp.fc_out.bias"
            else:
                print(f"  [SKIP] Unknown layer key: {key}")
                continue
        else:
            print(f"  [SKIP] Unknown key: {key}")
            continue

        if new_key is not None:
            new_state_dict[new_key] = value

    return new_state_dict


def create_config_json(model_name: str, output_dir: str, actual_vocab_size: int = None):
    """HuggingFace 호환 config.json을 생성합니다.

    Args:
        actual_vocab_size: 체크포인트에서 감지한 실제 vocab_size. None이면 MODEL_CONFIGS 값 사용.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[model_name]
    raw_vocab_size = actual_vocab_size if actual_vocab_size is not None else cfg["vocab_size"]
    vocab_size = pad_vocab_size(raw_vocab_size)
    if vocab_size != raw_vocab_size:
        print(f"  Padding vocab_size: {raw_vocab_size} → {vocab_size} (furiosa padding={FURIOSA_VOCAB_PADDING})")

    config = {
        "architectures": ["ProGenForCausalLM"],
        "model_type": "progen",
        "vocab_size": vocab_size,
        "n_positions": cfg["n_positions"],
        "n_ctx": cfg["n_positions"],
        "n_embd": cfg["n_embd"],
        "n_layer": cfg["n_layer"],
        "n_head": cfg["n_head"],
        "n_inner": cfg["n_inner"],
        "rotary_dim": cfg["rotary_dim"],
        "activation_function": "gelu_new",
        "layer_norm_epsilon": 1e-05,
        "initializer_range": 0.02,
        "scale_attn_weights": True,
        "use_cache": True,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config saved: {config_path}")
    return config


def find_checkpoint_files(ckpt_dir: str):
    """체크포인트 디렉토리에서 모델 파일을 찾습니다."""
    ckpt_path = Path(ckpt_dir)

    # pytorch_model.bin (단일 파일)
    single = ckpt_path / "pytorch_model.bin"
    if single.exists():
        return [str(single)]

    # pytorch_model-*.bin (샤드 파일)
    shards = sorted(ckpt_path.glob("pytorch_model-*.bin"))
    if shards:
        return [str(s) for s in shards]

    # 직접 .bin 파일
    bins = sorted(ckpt_path.glob("*.bin"))
    if bins:
        return [str(b) for b in bins]

    raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")


def load_state_dict(ckpt_files: list) -> dict:
    """체크포인트 파일(들)에서 state_dict를 로드합니다."""
    state_dict = {}
    for f in ckpt_files:
        print(f"  Loading: {f}")
        shard = torch.load(f, map_location="cpu", weights_only=True)
        state_dict.update(shard)
    return state_dict


def main():
    parser = argparse.ArgumentParser(description="ProGen2 체크포인트를 furiosa 호환 safetensors로 변환")
    parser.add_argument("--model", type=str, default="progen2-medium",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="모델 이름 (기본: progen2-medium)")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="원본 체크포인트 디렉토리 경로")
    parser.add_argument("--output", type=str, required=True,
                        help="변환된 모델 출력 디렉토리")
    parser.add_argument("--copy-tokenizer", type=str, default=None,
                        help="tokenizer.json 경로 (출력에 복사)")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]

    os.makedirs(args.output, exist_ok=True)

    # 1. 체크포인트 로드
    print("[1/4] Loading checkpoint...")
    ckpt_files = find_checkpoint_files(args.ckpt)
    state_dict = load_state_dict(ckpt_files)
    print(f"  Loaded {len(state_dict)} parameters")

    # 체크포인트에서 실제 vocab_size 감지
    actual_vocab_size = state_dict["transformer.wte.weight"].shape[0]
    print(f"  Detected vocab_size from checkpoint: {actual_vocab_size}")

    # 2. config.json 생성 (실제 vocab_size 사용)
    print("[2/4] Creating config.json...")
    create_config_json(args.model, args.output, actual_vocab_size=actual_vocab_size)

    # 3. 가중치 변환
    print("[3/4] Converting weights...")
    new_state_dict = convert_state_dict(
        state_dict,
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"],
    )
    print(f"  Converted to {len(new_state_dict)} parameters")

    # 검증: 예상 파라미터 수 확인
    n_layer = cfg["n_layer"]
    expected_keys = (
        1                    # model.wte.weight
        + 2                  # model.ln_f.weight/bias
        + 1                  # lm_head.weight
        + n_layer * (
            2                # ln_1.weight/bias
            + 3              # q_proj, k_proj, v_proj
            + 1              # out_proj
            + 2              # fc_in.weight/bias
            + 2              # fc_out.weight/bias
        )
    )
    actual_keys = len(new_state_dict)
    print(f"  Expected {expected_keys} keys, got {actual_keys}")
    if actual_keys != expected_keys:
        print(f"  WARNING: Key count mismatch!")

    # 4. safetensors로 저장
    print("[4/4] Saving as safetensors...")
    # safetensors는 non-contiguous 텐서를 허용하지 않음 (tied weights 등)
    new_state_dict = {k: v.contiguous() for k, v in new_state_dict.items()}
    output_path = os.path.join(args.output, "model.safetensors")
    save_file(new_state_dict, output_path)
    print(f"  Saved: {output_path}")

    # 선택: tokenizer 복사
    if args.copy_tokenizer and os.path.exists(args.copy_tokenizer):
        dest = os.path.join(args.output, "tokenizer.json")
        shutil.copy2(args.copy_tokenizer, dest)
        print(f"  Tokenizer copied: {dest}")

    # 요약
    total_params = sum(v.numel() for v in new_state_dict.values())
    total_bytes = sum(v.numel() * v.element_size() for v in new_state_dict.values())
    print(f"\nDone! Model: {args.model}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Size: {total_bytes / 1024**3:.2f} GB")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
