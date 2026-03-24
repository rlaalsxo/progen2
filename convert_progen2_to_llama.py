import os
import sys
import json
import shutil
import argparse

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from progen.modeling_progen import ProGenForCausalLM


def split_qkv_weight(qkv_weight, num_heads, hidden_size, mp_num=8):
    """ProGen2의 interleaved qkv_proj 가중치를 Q, K, V로 분할.

    modeling_progen.py:161-164 로직 기반:
      qkv_split = qkv.reshape(..., (mp_num, -1))
      local_dim = head_dim * num_heads // mp_num
      query, value, key = torch.split(qkv_split, local_dim, dim=-1)
    순서가 Q, V, K임에 주의.
    """
    head_dim = hidden_size // num_heads
    local_dim = head_dim * num_heads // mp_num  # 192 for medium

    # (3*hidden_size, hidden_size) → (mp_num, 3*local_dim, hidden_size)
    reshaped = qkv_weight.reshape(mp_num, 3 * local_dim, hidden_size)

    q_chunks = reshaped[:, :local_dim, :]
    v_chunks = reshaped[:, local_dim:2*local_dim, :]
    k_chunks = reshaped[:, 2*local_dim:, :]

    q_weight = q_chunks.reshape(hidden_size, hidden_size)
    k_weight = k_chunks.reshape(hidden_size, hidden_size)
    v_weight = v_chunks.reshape(hidden_size, hidden_size)

    return q_weight, k_weight, v_weight


def convert_config(progen_config_path, output_dir):
    with open(progen_config_path, "r") as f:
        pg = json.load(f)

    hidden_size = pg["n_embd"]
    num_heads = pg["n_head"]
    head_dim = hidden_size // num_heads

    # furiosa의 create_input_ids()가 torch.randint(0, 128)을 사용하므로
    # vocab_size >= 128 필요 (원본 32 → 128로 패딩)
    FURIOSA_MIN_VOCAB_SIZE = 128
    padded_vocab_size = max(pg["vocab_size"], FURIOSA_MIN_VOCAB_SIZE)

    llama_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": padded_vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": pg["n_layer"],
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_heads,
        "intermediate_size": pg.get("n_inner") or 4 * hidden_size,
        "hidden_act": "silu",
        "max_position_embeddings": pg["n_positions"],
        "rms_norm_eps": pg["layer_norm_epsilon"],
        "rope_theta": 10000.0,
        "bos_token_id": pg["bos_token_id"],
        "eos_token_id": pg["eos_token_id"],
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "use_cache": True,
        "_progen2_source": {
            "original_vocab_size": pg["vocab_size"],
            "rotary_dim": pg.get("rotary_dim"),
            "head_dim": head_dim,
            "note": "vocab_size padded for furiosa compatibility. rotary_dim < head_dim: partial rotation in original."
        }
    }

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(llama_config, f, indent=2)
    print(f"Config 저장: {config_path}")
    return llama_config


def pad_weight(weight, target_rows, dim=0):
    """가중치를 target_rows 크기로 제로패딩."""
    current_rows = weight.shape[dim]
    if current_rows >= target_rows:
        return weight
    pad_size = [0] * (2 * weight.ndim)
    pad_size[-(2 * dim + 1)] = target_rows - current_rows
    return torch.nn.functional.pad(weight, pad_size)


def convert_weights(model_path, output_dir, llama_config):
    print("ProGen2 모델 로딩...")
    model = ProGenForCausalLM.from_pretrained(model_path)
    sd = model.state_dict()

    hidden_size = llama_config["hidden_size"]
    num_heads = llama_config["num_attention_heads"]
    num_layers = llama_config["num_hidden_layers"]
    intermediate_size = llama_config["intermediate_size"]
    vocab_size = llama_config["vocab_size"]

    new_sd = {}

    # Embedding (원본 vocab_size → padded vocab_size로 제로패딩)
    embed = sd["transformer.wte.weight"].clone()
    new_sd["model.embed_tokens.weight"] = pad_weight(embed, vocab_size)
    print(f"  embed_tokens: {embed.shape} → {new_sd['model.embed_tokens.weight'].shape}")

    # Final norm
    new_sd["model.norm.weight"] = sd["transformer.ln_f.weight"].clone()
    print(f"  final norm: {new_sd['model.norm.weight'].shape}")

    # LM head (원본 vocab_size → padded vocab_size로 제로패딩)
    lm = sd["lm_head.weight"].clone()
    new_sd["lm_head.weight"] = pad_weight(lm, vocab_size)
    print(f"  lm_head: {lm.shape} → {new_sd['lm_head.weight'].shape}")

    for i in range(num_layers):
        prefix_pg = f"transformer.h.{i}"
        prefix_ll = f"model.layers.{i}"

        # Input layernorm (LayerNorm weight → RMSNorm weight, bias 버림)
        new_sd[f"{prefix_ll}.input_layernorm.weight"] = sd[f"{prefix_pg}.ln_1.weight"].clone()

        # Post-attention layernorm (ProGen2에는 없음 → 1.0 초기화)
        new_sd[f"{prefix_ll}.post_attention_layernorm.weight"] = torch.ones(hidden_size)

        # QKV 분할
        qkv_weight = sd[f"{prefix_pg}.attn.qkv_proj.weight"]
        q, k, v = split_qkv_weight(qkv_weight, num_heads, hidden_size)
        new_sd[f"{prefix_ll}.self_attn.q_proj.weight"] = q
        new_sd[f"{prefix_ll}.self_attn.k_proj.weight"] = k
        new_sd[f"{prefix_ll}.self_attn.v_proj.weight"] = v

        # Output projection
        new_sd[f"{prefix_ll}.self_attn.o_proj.weight"] = sd[f"{prefix_pg}.attn.out_proj.weight"].clone()

        # MLP: up_proj ← fc_in.weight, down_proj ← fc_out.weight, gate_proj ← 랜덤
        new_sd[f"{prefix_ll}.mlp.up_proj.weight"] = sd[f"{prefix_pg}.mlp.fc_in.weight"].clone()
        new_sd[f"{prefix_ll}.mlp.down_proj.weight"] = sd[f"{prefix_pg}.mlp.fc_out.weight"].clone()
        gate = torch.empty(intermediate_size, hidden_size)
        torch.nn.init.kaiming_uniform_(gate)
        new_sd[f"{prefix_ll}.mlp.gate_proj.weight"] = gate

        if (i + 1) % 5 == 0 or i == num_layers - 1:
            print(f"  layer {i}/{num_layers-1} 변환 완료")

    # safetensors로 저장
    output_path = os.path.join(output_dir, "model.safetensors")
    save_file(new_sd, output_path)
    print(f"가중치 저장: {output_path}")

    return new_sd


def verify_conversion(original_path, new_sd, num_heads, hidden_size):
    print("\n변환 검증...")
    model = ProGenForCausalLM.from_pretrained(original_path)
    sd = model.state_dict()

    orig_vocab = sd["transformer.wte.weight"].shape[0]

    checks = [
        # 패딩된 가중치는 원본 범위만 비교
        ("embed_tokens", sd["transformer.wte.weight"], new_sd["model.embed_tokens.weight"][:orig_vocab]),
        ("lm_head", sd["lm_head.weight"], new_sd["lm_head.weight"][:orig_vocab]),
        ("final_norm", sd["transformer.ln_f.weight"], new_sd["model.norm.weight"]),
        ("layer0_o_proj", sd["transformer.h.0.attn.out_proj.weight"], new_sd["model.layers.0.self_attn.o_proj.weight"]),
        ("layer0_input_ln", sd["transformer.h.0.ln_1.weight"], new_sd["model.layers.0.input_layernorm.weight"]),
    ]

    all_pass = True
    for name, orig, converted in checks:
        cos_sim = torch.nn.functional.cosine_similarity(
            orig.flatten().unsqueeze(0).float(),
            converted.flatten().unsqueeze(0).float()
        ).item()
        status = "PASS" if abs(cos_sim - 1.0) < 1e-6 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {name}: cosine_sim={cos_sim:.8f} [{status}]")

    # QKV 분할 검증: 원본 qkv로 복원 후 비교
    qkv_orig = sd["transformer.h.0.attn.qkv_proj.weight"]
    q = new_sd["model.layers.0.self_attn.q_proj.weight"]
    k = new_sd["model.layers.0.self_attn.k_proj.weight"]
    v = new_sd["model.layers.0.self_attn.v_proj.weight"]

    mp_num = 8
    local_dim = hidden_size // mp_num
    q_chunks = q.reshape(mp_num, local_dim, hidden_size)
    v_chunks = v.reshape(mp_num, local_dim, hidden_size)
    k_chunks = k.reshape(mp_num, local_dim, hidden_size)
    reconstructed = torch.cat([q_chunks, v_chunks, k_chunks], dim=1).reshape(-1, hidden_size)

    cos_sim = torch.nn.functional.cosine_similarity(
        qkv_orig.flatten().unsqueeze(0).float(),
        reconstructed.flatten().unsqueeze(0).float()
    ).item()
    status = "PASS" if abs(cos_sim - 1.0) < 1e-6 else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"  qkv_roundtrip: cosine_sim={cos_sim:.8f} [{status}]")

    return all_pass


def copy_tokenizer(source_dir, output_dir):
    """프로젝트 루트의 tokenizer.json 복사"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    tokenizer_src = os.path.join(project_root, "tokenizer.json")

    if os.path.exists(tokenizer_src):
        shutil.copy2(tokenizer_src, os.path.join(output_dir, "tokenizer.json"))
        print(f"토크나이저 복사: {tokenizer_src}")
    else:
        print(f"[WARNING] tokenizer.json 없음: {tokenizer_src}")


def main():
    parser = argparse.ArgumentParser(description="ProGen2 → LlamaForCausalLM 가중치 변환")
    parser.add_argument("--model", type=str, default="/mnt/elice/datahub/models/progen2/progen2-medium",
                        help="ProGen2 체크포인트 경로")
    parser.add_argument("--output", type=str, default="./progen2-medium-llama",
                        help="변환된 Llama 모델 출력 경로")
    parser.add_argument("--no-verify", action="store_true", default=False,
                        help="변환 후 검증 건너뛰기 (메모리 절약)")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    output_dir = os.path.abspath(args.output)

    if not os.path.exists(model_path):
        print(f"[ERROR] 모델 경로 없음: {model_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print(f"입력: {model_path}")
    print(f"출력: {output_dir}")
    print()

    # 1. Config 변환
    config_path = os.path.join(model_path, "config.json")
    llama_config = convert_config(config_path, output_dir)
    print()

    # 2. 가중치 변환
    new_sd = convert_weights(model_path, output_dir, llama_config)
    print()

    # 3. 토크나이저 복사
    copy_tokenizer(model_path, output_dir)
    print()

    # 4. 검증
    if not args.no_verify:
        all_pass = verify_conversion(
            model_path, new_sd,
            llama_config["num_attention_heads"],
            llama_config["hidden_size"]
        )
        print(f"\n검증 결과: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    print(f"\n변환 완료: {output_dir}")
    print("다음 단계:")
    print(f"  1. PyTorch 검증: python -c \"from transformers import LlamaForCausalLM; m = LlamaForCausalLM.from_pretrained('{args.output}'); print('OK')\"")
    print(f"  2. furiosa-llm 빌드: furiosa-llm build {args.output} ./output-progen2 -tp 8")
    print(f"  3. 미세조정 필요 (MLP gate_proj 랜덤 초기화 + 구조 변경)")


if __name__ == "__main__":
    main()
