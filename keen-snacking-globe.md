# ProGen2 → FuriosaAI RNGD NPU 포팅: Step 3 Llama 변환

## Context

Step 1-2 검증 완료. 결과:
- **Step 1 (furiosa-llm 로드)**: 전부 실패. `model_type: "progen"`이 transformers `CONFIG_MAPPING`에 미등록
- **Step 2 (torch.compiler)**: 전부 실패. `nn.Linear` 조차 `UnsupportedOpError` (tensor import 실패)

→ **Step 3: ProGen2 가중치를 LlamaForCausalLM 형식으로 변환** 이 유일한 경로.

**타겟**: `progen2-medium` (764M, n_embd=1536, n_layer=27, n_head=16, vocab_size=32, rotary_dim=48)
**체크포인트**: `/mnt/elice/datahub/models/progen2/progen2-medium/`

---

## 가중치 변환 스크립트

**생성 파일**: `convert_progen2_to_llama.py`

### 1. Config 변환 (ProGenConfig → LlamaConfig)

```
vocab_size: 32            → vocab_size: 32
n_embd: 1536              → hidden_size: 1536
n_layer: 27               → num_hidden_layers: 27
n_head: 16                → num_attention_heads: 16
                          → num_key_value_heads: 16 (MHA)
4 * n_embd = 6144         → intermediate_size: 6144
rotary_dim: 48            → (아래 RoPE 섹션 참조)
layer_norm_epsilon: 1e-5  → rms_norm_eps: 1e-5
                          → hidden_act: "silu"
n_positions: 1024         → max_position_embeddings: 1024
model_type: "progen"      → model_type: "llama"
architectures             → ["LlamaForCausalLM"]
bos_token_id: 1           → bos_token_id: 1
eos_token_id: 2           → eos_token_id: 2
```

### 2. 가중치 매핑

**직접 전이**:
| Llama | ProGen2 | 비고 |
|---|---|---|
| `model.embed_tokens.weight` | `transformer.wte.weight` | 직접 복사 |
| `model.layers[i].self_attn.o_proj.weight` | `transformer.h[i].attn.out_proj.weight` | 직접 복사 |
| `model.norm.weight` | `transformer.ln_f.weight` | 복사 (bias 버림) |
| `lm_head.weight` | `lm_head.weight` | 직접 복사 |
| `model.layers[i].input_layernorm.weight` | `transformer.h[i].ln_1.weight` | 복사 (bias 버림) |

**QKV 분할 전이** (핵심):

`modeling_progen.py:161-164`의 로직:
```python
qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))  # mp_num=8
local_dim = head_dim * num_heads // mp_num                # = 96*16//8 = 192
query, value, key = torch.split(qkv_split, local_dim, dim=-1)  # 순서: Q, V, K
```

변환 시: `qkv_proj.weight` (shape: `[4608, 1536]`, 즉 `3*1536`) 에서:
```python
qkv = weight.reshape(mp_num, 3 * local_dim, hidden_size)  # (8, 576, 1536)
# 각 chunk에서 local_dim=192 단위로 Q, V, K 분리
q = qkv[:, :192, :]           → reshape → q_proj.weight (1536, 1536)
v = qkv[:, 192:384, :]        → reshape → v_proj.weight (1536, 1536)
k = qkv[:, 384:576, :]        → reshape → k_proj.weight (1536, 1536)
```

**초기화 필요 (전이 불가)**:
| Llama | 초기화 | 이유 |
|---|---|---|
| `model.layers[i].mlp.gate_proj.weight` | 랜덤 | SwiGLU gate (ProGen2에 없음) |
| `model.layers[i].mlp.up_proj.weight` | `fc_in.weight` 복사 | 차원 동일 (1536→6144) |
| `model.layers[i].mlp.down_proj.weight` | `fc_out.weight` 복사 | 차원 동일 (6144→1536) |
| `model.layers[i].post_attention_layernorm.weight` | 1.0 | 병렬→순차 전환으로 신규 |

fc_in/fc_out의 bias는 버림 (LlamaForCausalLM의 Linear에는 bias 없음).

### 3. RoPE 처리

- ProGen2: `rotary_dim=48`, head_dim=96 중 48차원만 회전
- Llama: head_dim 전체 회전

LlamaConfig에서 `rope_scaling` 또는 `partial_rotary_factor` 미지원 시,
`max_position_embeddings=1024`, `rope_theta=10000.0` 설정 후 미세조정으로 보정.

### 4. 출력

```
progen2-medium-llama/
├── config.json             # LlamaConfig
├── model.safetensors       # 변환된 가중치
├── tokenizer.json          # 원본 복사
└── special_tokens_map.json
```

---

## 추론 스크립트 수정

### `furiosa_backend.py` (신규)

```python
class FuriosaProGen2:
    def __init__(self, model_path, devices="npu:0"):
        from furiosa_llm import LLM
        self.llm = LLM(model_path, devices=devices)

    def generate(self, context, max_length, top_p, temp, num_samples):
        from furiosa_llm import SamplingParams
        params = SamplingParams(
            temperature=temp, top_p=top_p,
            max_tokens=max_length, n=num_samples)
        outputs = self.llm.generate([context], params)
        return [out.text for out in outputs[0].outputs]

    def shutdown(self):
        self.llm.shutdown()
```

### `sample.py` 수정

- `--provider pytorch|furiosa` 인자 추가
- `--furiosa-model-path` 인자 추가 (변환된 Llama 모델 또는 아티팩트 경로)
- provider 분기: pytorch면 기존 로직, furiosa면 `FuriosaProGen2` 사용

### `likelihood.py` 수정

- 동일한 `--provider` 패턴
- likelihood는 logits 접근 필요 → furiosa에서는 변환된 모델을 PyTorch로 로드하여 계산 (NPU 불필요)

---

## 수정/생성 파일 목록

| 파일 | 작업 | 설명 |
|------|------|------|
| `convert_progen2_to_llama.py` | 신규 | 가중치 변환 스크립트 |
| `furiosa_backend.py` | 신규 | RNGD 추론 백엔드 |
| `sample.py` | 수정 | `--provider`, `--furiosa-model-path` 추가 |
| `likelihood.py` | 수정 | `--provider`, `--furiosa-model-path` 추가 |

---

## 검증

1. **변환 검증**: attention Q,K,V,O 가중치 cosine similarity == 1.0 확인
2. **PyTorch forward**: 변환된 Llama 모델 CPU forward pass 성공
3. **furiosa-llm build**: `furiosa-llm build ./progen2-medium-llama ./output -tp 8` 성공
4. **NPU generate**: RNGD 서버에서 `python sample.py --provider furiosa` 성공
5. **정확도**: 미세조정 전/후 cross-entropy 비교

---

## 참조 파일

- `progen/modeling_progen.py:161-164` - QKV 분할 로직 (Q, V, K 순서)
- `progen/configuration_progen.py` - 모델 설정
- `progen2-medium/config.json` - 실제 모델 파라미터

---

## 진행 상황 로그

### Step 1: furiosa-llm 직접 로드 (완료 - 실패)

- **test_furiosa_load.py** 작성 및 RNGD 서버에서 실행
- Test 1 (직접 로드): `KeyError: 'progen'` - CONFIG_MAPPING에 미등록
- Test 2 (architectures 교체): 동일 실패 - `model_type`이 여전히 `"progen"`
- Test 3 (ArtifactBuilder): 동일 실패
- **결론**: furiosa-llm은 `model_type: "progen"`을 인식 불가

### Step 2: furiosa.torch.compiler 컴파일 (완료 - 실패)

- **test_furiosa_compile.py** 작성 및 RNGD 서버에서 실행
- Test 1 (nn.Linear): `UnsupportedOpError` - tensor import 실패
- Test 2-6 (ProGenMLP, Attention, Block, Full model, torch.compile backend): 전부 동일 실패
- **결론**: 현재 furiosa SDK에서 기본 op조차 컴파일 불가

### Step 3: Llama 가중치 변환 (진행 중)

**구현 완료된 파일:**
- `convert_progen2_to_llama.py` - 가중치 변환 스크립트 (QKV 분할, config 변환, 검증 포함)
- `furiosa_backend.py` - RNGD 추론 백엔드 래퍼
- `sample.py` - `--provider pytorch|furiosa`, `--furiosa-model-path` 인자 추가
- `likelihood.py` - 동일 패턴 추가
- `requirements_furiosa.txt` - safetensors 의존성 추가

**RNGD 서버 실행 중 (2026-03-24):**
```bash
python convert_progen2_to_llama.py \
  --model /mnt/elice/datahub/models/progen2/progen2-medium \
  --output /mnt/elice/datahub/results/progen2-medium
```
- Config 저장 완료
- ProGen2 모델 메모리 로딩 중 (764M, CPU 환경이라 시간 소요)
- GenerationMixin 상속 warning 발생 (변환에 무관)

**변환 실행 결과 (2026-03-24):**
- 변환 스크립트 실행 완료 (검증 단계에서 메모리 이슈로 Ctrl+\ 강제종료, 변환 결과물 자체는 정상 저장됨)
- 검증 단계는 모델을 2번째 로드하는 과정에서 NFS I/O 병목으로 10분+ 소요 → 강제종료

**변환 결과물 검증 (성공):**
```
경로: /mnt/elice/datahub/results/progen2-medium/
파일:
  config.json        (648 bytes)
  model.safetensors  (4,077,622,728 bytes ≈ 3.8GB)
  tokenizer.json     (1,631 bytes)
```

config.json 내용 (정상):
```json
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "vocab_size": 32,
  "hidden_size": 1536,
  "num_hidden_layers": 27,
  "num_attention_heads": 16,
  "num_key_value_heads": 16,
  "intermediate_size": 6144,
  "hidden_act": "silu",
  "max_position_embeddings": 1024,
  "rms_norm_eps": 1e-05,
  "rope_theta": 10000.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "tie_word_embeddings": false,
  "torch_dtype": "float32",
  "use_cache": true,
  "_progen2_source": {
    "rotary_dim": 48,
    "head_dim": 96,
    "note": "rotary_dim < head_dim: partial rotation in original, full rotation in Llama. Fine-tuning required."
  }
}
```

PyTorch 로드 테스트 (성공):
```bash
python -c "
from transformers import LlamaForCausalLM
m = LlamaForCausalLM.from_pretrained('/mnt/elice/datahub/results/progen2-medium')
print('OK')
print(f'params: {sum(p.numel() for p in m.parameters()):,}')
"
# 출력: OK / params: 1,019,398,656
# 원본 764M + gate_proj 추가분 ~255M (27층 × 1536×6144) = ~1,019M → 일치
```

**결론: Step 3a (ProGen2 → Llama 가중치 변환) 성공.**

---

### Step 3b: furiosa-llm build (실패 - ArtifactBuilder 모델 레지스트리 문제)

**실행 명령:**
```bash
furiosa-llm build /mnt/elice/datahub/results/progen2-medium ./output-progen2 -tp 8
```

**에러:**
```
WARNING:root:Could not find a matching config for model_type='llama' and
architectures='['LlamaForCausalLM']'. Available configs for this model type:
['UpstageShareFuriosaAI/solar-lnc-enkoja-10.7b-32k-1.3.0-chat.2',
 'meta-llama/CodeLlama-13b-hf', 'meta-llama/CodeLlama-7b-hf',
 'meta-llama/Llama-2-70b-chat-hf', 'meta-llama/Llama-3.1-70B',
 'meta-llama/Llama-3.1-8B', 'upstage/SOLAR-10.7B-Instruct-v1.0']

ValueError: ArtifactBuilder doesn't support <class 'transformers.models.llama.configuration_llama.LlamaConfig'> class
```

**원인 분석:**

`ArtifactBuilder.__init__` 내부 흐름:
1. `AutoConfig.from_pretrained(model_path)` → `LlamaConfig` 로드 성공
2. `find_canonical_model_id(hf_model_config, model_path)` → `None` 반환 → 실패

`find_canonical_model_id` 로직 (`builder.py`에서 import):
```python
def find_canonical_model_id(config, pretrained_id, task):
    config = config.to_dict()
    model_type = config['model_type']  # 'llama'

    # furiosa_llm.optimum.configs.llama/ 디렉토리에서 등록된 모델 config JSON 로드
    package = f"furiosa_llm.optimum.configs.{model_type}"
    files = resources.files(package)
    # JSON 파일명에서 model_id 복원: "org_model-name.json" → "org/model-name"
    local_configs = {파일명→model_id: json.loads(내용) for json 파일들}

    # 1단계: pretrained_id(=경로) 와 model_id 정확 일치 비교
    for model_id in local_configs:
        if model_id == pretrained_id:  # 우리: "/mnt/elice/..." ≠ "meta-llama/..."
            return model_id

    # 2단계: default_matcher(local_config, config) 로 config 값 비교
    for model_id, local_config in local_configs.items():
        if default_matcher(local_config, config):  # 차원/파라미터 불일치로 매칭 실패
            return model_id

    return None  # → ArtifactBuilder에서 ValueError 발생
```

**실패 이유:**
- 1단계: 우리 경로 `/mnt/elice/datahub/results/progen2-medium`이 등록된 model_id(`meta-llama/Llama-3.1-8B` 등)와 불일치
- 2단계: `default_matcher`가 config 파라미터(hidden_size, num_layers 등)를 비교하는데, 우리 모델(hidden_size=1536, 27층, vocab_size=32)이 등록된 Llama 모델들과 완전히 다름

**등록된 Llama 모델 config들:**
```
디렉토리: /home/elicer/.local/lib/python3.10/site-packages/furiosa_llm/optimum/configs/llama/
파일 목록:
  UpstageShareFuriosaAI_solar-lnc-enkoja-10.7b-32k-1.3.0-chat.2.json
  meta-llama_CodeLlama-13b-hf.json
  meta-llama_CodeLlama-7b-hf.json
  meta-llama_Llama-2-70b-chat-hf.json
  meta-llama_Llama-3.1-70B.json
  meta-llama_Llama-3.1-8B.json
  upstage_SOLAR-10.7B-Instruct-v1.0.json
```

Llama-3.1-8B의 config 예시 (참고):
```json
{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 14336,
  "vocab_size": 128256,
  ...
}
```
→ 우리 모델(1536/27/16/16/6144/32)과 완전히 다른 차원.

---

### Step 3c: 우회 방안 조사 (진행 중)

**접근법: 우리 모델용 config JSON을 `furiosa_llm.optimum.configs.llama/` 디렉토리에 직접 추가**

`find_canonical_model_id`가 해당 디렉토리의 JSON 파일을 순회하면서 매칭하므로,
우리 모델 파라미터에 맞는 JSON을 추가하면 `canonical_model_id`를 반환받을 수 있음.

**조사 진행:**

1. `default_matcher` 함수 위치 추적:
   - `from furiosa_llm.artifact.builder import default_matcher` → ImportError (builder.py에 정의되어 있지 않음)
   - `dir(b)`에서 'matcher' 검색 → 빈 리스트 (`[]`)
   - builder.py 파일 내 'matcher' 문자열 grep → 결과 없음
   - `find_canonical_model_id`의 실제 모듈 확인:
     ```
     find_canonical_model_id.__module__ = 'furiosa_llm.optimum.model_configs'
     inspect.getfile() = '/home/elicer/.local/lib/python3.10/site-packages/furiosa_llm/optimum/model_configs.py'
     ```
   - `grep -r "default_matcher" furiosa_llm/ --include="*.py" -l` 결과:
     ```
     /home/elicer/.local/lib/python3.10/site-packages/furiosa_llm/optimum/model_configs.py
     ```
   - **결론**: `find_canonical_model_id`와 `default_matcher` 모두 `furiosa_llm.optimum.model_configs` 모듈에 정의됨
     (builder.py에서 이 모듈을 import해서 사용)

2. `default_matcher` 소스 확인 필요 (다음 명령어):
   ```bash
   python -c "
   import inspect
   from furiosa_llm.optimum.model_configs import default_matcher
   print(inspect.getsource(default_matcher))
   "
   ```
   → 이 결과를 보고 어떤 필드를 비교하는지 파악 후 config JSON 작성 가능

3. `default_matcher` 소스 확인 완료:
   ```python
   def default_matcher(base, other):
       assert base['model_type'] == other['model_type']
       # 1. _name_or_path 일치하면 바로 True
       if base.get("_name_or_path", None) == other.get("_name_or_path", None):
           return True
       # 2. config 호환성 체크들
       if not _check_config_compatibility(other): return False
       if ('quantization_config' in base) != ('quantization_config' in other): return False
       if not _check_dtype_compatibility(base, other): return False
       if not _check_sliding_window(base, other): return False
       if not _check_vocab_size(base, other): return False
       if not _check_boolean_flags(base, other): return False
       # 3. 필터링 후 재귀 비교
       base = sort_config(filter_keys(base, DEFAULT_KEY_FILTERS))
       other = sort_config(filter_keys(other, DEFAULT_KEY_FILTERS))
       return _compare_configs_recursively(base, other)
   ```

**config JSON 추가 시도 #1 (부분 성공 → 다음 단계에서 실패):**

config JSON 파일 추가 완료:
- 파일 경로: `/home/elicer/.local/lib/python3.10/site-packages/furiosa_llm/optimum/configs/llama/progen2_progen2-medium.json`
- 파일명 규칙: `org_model-name.json` → model_id는 `progen2/progen2-medium`이 됨
- 내용: 우리 모델의 config.json과 동일한 파라미터 (`_progen2_source` 제외)

`furiosa-llm build` 재시도 결과:
- `find_canonical_model_id` **통과** → `canonical_model_id = "progen2/progen2-medium"` 반환 성공
- **다음 단계에서 실패**: `get_model_metadata_from_model_id` 내부에서
  `canonical_model_id` ("progen2/progen2-medium")를 HuggingFace Hub에서 다운로드 시도

**에러 트레이스:**
```
builder.py:248 → get_model_metadata_from_model_id(model_id="progen2/progen2-medium", ...)
  → metadata.py:330 → ModelMetadata.__init__()
    → metadata.py:282 → get_default_task_type_from_pretrained_id("progen2/progen2-medium")
      → metadata.py:172 → get_model_cls_from_pretrained_id("progen2/progen2-medium")
        → metadata.py:122 → get_config_from_pretrained_id("progen2/progen2-medium")
          → metadata.py:102 → AutoConfig.from_pretrained("progen2/progen2-medium")
            → HuggingFace Hub에 "progen2/progen2-medium" 조회 시도
            → 401 Unauthorized / Repository Not Found
```

**원인:**
`get_model_metadata_from_model_id`는 `canonical_model_id`(= `"progen2/progen2-medium"`)를
HuggingFace Hub의 model_id로 취급하여 `AutoConfig.from_pretrained`를 재호출함.
실제 HF Hub에 "progen2/progen2-medium" 레포가 없으므로 404/401 에러 발생.

**핵심 문제:**
furiosa-llm의 ArtifactBuilder는 `canonical_model_id`를 HuggingFace Hub model_id로 사용하여
원격 config를 다시 가져오는 구조. 커스텀 로컬 모델은 이 흐름에서 반드시 실패함.

---

### Step 3c 우회 방안 분석

**방안 A: `model_id_or_path`를 로컬 경로 대신 등록된 model_id로 전달**
- `furiosa-llm build meta-llama/Llama-3.1-8B ./output -tp 8` 형태로 호출하되,
  실제 가중치는 우리 모델 경로에 있도록 조작
- 문제: `AutoConfig.from_pretrained(model_id_or_path)` 첫 호출에서 HF Hub 접근 → 실패 가능

**방안 B: `get_model_metadata_from_model_id` 내부의 `model_id_or_path` 인자 활용**
- `builder.py:248` 코드:
  ```python
  self.model_metadata = get_model_metadata_from_model_id(
      model_id=self.canonical_model_id,
      model_id_or_path=self.model_id_or_path,  # ← 원본 로컬 경로
      ...
  )
  ```
- `get_model_metadata_from_model_id`가 `model_id_or_path`를 어떻게 사용하는지 확인 필요
- `metadata.py`의 `get_config_from_pretrained_id`가 `model_id`만 사용하고 `model_id_or_path`를 무시하는 것이 문제

**방안 C: `metadata.py`의 `get_config_from_pretrained_id` 함수를 몽키패치**
- `get_config_from_pretrained_id`가 "progen2/progen2-medium"을 받으면
  로컬 경로로 리다이렉트하도록 패치
- 가장 확실하지만 SDK 내부 수정이라 유지보수 부담

**방안 D: 기존 등록 모델의 config JSON을 우리 모델 파라미터로 덮어쓰기**
- 예: `meta-llama_Llama-3.1-8B.json`의 내용을 우리 파라미터로 교체
- `canonical_model_id`가 `"meta-llama/Llama-3.1-8B"`이 되면 HF Hub 접근 성공 (실제 존재하는 레포)
- 하지만 `AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B")`가 원본 Llama-3.1-8B config를 반환하므로
  우리 모델과 불일치

**`get_model_metadata_from_model_id` 소스 조사 완료:**

`metadata.py`의 `get_model_metadata_from_model_id` 함수 확인:
```python
def get_model_metadata_from_model_id(model_id, model_id_or_path, ...):
    # model_id = canonical_model_id ("progen2/progen2-medium")
    # model_id_or_path = 원본 로컬 경로 ("/mnt/elice/datahub/results/progen2-medium")
    ...
    return ModelMetadata(model_id=model_id, ...)  # model_id_or_path는 무시됨
```

`ModelMetadata.__init__` → `get_default_task_type_from_pretrained_id(model_id)` →
`get_model_cls_from_pretrained_id(model_id)` → `get_config_from_pretrained_id(model_id)` →
`AutoConfig.from_pretrained(model_id)` 호출.

여기서 `model_id` = `"progen2/progen2-medium"` → HuggingFace Hub 조회 시도 → 404/401 실패.

**핵심**: `AutoConfig.from_pretrained`는 인자가 HF Hub model_id처럼 보이면 원격 조회하고,
로컬 디렉토리 경로처럼 보이면 로컬에서 config.json을 읽음.
`"progen2/progen2-medium"`은 `org/model` 형식이라 HF Hub model_id로 취급됨.

---

### Step 3c 우회 방안: 심볼릭 링크 (제안)

**방안 E: 로컬에 `progen2/progen2-medium` 디렉토리 경로 생성**

`AutoConfig.from_pretrained("progen2/progen2-medium")`가 로컬 경로로 인식되려면,
실제로 `progen2/progen2-medium/config.json`이 존재해야 함.

```bash
# 작업 디렉토리에 심볼릭 링크 생성
mkdir -p progen2
ln -s /mnt/elice/datahub/results/progen2-medium progen2/progen2-medium

# 이제 "progen2/progen2-medium/config.json"이 상대 경로로 존재
# AutoConfig.from_pretrained("progen2/progen2-medium") → 로컬 config.json 로드
```

그 후 다시:
```bash
furiosa-llm build /mnt/elice/datahub/results/progen2-medium ./output-progen2 -tp 8
```

**주의점:**
- `find_canonical_model_id`가 반환한 `canonical_model_id = "progen2/progen2-medium"`을
  `AutoConfig.from_pretrained`에 전달할 때, CWD 기준 상대 경로로 해석되어야 함
- `os.path.isdir("progen2/progen2-medium")` 가 True면 로컬로 인식
- 하지만 furiosa-llm 내부에서 CWD가 변경되지 않는다는 보장 필요

**리스크:**
- `AutoConfig.from_pretrained`가 `os.path.isdir()` 체크 전에 HF Hub 조회를 먼저 시도할 수 있음
- transformers 버전에 따라 경로 해석 로직이 다를 수 있음

**심볼릭 링크 시도 결과: 실패**
- NFS 마운트 경로(`/mnt/elice/datahub/`)에서 symlink 생성 불가 (`Input/output error`)

**대안: 로컬 파일시스템에 config.json만 복사**

`AutoConfig.from_pretrained`는 config.json만 있으면 되므로,
`/home/elicer/`(로컬 FS)에 디렉토리 구조만 생성:

```bash
cd /home/elicer
mkdir -p progen2/progen2-medium
cp /mnt/elice/datahub/results/progen2-medium/config.json progen2/progen2-medium/
furiosa-llm build /mnt/elice/datahub/results/progen2-medium ./output-progen2 -tp 8
```

CWD가 `/home/elicer`이면:
- `AutoConfig.from_pretrained("progen2/progen2-medium")` → `os.path.isdir("progen2/progen2-medium")` = True → 로컬로 인식

**심볼릭 링크 대안 (config.json 복사) 실행 결과: 성공**
- `/home/elicer/progen2/progen2-medium/config.json` 생성
- `furiosa-llm build` 실행 → `find_canonical_model_id` 및 `get_model_metadata_from_model_id` 모두 통과!
- 모델 메타데이터 로드 성공: `pretrained_id='progen2/progen2-medium'`, `task_type='generate'`

**가중치 로딩에서 새로운 에러 발생:**
```
AssertionError: Attempted to load weight (torch.Size([32, 1536])) into parameter (torch.Size([64, 1536]))
```

**원인:**
- furiosa 내부 Llama 구현이 `vocab_size`를 패딩 (32 → 64)
- 우리 `embed_tokens.weight`/`lm_head.weight`: `[32, 1536]`
- furiosa가 기대하는 크기: `[64, 1536]`

**수정 방안:**
1. `config.json`의 `vocab_size`를 64로 변경
2. `embed_tokens.weight`, `lm_head.weight`를 `[64, 1536]`으로 제로패딩
3. furiosa configs의 `progen2_progen2-medium.json`도 vocab_size=64로 변경
4. 로컬 폴백 config도 동일하게 업데이트
5. 다시 `furiosa-llm build` 실행

**vocab_size 패딩 적용 후 재실행 결과:**

가중치 로딩 성공! 다음 단계까지 진행됨:
- `[CACHE] Saved params-...-llama-27L-...safetensors` → 파라미터 캐시 저장 성공
- Bucket 계산 완료: `batch_size=1, attention_size=2048`
- Ray 워커 시작 → 모델 로드 성공

**torch._dynamo.export 단계에서 새로운 에러 발생:**
```
torch._dynamo.exc.Unsupported: Observed exception
  Hint: Dynamo has detected that tracing the code will result in an error when running in eager.

from user code:
  File "furiosa/models/language/architecture/llama.py", line 596, in forward
    model_output: torch.Tensor = self.model(
```

**원인 분석:**
- 가중치 로딩은 성공했으나, furiosa의 자체 Llama 구현(`furiosa.models.language.architecture.llama`)의
  forward pass에서 AssertionError 발생
- `torch._dynamo.export`가 모델을 트레이싱할 때 eager 실행 중 에러 감지
- furiosa의 Llama 구현 내부 assertion이 우리 모델 파라미터와 불일치

**다음 조사 필요:**
1. 실제 AssertionError 내용 확인:
   ```bash
   TORCHDYNAMO_VERBOSE=1 furiosa-llm build /mnt/elice/datahub/results/progen2-medium ./output-progen2 -tp 8
   ```
   또는 furiosa의 Llama 모델을 직접 eager 모드로 실행:
   ```python
   python -c "
   import torch
   from furiosa.models.language.architecture.llama import LlamaForCausalLM as FuriosaLlama
   from transformers import AutoConfig
   config = AutoConfig.from_pretrained('/mnt/elice/datahub/results/progen2-medium')
   print(config)
   # furiosa 내부에서 config를 어떻게 변환하는지 확인
   "
   ```
2. `furiosa/models/language/architecture/llama.py:596` 부근 소스 확인
3. 어떤 차원/파라미터 assertion이 실패하는지 특정

**추가 조사 (2026-03-24):**

`TORCHDYNAMO_VERBOSE=1`로도 실제 assertion 메시지 미노출 (dynamo 스택만 표시).

furiosa의 `LlamaForCausalLM` API 확인:
- HuggingFace API와 완전히 다름: `__init__(self)` config를 인자로 받지 않음
- `DEFAULT_VOCAB_PADDING_SIZE` 상수 존재 (vocab_size 패딩 근거)
- forward 시그니처: `(input_ids, position_ids, kv_caches, attention_metadata, attention_masks, inputs_embeds, compute_logits)`
- 직접 eager 테스트 불가: 초기화 방식이 다름 (furiosa 내부 빌드 파이프라인 통해서만 인스턴스화)

**assertion 위치 파악:**

`llama.py` 내 assert는 1개뿐: `assert self.head_dim % 2 == 0` (line 149) → 우리 모델 head_dim=96, 통과.

`DEFAULT_VOCAB_PADDING_SIZE = 64` 확인.

즉 AssertionError는 llama.py가 아닌 **import된 하위 모듈**에서 발생.
furiosa/models/ 전체 assert 검색 결과, forward pass 경로에서 가능한 후보:

| 파일 | 라인 | assert 내용 | 관련성 |
|------|------|-------------|--------|
| `rotary_embedding.py:157` | `z.shape[-1] // 2 == rotary_matrix.shape[-3]` | **높음** - RoPE 차원 불일치 가능 |
| `scaled_dot_product.py:155` | `L % num_chunks == 0 and S % num_chunks == 0` | 중간 - attention 청크 분할 |
| `scaled_dot_product.py:224` | `L <= S` | 낮음 |
| `attention/backends/llm.py:268` | `key_cache.shape == value_cache.shape` | 낮음 |

**가장 유력한 원인: `rotary_embedding.py:157`**
```python
assert z.shape[-1] // 2 == rotary_matrix.shape[-3]
```
- 우리 config: head_dim=96, 원본 ProGen2는 rotary_dim=48 (절반만 회전)
- Llama는 head_dim 전체(96)에 RoPE 적용
- furiosa 내부에서 RoPE matrix를 head_dim=96 기준으로 생성하면 shape[-3]=48 (96//2)
- 하지만 실제 attention의 query/key z.shape[-1]이 다른 값이면 assertion 실패

**rotary_embedding.py:156-157 소스 확인:**
```python
assert z.shape[-1] % 2 == 0, "z must have an even number of dimensions."
assert z.shape[-1] // 2 == rotary_matrix.shape[-3], (
    "z and rotary_matrix must have the same number of pair_dim(D) dimensions. "
    f"Got {z.shape[-1] // 2} and {rotary_matrix.shape[-3]} respectively."
)
```
- z shape: `(..., H, D*2)`, rotary_matrix shape: `(..., D, 2, 2)`
- Llama에서 D = head_dim // 2 = 96 // 2 = 48
- 검증: `96 // 2 == 48` → **통과해야 함**

**LlamaModel.forward() 소스 확인 (llama.py:377~):**
- 각 `LlamaDecoder` 레이어를 순회하며 `layer(position_ids, hidden_states, kv_cache, attention_metadata, attention_mask)` 호출
- AssertionError는 `LlamaDecoder` → `LlamaAttention` → 하위 attention/RoPE 연산 중 발생

**RoPE assertion은 통과할 것으로 예상 (head_dim=96, 짝수).**
→ 실제 실패 지점은 **`scaled_dot_product.py`의 attention 청크 분할** 가능성 높음.

**LlamaAttention 소스 확인 (llama.py:107~250):**
```python
# __init__:
partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)  # 우리: 기본값 1
self.rotary_dim = int(partial_rotary_factor * self.head_dim)  # 1 * 96 = 96

self.rotary_emb = RotaryEmbedding(
    self.head_dim,  # 96
    rotary_dim=self.rotary_dim,  # 96
    max_position=max_position_embeddings,  # 1024 ← 우리 config
    ...
)

# forward:
q = self.q_proj(hidden_states)
k = self.k_proj(hidden_states)
v = self.v_proj(hidden_states)
q, k = self.rotary_emb(position_ids, q, k)  # ← RoPE 적용
attn_output = self.attn(q, k, v, kv_cache, attention_metadata, attention_mask)
```

- 차원 계산 모두 정상 (head_dim=96, rotary_dim=96, q_size=1536, kv_size=1536)
- RoPE/attention assertion 조건들은 우리 모델 차원에서 통과해야 함

**`scaled_dot_product.py:155` 확인:**
```python
assert L % num_chunks == 0 and S % num_chunks == 0
```
- `num_chunks` 기본값 1 → 항상 통과

**새로운 가설: `max_position_embeddings` vs `attention_size` 불일치**

빌드 로그: `attention_size=2048` (버킷), 우리 config: `max_position_embeddings=1024`
- `RotaryEmbedding(max_position=1024)` → cos/sin 캐시를 1024 위치까지만 생성
- 트레이싱 시 시퀀스 길이 2048으로 실행 → position_ids가 0~2047
- cos/sin 캐시 범위 (0~1023) 초과 → 에러 발생 가능

**수정 시도:** 3개 config의 `max_position_embeddings`를 `1024→2048`로 변경 후 재빌드.

**결과: 실패.** 동일한 AssertionError 발생. 가설 틀림.

---

### Step 3d: furiosa 소스 정밀 분석 — 근본 원인 특정 완료

로컬에 복사한 `furiosa_llm_src`, `furiosa_models_src` 정밀 분석으로 **근본 원인 확정**.

#### 근본 원인: `create_input_ids()`의 하드코딩된 토큰 범위

**파일**: `furiosa_models_src/common/export/serve/utils.py:76-77`
```python
input_ids = torch.randint(low=0, high=128, size=(batch_size, seq_len), ...)
```

**발생 경로**:
1. `furiosa-llm build` → `build_pipeline()` → `build_for_bucket()` → `make_example_inputs()`
2. `make_example_inputs()` → `_make_example_inputs()` → `CausalModelUtils.create_input_ids()`
3. `create_input_ids()`가 **토큰 ID 범위 [0, 128)** 으로 랜덤 생성
4. `torch._dynamo.export(model, ...)(*fake_args, **fake_kwargs)` → eager 실행
5. `LlamaForCausalLM.forward()` → `LlamaModel.forward()` → `self.embed_tokens(input_ids)`
6. 임베딩 테이블 크기 = 64, 토큰 ID >= 64 → **IndexError: index out of range**
7. `torch._dynamo`가 "Observed exception" 보고

**증거**:
- `VocabEmbedding(64, 1536, orig_num_embeddings=64)` → `pad_vocab_size(64, 64) = 64` → 테이블 크기 64
- `config.json`의 `vocab_size = 64` (원본 32에서 `DEFAULT_VOCAB_PADDING_SIZE=64`로 패딩)
- `create_input_ids`의 `high=128`은 일반 LLM (vocab_size >> 128) 전제 → 단백질 모델(vocab 32)에서 불일치

**수정**: `vocab_size`를 128 이상으로 설정 + 가중치를 128 크기로 제로패딩

#### 후속 블로커: GeneratorPipelineMetadata (서빙 시점에만 영향)

**파일**: `furiosa_llm_src/models/metadata.py:832-863`
```python
if model_qname in ["transformers.models.gptj.modeling_gptj.GPTJForCausalLM"]:
    ...
elif model_qname in ["furiosa.models.language.architecture.qwen2.Qwen2ForCausalLM", ...]:
    ...
else:
    raise ValueError(f"unknown model qname: {model_qname}")
```

`furiosa.models.language.architecture.llama.LlamaForCausalLM`이 목록에 없음.
- 단, 이 함수는 `config_types.GeneratorConfig` (서빙 시점)에서만 사용됨
- 빌드 시점에서는 `artifact.types.next_gen.GeneratorConfig` 사용 → **빌드에는 영향 없음**
- 서빙 시 `metadata.py`의 이 분기에 Llama를 추가해야 함

---

## Step 3e: vocab_size 128 패딩 적용

### 1. 변환 스크립트 수정 — ✅ 완료

`convert_progen2_to_llama.py` 수정 내용:
- `FURIOSA_MIN_VOCAB_SIZE = 128` 상수 추가, `vocab_size = max(원본, 128)` 로 패딩
- `pad_weight()` 함수 추가: 임베딩/lm_head를 제로패딩
- `embed_tokens.weight`: `[32, 1536]` → `[128, 1536]`
- `lm_head.weight`: `[32, 1536]` → `[128, 1536]`
- `--no-verify` 옵션 추가 (메모리 절약)
- 검증 함수도 패딩 고려 (원본 범위만 비교)

### 2. RNGD 서버 실행 — ⏳ 다음 단계

수정된 `convert_progen2_to_llama.py`를 서버에 업로드 후 아래 실행:

```bash
# 1. 재변환 (vocab_size=128로 패딩)
python convert_progen2_to_llama.py \
  --model /mnt/elice/datahub/models/progen2/progen2-medium \
  --output /mnt/elice/datahub/results/progen2-medium \
  --no-verify

# 2. furiosa config JSON 업데이트 (vocab_size=128)
python3 -c "
import json, os

config_path = '/mnt/elice/datahub/results/progen2-medium/config.json'
with open(config_path) as f:
    config = json.load(f)

# furiosa optimum config
furiosa_config = {k: v for k, v in config.items() if not k.startswith('_')}
furiosa_path = '/home/elicer/.local/lib/python3.10/site-packages/furiosa_llm/optimum/configs/llama/progen2_progen2-medium.json'
with open(furiosa_path, 'w') as f:
    json.dump(furiosa_config, f, indent=2)
print(f'vocab_size={furiosa_config[\"vocab_size\"]}')

# 로컬 폴백 config
os.makedirs('/home/elicer/progen2/progen2-medium', exist_ok=True)
with open('/home/elicer/progen2/progen2-medium/config.json', 'w') as f:
    json.dump(config, f, indent=2)
print('configs updated')
"

# 3. furiosa-llm build
cd /home/elicer
furiosa-llm build /mnt/elice/datahub/results/progen2-medium ./output-progen2 -tp 8
```

기존 결과물은 덮어쓰기됨 (삭제 불필요).

### 3. 서빙 시 GeneratorPipelineMetadata 패치 (빌드 성공 후)

`metadata.py`의 `from_pipeline_metadata()`에 Llama 분기 추가:
```python
elif model_qname in [
    "furiosa.models.language.architecture.llama.LlamaForCausalLM",
]:
    include_softmax_in_graph = False
    prefill_mask_dim = 3
    decode_mask_dim = 3
```

### 검증

1. **빌드 성공**: `furiosa-llm build` 완료 확인
2. **아티팩트 파일 생성**: `output-progen2/` 디렉토리에 결과물 확인
3. **서빙 테스트**: `furiosa-llm serve` 또는 `python sample.py --provider furiosa` 실행

---

#### 참조 파일 경로

**furiosa_models_src:**

| 파일 | 역할 |
|------|------|
| `language/architecture/llama.py` | Llama 모델 전체 |
| `common/export/serve/causal.py` | CausalModelServer (make_example_inputs) |
| `common/export/serve/utils.py` | CausalModelUtils (create_input_ids — **근본 원인**) |
| `common/export/serve/specs/inputs.py` | CausalModelForwardInputs (입력 검증) |
| `core/layers/vocab_embedding.py` | VocabEmbedding/LMHead (DEFAULT_VOCAB_PADDING_SIZE=64) |

**furiosa_llm_src:**

| 파일 | 역할 |
|------|------|
| `artifact/builder.py` | ArtifactBuilder |
| `parallelize/trace.py` | trace_model() → torch._dynamo.export |
| `parallelize/new_pipeline_builder.py` | build_pipeline, build_for_bucket |
| `models/metadata.py` | ModelMetadata, GeneratorPipelineMetadata |
| `models/utils.py` | generate_input_sample |
| `optimum/model_configs.py` | find_canonical_model_id, default_matcher |
