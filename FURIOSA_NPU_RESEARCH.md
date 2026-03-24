# FuriosaAI RNGD NPU 포팅 조사 결과

## 1. 서버 환경 (엘리스 컨테이너)

- **NPU 디바이스**: `/dev/rngd/npu4*` (RNGD 4번 디바이스, PE 0~7)
- **APT 패키지**: `furiosa-compiler 0.10.3`, `furiosa-libnux 0.10.1` (renegade), `furiosa-toolkit 0.11.0`
- **Python**: 3.10
- **PyPI SDK**: `furiosa-sdk 0.10.2` (Warboy용) → RNGD 디바이스 못 찾음

### SDK 버전 문제

| SDK | 버전 | 대상 하드웨어 | 디바이스 경로 | 상태 |
|-----|------|-------------|-------------|------|
| `furiosa-sdk` (PyPI) | 0.10.2 | Warboy | `/dev/npu*` | RNGD에서 사용 불가 |
| `furiosa-llm` (PyPI) | 2026.1.0 | RNGD | `/dev/rngd/*` | LLM 전용, 범용 ONNX 런타임 없음 |

---

## 2. RNGD 공식 지원 모델 (SDK 2026.1.0 기준)

### 디코더 전용 (Text Generation)
| 모델 | 아키텍처 클래스 |
|------|---------------|
| DeepSeek R1 | `LlamaForCausalLM` |
| EXAONE 4.0 | `Exaone4ForCausalLM` |
| Llama 3.1, 3.3 | `LlamaForCausalLM` |
| Solar 1.0 | `LlamaForCausalLM` |
| Qwen 2, 2.5 | `Qwen2ForCausalLM` |
| Qwen 3 | `Qwen3ForCausalLM` |

### 풀링 모델 (Embedding/Reranking)
| 모델 | 아키텍처 클래스 |
|------|---------------|
| Qwen 3 Embedding | `Qwen3Model` |
| Qwen 3 Reranker | `Qwen3ForSequenceClassification` |

### 지원하지 않는 모델
- **Encoder-only**: BERT, T5 Encoder, ESM 등
- **Encoder-Decoder**: T5 (full), BART 등
- **커스텀 아키텍처**: ProGenForCausalLM 등 (공식 리스트에 없는 것)

> 참고: BERT는 이전 SDK(v2024.1.0)에서 MLPerf 벤치마크용으로만 지원. 최신 SDK에서는 리스트에서 제외됨.

---

## 3. TemStaPro (T5 Encoder) → RNGD: 불가

### TemStaPro 모델 구조
- **T5 Encoder** (`Rostlab/prot_t5_xl_half_uniref50-enc`): ~1.2B params
- **MLP 분류기** (`MLP_C2H2`): 1024→256→128→1, ~30만 params
- T5 Encoder가 99%+ 연산량 차지

### 실패 이유
1. T5 Encoder는 **encoder-only** 아키텍처 → RNGD 지원 범위 밖
2. RNGD SDK(2026.1.0)는 `furiosa-llm` 기반으로 **decoder-only LLM만** 지원
3. Warboy SDK(0.10.x)는 범용 ONNX 런타임이 있지만 RNGD 디바이스(`/dev/rngd/*`)를 인식 못 함
4. `furiosa.torch.compiler.compile_gm()`으로 직접 컴파일 시도 → `aten::linear`조차 `UnsupportedOpError`

### 구현했던 코드 (현재 사용 불가)
- `furiosa_backend.py`: ONNX 런타임 기반 T5 추론 모듈 (Warboy API 사용)
- `temstapro` CLI: `--provider furiosa --onnx-model` 옵션 추가
- `export_t5_onnx.py`: PyTorch → ONNX 변환 유틸리티
- `requirements_FURIOSA.txt`: 의존성 목록

---

## 4. ProGen2 → RNGD: 가능성 높음

### ProGen2 모델 구조
- **ProGenForCausalLM**: decoder-only causal LM (GPT 계열)
- HuggingFace `PreTrainedModel` 기반
- Rotary Positional Embedding (RoPE) 사용
- Grouped Query Attention 지원
- FP16 지원

### RNGD 지원 모델과 구조 비교

| | ProGen2 | Llama 3 (RNGD 지원) |
|---|---|---|
| 구조 | Decoder-only | Decoder-only |
| 위치 인코딩 | RoPE | RoPE |
| Attention | Causal + GQA | Causal + GQA |
| Activation | GELU | SiLU |
| Norm | Pre-LayerNorm | RMSNorm |
| KV Cache | 지원 | 지원 |

### 모델 사이즈

| 모델 | 파라미터 |
|------|---------|
| progen2-small | 151M |
| progen2-medium | 764M |
| progen2-base | 764M |
| progen2-large | 2.7B |
| progen2-xlarge | 6.4B |

### 포팅 방법 (검토 필요)

**방법 1: Llama 아키텍처로 가중치 변환**
- ProGen2 가중치를 `LlamaForCausalLM` 형식으로 매핑
- Activation(GELU→SiLU), Norm(LayerNorm→RMSNorm) 차이 때문에 정확한 1:1 변환 어려움
- 정확도 손실 가능성

**방법 2: FuriosaAI에 ProGenForCausalLM 아키텍처 지원 요청**
- 구조가 Llama와 매우 유사하므로 지원 추가가 비교적 간단할 수 있음
- 공식 지원이 가장 안정적

**방법 3: furiosa.torch.compiler로 직접 컴파일 시도**
- `torch.export` → `compile_gm()` 파이프라인
- 지원 연산자 제한으로 실패 가능성 있음 (Linear조차 실패한 전례)

---

## 5. RNGD SDK API 구조 (2026.1.0)

### Python 모듈

```
furiosa.
├── models              # 모델 정의
├── native_compiler     # 컴파일러
├── native_llm_common   # LLM 공통 유틸
├── native_runtime      # 네이티브 런타임 (llm 서브모듈만 있음)
├── native_torch        # PyTorch 네이티브 바인딩
└── torch               # PyTorch 연동
    ├── backend         # torch.compile 백엔드
    ├── compiler        # 모델 컴파일 (compile_gm)
    ├── config
    ├── debug
    ├── execution_pool
    ├── extension
    ├── fx              # FX 그래프 변환
    ├── lla
    ├── profiler
    └── utils
```

### 핵심 API
- `furiosa.torch.compiler.compile_gm(gm, inputs)` → `Edf` (컴파일된 모델)
- `furiosa.torch.backend.backend` → `torch.compile` 백엔드
- Warboy의 `furiosa.runtime.sync.create_runner()` → **RNGD에서 없음**

---

## 6. 참고 링크

- [RNGD 공식 지원 모델](https://developer.furiosa.ai/latest/en/overview/supported_models.html)
- [Warboy SDK 문서](http://developer.furiosa.ai/docs/latest/en/)
- [RNGD MLPerf BERT 벤치마크](https://developer.furiosa.ai/docs/v2024.2.1/en/getting_started/furiosa_mlperf.html)
- [FuriosaAI SDK 2025.3 릴리즈](https://furiosa.ai/blog/furiosaai-sdk-2025-3-boosts-rngd-performance-with-multichip-scaling-and-more)
- [HuggingFace ProtTrans T5 ONNX](https://huggingface.co/Rostlab/prot-t5-xl-uniref50-enc-onnx)

---

## 7. 결론

| 모델 | RNGD 호환 | 이유 |
|------|----------|------|
| TemStaPro (T5 Encoder) | **불가** | Encoder-only, RNGD는 decoder-only만 지원 |
| ProGen2 | **가능성 높음** | Decoder-only, Llama와 구조 유사. 단, 공식 지원 아키텍처 아님 |

**다음 단계 (ProGen2):**
1. `furiosa-llm` API로 ProGen2 로드 시도 (아키텍처 클래스 매핑)
2. 안 되면 FuriosaAI에 `ProGenForCausalLM` 지원 요청
3. 또는 Llama 아키텍처로 가중치 변환 검토
