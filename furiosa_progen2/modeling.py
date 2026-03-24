"""
ProGen2 아키텍처의 furiosa-models 호환 구현.

ProGen2는 GPT-J 계열 decoder-only 단백질 언어 모델로,
Llama와 주요 차이점은 다음과 같습니다:

1. GPT-J 병렬 블록: attn + mlp가 같은 normed input을 병렬로 처리
2. LayerNorm (RMSNorm 대신), 블록당 1개
3. Simple FFN: fc_in → GELU → fc_out (SwiGLU 대신)
4. 부분 RoPE: rotary_dim=64만 적용 (GPT-J 인터리브 스타일)
5. MLP에 bias 있음, attention에는 bias 없음

이 파일은 furiosa.models.language.architecture 패턴을 따르며,
furiosa_progen2.register 모듈을 통해 런타임에 등록됩니다.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from furiosa.models.common.export.serve import CausalModelServer
from furiosa.models.common.utils.blocks import (
    append_prefix,
    make_layers,
)
from furiosa.models.core.attention import AttentionMetadataBase
from furiosa.models.core.attention.attention import AttentionLayer as Attention
from furiosa.models.core.attention.ops.attention_mask import (
    FullAttentionMask,
    LLMAttentionMask,
    LLMAttentionMasks,
)
from furiosa.models.core.layers import LinearLayer as Linear
from furiosa.models.core.layers.activation import GELULayer as GELU
from furiosa.models.core.layers.layernorm import LayerNormLayer as LayerNorm
from furiosa.models.core.layers.logits_processor import LogitsProcessor
from furiosa.models.core.layers.rotary_embedding import (
    RotaryEmbeddingLayer as RotaryEmbedding,
)
from furiosa.models.core.layers.vocab_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    LMHeadLayer as LMHead,
    VocabEmbeddingLayer as VocabEmbedding,
)
from furiosa.models.core.quantization import QuantizationConfig
from furiosa.models.language.base import LanguageModelBase
from furiosa.models.language.config import CacheConfig, LLMConfig


class ProGen2MLP(nn.Module):
    """ProGen2 MLP 블록: fc_in → GELU(tanh) → fc_out.

    Llama의 SwiGLU 3-layer와 달리, simple 2-layer FFN 구조.
    fc_in과 fc_out 모두 bias를 사용합니다.

    Args:
        hidden_size: 히든 스테이트 차원.
        intermediate_size: FFN 중간 차원 (= 4 * hidden_size).
        quant_config: 양자화 설정.
        prefix: 가중치 이름 접두사.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.fc_in = Linear(
            input_size=hidden_size,
            output_size=intermediate_size,
            use_bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc_in",
        )
        self.fc_out = Linear(
            input_size=intermediate_size,
            output_size=hidden_size,
            use_bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc_out",
        )
        # gelu_new = GELU with tanh approximation
        self.act_fn = GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        x = self.act_fn(x)
        x = self.fc_out(x)
        return x


class ProGen2Attention(nn.Module):
    """ProGen2 어텐션 블록: MHA + partial RoPE (GPT-J 인터리브).

    Llama와의 차이:
    - 부분 RoPE: rotary_dim=64만 적용 (나머지 dims는 그대로)
    - is_neox_style=False: GPT-J 인터리브 방식 회전
    - MHA (GQA 아님): num_kv_heads == num_heads
    - bias=False (Q, K, V, O 모두)

    Args:
        hidden_size: 히든 스테이트 차원.
        num_heads: 어텐션 헤드 수.
        rotary_dim: RoPE 적용 차원 (기본 64).
        max_position_embeddings: 최대 시퀀스 길이.
        cache_config: KV 캐시 설정.
        quant_config: 양자화 설정.
        prefix: 가중치 이름 접두사.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rotary_dim: int = 64,
        max_position_embeddings: int = 2048,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim % 2 == 0, "head_dim must be divisible by 2."

        self.rotary_dim = rotary_dim
        self.scaling = self.head_dim ** -0.5
        self.max_position_embeddings = max_position_embeddings

        # 분리된 Q, K, V, O 프로젝션 (bias=False)
        # 원본 ProGen2는 결합 qkv_proj를 사용하지만,
        # 체크포인트 변환 시 분리됨 (convert_checkpoint.py)
        self.q_proj = Linear(
            input_size=hidden_size,
            output_size=hidden_size,
            use_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = Linear(
            input_size=hidden_size,
            output_size=hidden_size,
            use_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = Linear(
            input_size=hidden_size,
            output_size=hidden_size,
            use_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )
        self.out_proj = Linear(
            input_size=hidden_size,
            output_size=hidden_size,
            use_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        # 부분 RoPE: rotary_dim만 회전, 나머지는 패스스루
        # GPT-J 인터리브 스타일 (is_neox_style=False)
        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=10000.0,
            is_neox_style=False,
        )

        # 어텐션 레이어 (MHA: num_kv_heads = num_heads)
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attention_metadata: AttentionMetadataBase,
        attention_mask: Optional[LLMAttentionMask] = None,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q, k = self.rotary_emb(position_ids, q, k)

        attn_output = self.attn(
            q, k, v, kv_cache, attention_metadata, attention_mask
        )
        output: torch.Tensor = self.out_proj(attn_output)
        return output


class ProGen2Block(nn.Module):
    """ProGen2 트랜스포머 블록 (GPT-J 병렬 구조).

    Llama의 순차 구조와 달리, 단일 LayerNorm 후
    어텐션과 MLP가 같은 normed input을 병렬로 처리합니다:

        normed = LayerNorm(x)
        output = x + Attention(normed) + MLP(normed)

    Args:
        config: ProGen2 HuggingFace config.
        cache_config: KV 캐시 설정.
        quant_config: 양자화 설정.
        prefix: 가중치 이름 접두사.
    """

    def __init__(
        self,
        config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.n_embd
        intermediate_size = config.n_inner if config.n_inner is not None else 4 * hidden_size
        rotary_dim = getattr(config, "rotary_dim", 64)
        max_position_embeddings = getattr(config, "n_positions", 2048)

        # 단일 LayerNorm (블록당 1개, bias 포함)
        self.ln_1 = LayerNorm(
            hidden_size=hidden_size,
            eps=config.layer_norm_epsilon,
            elementwise_affine=True,
            use_bias=True,
        )

        self.self_attn = ProGen2Attention(
            hidden_size=hidden_size,
            num_heads=config.n_head,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )

        self.mlp = ProGen2MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attention_metadata: AttentionMetadataBase,
        attention_mask: Optional[LLMAttentionMask] = None,
    ) -> torch.Tensor:
        # GPT-J 병렬 블록: attn과 mlp가 같은 normed input을 받음
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        attn_output = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attention_metadata=attention_metadata,
            attention_mask=attention_mask,
        )
        mlp_output = self.mlp(hidden_states)

        # 3-way 합산: residual + attn + mlp
        hidden_states = residual + attn_output + mlp_output
        return hidden_states


class _ProGen2Base(LanguageModelBase):
    """Internal ProGen2 base shim."""

    def __init__(self, *, llm_config: LLMConfig) -> None:
        config = llm_config.model_config.hf_config
        super().__init__(config=config)


class ProGen2Model(_ProGen2Base):
    """ProGen2 모델 본체: 임베딩 + N개 블록 + 최종 LayerNorm.

    Args:
        llm_config: LLM 설정.
        prefix: 가중치 이름 접두사.
    """

    def __init__(
        self,
        *,
        llm_config: LLMConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(llm_config=llm_config)
        config = llm_config.model_config.hf_config
        cache_config = llm_config.cache_config
        quant_config = llm_config.quant_config

        self.config = config
        hidden_size = config.n_embd
        num_layers = config.n_layer

        # 토큰 임베딩
        self.wte = VocabEmbedding(
            config.vocab_size,
            hidden_size,
            orig_num_embeddings=config.vocab_size,
        )

        # 트랜스포머 블록들
        self.layers = make_layers(
            num_layers=num_layers,
            layer_fn=lambda idx: ProGen2Block(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{idx}",
            ),
        )

        # 최종 LayerNorm
        self.ln_f = LayerNorm(
            hidden_size=hidden_size,
            eps=config.layer_norm_epsilon,
            elementwise_affine=True,
            use_bias=True,
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings: torch.Tensor = self.wte(input_ids)
        return embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        attention_metadata: AttentionMetadataBase,
        attention_masks: Optional[LLMAttentionMasks] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states: torch.Tensor = inputs_embeds
        else:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            hidden_states = self.get_input_embeddings(input_ids)

        if len(kv_caches) != len(self.layers):
            raise ValueError(
                f"KV caches must match the number of layers. "
                f"Got {len(kv_caches)} but expected {len(self.layers)}."
            )

        if not (
            attention_masks is None or isinstance(attention_masks, FullAttentionMask)
        ):
            raise TypeError(
                f"attention_masks must be a FullAttentionMask, got {type(attention_masks)}."
            )

        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                position_ids,
                hidden_states,
                kv_caches[i],
                attention_metadata,
                attention_masks,
            )

        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class ProGenForCausalLM(_ProGen2Base, CausalModelServer):
    """ProGen2 Causal Language Model for furiosa RNGD NPU.

    HuggingFace의 ProGenForCausalLM과 동일한 아키텍처를
    furiosa-models CausalModelServer 인터페이스로 구현합니다.

    Args:
        llm_config: LLM 설정.
        prefix: 가중치 이름 접두사.
    """

    def __init__(self, *, llm_config: LLMConfig, prefix: str = "") -> None:
        CausalModelServer.__init__(self, llm_config=llm_config)
        _ProGen2Base.__init__(self, llm_config=llm_config)

        self.model = self._init_model(
            llm_config=self.llm_config,
            prefix=append_prefix(prefix, "model"),
        )

        self.unpadded_vocab_size = self.config.vocab_size
        self.lm_head = LMHead(
            self.unpadded_vocab_size,
            self.config.n_embd,
            orig_num_embeddings=self.config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
        )
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = self.lm_head.tie_weights(self.model.wte)

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            lm_head=self.lm_head,
            vocab_size=self.unpadded_vocab_size,
            org_vocab_size=self.config.vocab_size,
            scale=logit_scale,
        )

    def _init_model(self, llm_config: LLMConfig, prefix: str = "") -> ProGen2Model:
        return ProGen2Model(llm_config=llm_config, prefix=prefix)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        attention_metadata: AttentionMetadataBase,
        attention_masks: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        compute_logits: bool = True,
    ) -> torch.Tensor:
        model_output: torch.Tensor = self.model(
            input_ids,
            position_ids,
            kv_caches,
            attention_metadata,
            attention_masks,
            inputs_embeds,
        )
        if compute_logits:
            return self.compute_logits(model_output)
        return model_output

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor.forward(hidden_states)
