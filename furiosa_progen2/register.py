"""
furiosa.models 및 furiosa-llm에 ProGenForCausalLM을 런타임 등록합니다.

이 모듈은 furiosa-llm이 ProGen2 모델을 발견하고 로드할 수 있도록
필요한 모든 패치를 적용합니다.

등록 순서:
1. HuggingFace transformers에 ProGenConfig 등록
2. furiosa.models에 ProGenForCausalLM 주입
3. furiosa.models 레지스트리에 추가
4. furiosa-llm 메타데이터 패치 (mask dimension)

사용법:
    from furiosa_progen2.register import register_progen2
    register_progen2()

    from furiosa.llm import LLM
    llm = LLM(model="./checkpoints/progen2-medium-furiosa/")
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _register_hf_config():
    """HuggingFace transformers에 ProGenConfig와 ProGenForCausalLM을 등록합니다."""
    from transformers import AutoConfig, AutoModelForCausalLM

    # progen 패키지가 import 경로에 있는지 확인
    progen2_root = Path(__file__).parent.parent
    if str(progen2_root) not in sys.path:
        sys.path.insert(0, str(progen2_root))

    from progen.configuration_progen import ProGenConfig
    from progen.modeling_progen import ProGenForCausalLM as HFProGenForCausalLM

    # AutoConfig에 "progen" model_type 등록
    AutoConfig.register("progen", ProGenConfig)

    # AutoModelForCausalLM에 등록
    AutoModelForCausalLM.register(ProGenConfig, HFProGenForCausalLM)

    # AutoTokenizer에 등록 — ProGen2는 커스텀 tokenizer.json을 사용
    # GPT2TokenizerFast가 호환되므로 이를 매핑
    from transformers import AutoTokenizer, GPT2TokenizerFast
    AutoTokenizer.register(ProGenConfig, slow_tokenizer_class=None, fast_tokenizer_class=GPT2TokenizerFast)

    logger.info("Registered ProGenConfig, ProGenForCausalLM, and tokenizer with HuggingFace transformers")


def _register_furiosa_models():
    """furiosa.models 패키지에 ProGenForCausalLM을 주입합니다.

    furiosa-llm은 다음 코드로 최적화된 모델 클래스를 찾습니다:
        cls = getattr(furiosa.models, model_cls.__name__, None)

    따라서 furiosa.models 모듈에 직접 속성을 추가합니다.
    """
    from furiosa_progen2.modeling import ProGenForCausalLM, ProGen2Model

    import furiosa.models
    furiosa.models.ProGenForCausalLM = ProGenForCausalLM
    furiosa.models.ProGen2Model = ProGen2Model

    # language 서브모듈에도 등록
    try:
        import furiosa.models.language
        furiosa.models.language.ProGenForCausalLM = ProGenForCausalLM
        furiosa.models.language.ProGen2Model = ProGen2Model
    except ImportError:
        pass

    logger.info("Injected ProGenForCausalLM into furiosa.models")


def _register_model_registry():
    """furiosa.models 레지스트리에 ProGenForCausalLM을 추가합니다."""
    try:
        from furiosa.models.interfaces.registry import (
            _TEXT_ONLY_GENERATIVE_TEXT_GENERATION_MODELS,
        )
        if "ProGenForCausalLM" not in _TEXT_ONLY_GENERATIVE_TEXT_GENERATION_MODELS:
            _TEXT_ONLY_GENERATIVE_TEXT_GENERATION_MODELS.append("ProGenForCausalLM")
            logger.info("Added ProGenForCausalLM to model registry")
    except ImportError:
        logger.warning("Could not import furiosa.models.interfaces.registry")


def _patch_metadata():
    """furiosa-llm의 GeneratorPipelineMetadata에 ProGen2 mask 설정을 추가합니다.

    원본 from_pipeline_metadata()는 model_qname으로 분기하여
    mask_dim을 설정합니다. ProGen2는 Llama/Qwen과 동일한 3D mask를 사용합니다.
    """
    try:
        from furiosa_llm.models.metadata import GeneratorPipelineMetadata

        original_method = GeneratorPipelineMetadata.from_pipeline_metadata

        @staticmethod
        def patched_from_pipeline_metadata(generator_config, pipelines, pipeline_metadata):
            model_qname = generator_config.model_qname
            # ProGen2 모델인 경우 직접 처리
            if "progen" in model_qname.lower() or "ProGen" in model_qname:
                from furiosa_llm.models.metadata import (
                    _get_bucket_from_pipeline_name,
                )
                buckets = [_get_bucket_from_pipeline_name(p.name) for p in pipelines]
                return [
                    GeneratorPipelineMetadata(
                        batch_size=bucket.batch_size,
                        attention_size=bucket.attention_size,
                        kv_cache_size=bucket.kv_cache_size,
                        output_logits_size=pipeline.output_logits_size,
                        include_softmax_in_graph=False,
                        mask_dim=3 if bucket.is_prefill else 3,
                        sliding_window_size=None,
                    )
                    for bucket, pipeline in zip(buckets, pipelines)
                ]
            # 원본 메서드로 폴백
            return original_method(generator_config, pipelines, pipeline_metadata)

        GeneratorPipelineMetadata.from_pipeline_metadata = patched_from_pipeline_metadata
        logger.info("Patched GeneratorPipelineMetadata for ProGen2")
    except ImportError:
        logger.warning("Could not import furiosa_llm.models.metadata for patching")
    except Exception as e:
        logger.warning(f"Failed to patch metadata: {e}")


def register_progen2():
    """ProGen2를 furiosa 에코시스템에 등록하는 메인 함수.

    이 함수를 furiosa-llm으로 ProGen2를 로드하기 전에 호출하세요.

    등록 내용:
    1. HuggingFace transformers에 ProGenConfig + ProGenForCausalLM 등록
    2. furiosa.models에 최적화된 ProGenForCausalLM 주입
    3. furiosa.models 레지스트리에 모델 추가
    4. furiosa-llm 메타데이터에 mask dimension 패치
    """
    logger.info("Registering ProGen2 with furiosa ecosystem...")

    _register_hf_config()
    _register_furiosa_models()
    _register_model_registry()
    _patch_metadata()

    logger.info("ProGen2 registration complete!")
