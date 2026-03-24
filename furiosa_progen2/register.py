"""
furiosa.models 및 furiosa-llm에 ProGenForCausalLM을 런타임 등록합니다.

이 모듈은 furiosa-llm이 ProGen2 모델을 발견하고 로드할 수 있도록
필요한 모든 패치를 적용합니다.

이전 시도에서 발견된 문제점들을 모두 반영:
1. model_type "progen"이 transformers CONFIG_MAPPING에 미등록 → AutoConfig 등록
2. AutoTokenizer가 ProGenConfig 매핑을 모름 → GPT2TokenizerFast 등록
3. furiosa.models에 ProGenForCausalLM 없음 → 런타임 주입
4. find_canonical_model_id가 config JSON 매칭 실패 → 패치
5. canonical_model_id를 HF Hub에서 조회 시도 → 로컬 경로로 리다이렉트
6. GeneratorPipelineMetadata에 model_qname 분기 없음 → 패치

사용법:
    from furiosa_progen2.register import register_progen2
    register_progen2()

    from furiosa_llm import LLM
    llm = LLM(model_id_or_path="./artifact/")
"""

import json
import logging
import os
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
    try:
        AutoConfig.register("progen", ProGenConfig)
    except ValueError:
        pass  # 이미 등록됨

    # AutoModelForCausalLM에 등록
    try:
        AutoModelForCausalLM.register(ProGenConfig, HFProGenForCausalLM)
    except ValueError:
        pass

    # AutoTokenizer에 등록 — ProGen2는 커스텀 tokenizer.json을 사용
    from transformers import AutoTokenizer, GPT2TokenizerFast
    try:
        AutoTokenizer.register(ProGenConfig, slow_tokenizer_class=None, fast_tokenizer_class=GPT2TokenizerFast)
    except ValueError:
        pass

    logger.info("Registered ProGenConfig, ProGenForCausalLM, and tokenizer with HuggingFace transformers")


def _register_furiosa_models():
    """furiosa.models 패키지에 ProGenForCausalLM을 주입합니다."""
    from furiosa_progen2.modeling import ProGenForCausalLM, ProGen2Model

    import furiosa.models
    furiosa.models.ProGenForCausalLM = ProGenForCausalLM
    furiosa.models.ProGen2Model = ProGen2Model

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


def _inject_canonical_config(model_path: str = None):
    """furiosa-llm의 find_canonical_model_id가 ProGen2를 인식하도록 패치합니다.

    이전 시도에서 발견된 문제:
    1. find_canonical_model_id()가 optimum/configs/{model_type}/ 에서 config JSON을 찾음
    2. model_type "progen"에 해당하는 디렉토리가 없으면 매칭 실패
    3. 매칭 성공해도 canonical_model_id를 HF Hub에서 조회 시도 → 실패

    해결: find_canonical_model_id를 패치하여 "progen" 모델이면 로컬 경로를 반환
    """
    try:
        import furiosa_llm.optimum.model_configs as model_configs_module

        original_find = model_configs_module.find_canonical_model_id

        def patched_find_canonical_model_id(config, pretrained_id=None, *args, **kwargs):
            config_dict = config if isinstance(config, dict) else config.to_dict()
            model_type = config_dict.get("model_type", "")

            # ProGen2 모델이면 pretrained_id(=로컬 경로)를 그대로 반환
            if model_type == "progen":
                result = pretrained_id or "progen2-medium"
                logger.info(f"find_canonical_model_id: progen detected, returning '{result}'")
                return result

            # 원본으로 폴백
            return original_find(config, pretrained_id, *args, **kwargs)

        model_configs_module.find_canonical_model_id = patched_find_canonical_model_id

        # builder.py에서도 import된 참조를 패치
        try:
            import furiosa_llm.artifact.builder as builder_module
            builder_module.find_canonical_model_id = patched_find_canonical_model_id
        except (ImportError, AttributeError):
            pass

        logger.info("Patched find_canonical_model_id for ProGen2")
    except ImportError:
        logger.warning("Could not patch find_canonical_model_id")


def _patch_model_metadata():
    """get_model_metadata_from_model_id가 canonical_model_id로 HF Hub 조회하는 것을 방지합니다.

    이전 시도에서 발견된 문제:
    - canonical_model_id가 "progen2/progen2-medium" 등으로 설정되면
      AutoConfig.from_pretrained()가 HF Hub에서 조회 시도 → 404/401 에러
    - 해결: get_config_from_pretrained_id를 패치하여 progen 모델이면 로컬에서 로드
    """
    try:
        import furiosa_llm.models.metadata as metadata_module

        original_get_config = metadata_module.get_config_from_pretrained_id

        def patched_get_config(pretrained_id, *args, **kwargs):
            pretrained_str = str(pretrained_id)

            # 이미 로컬 경로이면 그대로 진행
            if os.path.isdir(pretrained_str):
                return original_get_config(pretrained_id, *args, **kwargs)

            # progen 관련 model_id이면 로컬에서 로드 시도
            if "progen" in pretrained_str.lower():
                # 이미 등록된 ProGenConfig로 직접 생성
                from transformers import AutoConfig
                # 실제 모델 경로에서 config 로드 시도
                for candidate in [
                    pretrained_str,
                    f"/mnt/elice/datahub/models/progen2/results",
                    f"/mnt/elice/datahub/models/progen2/results/",
                ]:
                    if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "config.json")):
                        logger.info(f"Redirecting config load from '{pretrained_str}' to '{candidate}'")
                        return AutoConfig.from_pretrained(candidate)

                logger.warning(f"Could not find local config for '{pretrained_str}', trying original")

            return original_get_config(pretrained_id, *args, **kwargs)

        metadata_module.get_config_from_pretrained_id = patched_get_config
        logger.info("Patched get_config_from_pretrained_id for ProGen2")
    except ImportError:
        logger.warning("Could not patch get_config_from_pretrained_id")


def _patch_pipeline_metadata():
    """furiosa-llm의 GeneratorPipelineMetadata에 ProGen2 mask 설정을 추가합니다."""
    try:
        from furiosa_llm.models.metadata import GeneratorPipelineMetadata

        original_method = GeneratorPipelineMetadata.from_pipeline_metadata

        @staticmethod
        def patched_from_pipeline_metadata(generator_config, pipelines, pipeline_metadata):
            model_qname = generator_config.model_qname
            # ProGen2 또는 Llama 모델인 경우 직접 처리
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
                        mask_dim=3,
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
    1. HuggingFace transformers에 ProGenConfig + ProGenForCausalLM + Tokenizer 등록
    2. furiosa.models에 최적화된 ProGenForCausalLM 주입
    3. furiosa.models 레지스트리에 모델 추가
    4. find_canonical_model_id 패치 (config JSON 매칭 우회)
    5. get_config_from_pretrained_id 패치 (HF Hub 조회 → 로컬 리다이렉트)
    6. GeneratorPipelineMetadata 패치 (mask dimension)
    """
    logger.info("Registering ProGen2 with furiosa ecosystem...")

    _register_hf_config()
    _register_furiosa_models()
    _register_model_registry()
    _inject_canonical_config()
    _patch_model_metadata()
    _patch_pipeline_metadata()

    logger.info("ProGen2 registration complete!")
