"""
furiosa_progen2 — ProGen2 protein language model for FuriosaAI RNGD NPU.

Usage:
    # 1. Convert checkpoint (PyTorch → safetensors)
    python -m furiosa_progen2.convert_checkpoint \
        --model progen2-medium \
        --ckpt ./checkpoints/progen2-medium/ \
        --output ./checkpoints/progen2-medium-safetensors/

    # 2. NPU inference
    from furiosa_progen2.register import register_progen2
    register_progen2()

    from furiosa.llm import LLM
    llm = LLM(model="./checkpoints/progen2-medium-safetensors/")
    output = llm.generate("1MKTL")
"""
