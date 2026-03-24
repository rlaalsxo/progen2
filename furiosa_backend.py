class FuriosaProGen2:
    def __init__(self, model_path, devices="npu:0"):
        from furiosa_llm import LLM
        self.llm = LLM(model_path, devices=devices)

    def generate(self, context, max_length, top_p, temp, num_samples):
        from furiosa_llm import SamplingParams
        params = SamplingParams(
            temperature=temp,
            top_p=top_p,
            max_tokens=max_length,
            n=num_samples,
        )
        outputs = self.llm.generate([context], params)
        return [out.text for out in outputs[0].outputs]

    def shutdown(self):
        self.llm.shutdown()
