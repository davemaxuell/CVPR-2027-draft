import torch
from transformers import AutoTokenizer, AutoProcessor
from typing import Any, Mapping

from transformers import MllamaForConditionalGeneration, AutoProcessor

from models.base_mllm import BaseMLLM

class LlamaVision(BaseMLLM):
    def __init__(
        self,
        max_length: int = 512,
    ):
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load tokenizer, processor, and multimodal LLM checkpoint (Qwen2.5-VL)
        try:
            model_name = "meta-llama/Llama-3.2-11B-Vision"

            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.processor = AutoProcessor.from_pretrained(model_name)

            print(f"Successfully loaded Llama Vision model: {model_name}")
        except Exception as e:
            raise OSError(f"Failed to load Llama Vision model ({model_name}): {e}")

        # Generation parameters
        self.max_length = max_length

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "ckpt": self.model.config._name_or_path,
            "max_length": self.max_length,
        }

    def __call__(self, input: dict, prompt: str) -> str:
        self.model.eval()
        with torch.no_grad():
            if self.processor.tokenizer.eos_token_id is not None:
                eos_token_id = self.processor.tokenizer.eos_token_id
            else:
                # Fallback if no specific EOS token ID is found, though unlikely for Llama
                eos_token_id = self.model.config.eos_token_id # Or a common one like 2 for Llama 3
                print(f"Warning: Using model config's EOS token ID: {eos_token_id}")
            text = f"<|image|><|begin_of_text|>{prompt}"
            inputs = self.processor(input['image'], text, return_tensors="pt").to(self.model.device)

            input_ids_len = inputs.input_ids.shape[1]
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128, # Use a small number here
                eos_token_id=eos_token_id, # Add this
                pad_token_id=eos_token_id # Often good practice to set pad_token_id to eos_token_id for generation
            )

            generated_ids_trimmed = generated_ids[0, input_ids_len:]
            answer = self.processor.decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            return answer