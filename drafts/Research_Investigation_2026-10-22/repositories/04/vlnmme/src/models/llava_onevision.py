import torch
import requests
from transformers import AutoTokenizer, AutoProcessor, LlavaOnevisionForConditionalGeneration
from typing import Any, Mapping, List
from PIL import Image


from models.base_mllm import BaseMLLM

class Llava_OneVision(BaseMLLM):
    def __init__(
        self,
        max_length: int = 512,
    ):
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load tokenizer, processor, and multimodal LLM checkpoint (Qwen2.5-VL)
        model_name = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval() # Set model to evaluation mode
            print(f"Successfully loaded Llava-OV-Qwen2 model: {model_name}")
        except Exception as e:
            raise OSError(f"Failed to load Llava-OV-Qwen2 model ({model_name}): {e}")

        self.max_length = max_length

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "ckpt": self.model.config._name_or_path,
            "max_length": self.max_length,
        }

    def _construct_input(self, input: dict, prompt: str):
        # Build the multimodal message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=text,
            images=input['image'],
            return_tensors="pt",
        ).to(self.device)

        return inputs

    def __call__(self, input: dict, prompt: str) -> str:
        self.model.eval()
        with torch.no_grad():
            inputs = self._construct_input(input, prompt)
        
            # Generate output text
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_length)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]