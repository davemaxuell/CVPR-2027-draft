import torch
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from typing import Any, Mapping

from qwen_vl_utils import process_vision_info
from models.base_mllm import BaseMLLM

class Qwen2_5_VL(BaseMLLM):
    def __init__(
        self,
        num_beams: int = 1,
        max_length: int = 512,
        min_length: int = 1,
    ):
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load tokenizer, processor, and multimodal LLM checkpoint (Qwen2.5-VL)
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            ).to(self.device)

            self.model.eval() # Set model to evaluation mode
            print(f"Successfully loaded Qwen2.5-VL model: {model_name}")
        except Exception as e:
            raise OSError(f"Failed to load Qwen2.5-VL model ({model_name}): {e}")

        # Generation parameters
        self.num_beams = num_beams
        self.max_length = max_length
        self.min_length = min_length

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "ckpt": self.model.config._name_or_path,
            "max_length": self.max_length,
            "min_length": self.min_length,
            "num_beams": self.num_beams
        }

    def _construct_input(self, input: dict, prompt: str):
         # Build the multimodal message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": input["image"]},  # PIL.Image or file path
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        return inputs

    def __call__(self, input: dict, prompt: str) -> str:
        self.model.eval()
        with torch.no_grad():
            inputs = self._construct_input(input, prompt)

            # Generate output text
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_length, num_beams=self.num_beams, min_length=self.min_length)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]