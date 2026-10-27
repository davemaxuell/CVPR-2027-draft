from openai import OpenAI
import time
import io
import base64
from typing import Any, Mapping
from PIL import Image

from models.base_mllm import BaseMLLM


class GPT5(BaseMLLM):
    def __init__(
        self,
        api_key: str = "",
        max_output_tokens: int = 512,
        temperature: float = 0.0,
        model_name: str = "gpt-5.1"
    ):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

        print(f"Successfully initialized GPT model: {self.model_name}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model_name,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }

    def _encode_image(self, img: Image.Image) -> str:
        """Convert PIL image → base64 JPEG data URI"""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _construct_input(self, input: dict, prompt: str):
        """
        Build the exact multimodal structure the Responses API expects.
        """

        image = input["image"]
        data_uri = self._encode_image(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt
                    },
                    {
                        "type": "input_image",
                        "image_url": data_uri
                    }
                ]
            }
        ]

        return messages

    def __call__(self, input: dict, prompt: str) -> str:
        messages = self._construct_input(input, prompt)

        max_retries = 10
        backoff = 1

        for attempt in range(max_retries):
            try:
                response = self.client.responses.create(
                    model=self.model_name,
                    input=messages,
                    max_output_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                )

                return response.output_text.strip()

            except Exception as e:
                if "rate" in str(e).lower() or "timeout" in str(e).lower():
                    print(f"[GPT-5] Server busy (attempt {attempt+1}/{max_retries}). Waiting {backoff}s...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue

                raise RuntimeError(f"GPT-5 API error: {e}")

        raise RuntimeError("GPT-5 API repeatedly failed after retries.")
