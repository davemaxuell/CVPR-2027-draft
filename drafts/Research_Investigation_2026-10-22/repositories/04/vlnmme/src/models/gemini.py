from google.genai import Client
from google.genai import types
import time
import google.genai.errors as genai_errors

from typing import Any, Mapping
from PIL import Image
import io

from models.base_mllm import BaseMLLM

class Gemini2_5Pro(BaseMLLM):
    def __init__(
        self,
        api_key: str="",
        max_output_tokens: int = 128,
        temperature: float = 0.0,
    ):
        # Initialize the client
        self.client = Client(api_key=api_key)
        self.model_name = 'gemini-2.5-pro'
        
        # Generation parameters
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        
        print(f"Successfully initialized Gemini 2.5 Pro model: {self.model_name}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model_name,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }

    def _construct_input(self, input: dict, prompt: str):
        # Get PIL Image from input
        image = input["image"]

        # Convert PIL Image to bytes (PNG format)
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)

        contents = [
            types.Part.from_bytes(data=buf.read(), mime_type='image/png'),
            prompt
        ]

        return contents

    def __call__(self, input: dict, prompt: str) -> str:
        contents = self._construct_input(input, prompt)

        max_retries = 10
        backoff = 1

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                )
                return response.text.strip()

            except genai_errors.ServerError as e:
                # Only retry if temporary (503, 500, 429)
                if e.code in [500, 503, 429]:
                    print(f"[Gemini] Temporary server issue (attempt {attempt+1}/{max_retries}). Retrying in {backoff}s...")
                    time.sleep(backoff)
                    continue
                else:
                    raise RuntimeError(f"Gemini server error: {e}")

            except Exception as e:
                raise RuntimeError(f"Error during Gemini API call: {e}")

        raise RuntimeError("Gemini API repeatedly failed after retries.")