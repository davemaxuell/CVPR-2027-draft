# import torch
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from PIL import Image

import json
import os
import time

# Initialize Qwen model (lazy loading - will be loaded on first use)
# _qwen_model = None
# _qwen_processor = None


def _get_qwen_model(model):
    global _qwen_model, _qwen_processor
    # if _qwen_model is None or _qwen_processor is None:
    if model == "Qwen2.5-VL-7B-Instruct":
        _qwen_model_dir = "/home/zhan1624/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/5b5eecc7efc2c3e86839993f2689bbbdf06bd8d4"
        print("Loading Qwen2.5-VL-7B-Instruct model...")
    elif model == "Qwen2.5-VL-32B-Instruct":
        _qwen_model_dir = "/home/zhan1624/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/6bcf1c9155874e6961bcf82792681b4f4421d2f7"
        # os.environ["FLASH_ATTENTION_2_ENABLED"] = "0"
        print("Loading Qwen2.5-VL-32B-Instruct model...")

    _qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        _qwen_model_dir,
        torch_dtype="auto",
        # torch_dtype=torch.float16,
        device_map="auto",
    )
    _qwen_processor = AutoProcessor.from_pretrained(_qwen_model_dir, use_fast=True)
    print(f"{model} model loaded successfully")
    print(
        "The model is loaded on: ", next(_qwen_model.parameters()).device
    )  # Check the device of the model parameters
    print("-" * 25)
    return _qwen_model, _qwen_processor


def qwen_infer(
    system, text, image_desp, image_list, model, processor, max_tokens=1000, response_format=None
):
    """
    Run inference with Qwen2.5-VL-7B-Instruct model.

    Args:
        system: System prompt
        text: Input text prompt
        image_list: List of image paths
        model: Model name (ignored, always uses Qwen2.5-VL-7B-Instruct)
        max_tokens: Maximum number of tokens to generate
        response_format: Response format specification (ignored for Qwen)

    Returns:
        Tuple of (generated_text, token_info)
    """

    user_content = []

    # Add images
    for i, image_path in enumerate(image_list):
        if image_path is not None and os.path.exists(image_path):
            user_content.append({"type": "text", "text": f"Image {i}:"+image_desp[i]})
            user_content.append({"type": "image", "image": image_path})
        else:
            print(f"Warning: Image {i} not found at path: {image_path}")

    # Add text prompt
    user_content.append({"type": "text", "text": text})

    # print("User content for Qwen inference:", json.dumps(user_content, indent=4))  # For debugging

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": user_content},
    ]

    # # Display the messages for debugging
    # print(json.dumps(messages, indent=4)) # Convert to JSON format if needed

    # Track token usage (approximate)
    start_time = time.time()

    # Preparation for inference
    text_inputs = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # image_inputs, video_inputs = process_vision_info(messages)
    image_inputs, _ = process_vision_info(messages)  # Ignore video inputs for now

    # print("-"*25)
    # print("Processed text inputs:", text_inputs)  # For debugging
    # print("Processed image inputs:", image_inputs)  # For debugging
    # print("-"*25)

    inputs = processor(
        text=[text_inputs],
        images=image_inputs,
        # videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Deterministic generation (temperature=0)
        )

        # Extract only the newly generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode the generated text
        answer = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    # Create a simple token usage object (format similar to OpenAI for compatibility)
    elapsed = time.time() - start_time
    token_info = {
        "completion_tokens": len(generated_ids_trimmed[0]),
        "prompt_tokens": len(inputs.input_ids[0]),
        "total_tokens": len(inputs.input_ids[0]) + len(generated_ids_trimmed[0]),
        "elapsed_time": elapsed,
    }

    return answer, token_info


def qwen_infer_desription(
    system, text, image_list, model, processor, max_tokens=2000, response_format=None
):
    """
    Run inference with Qwen2.5-VL-7B-Instruct model.

    Args:
        system: System prompt
        text: Input text prompt
        image_list: List of image paths
        model: Model name (ignored, always uses Qwen2.5-VL-7B-Instruct)
        max_tokens: Maximum number of tokens to generate
        response_format: Response format specification (ignored for Qwen)

    Returns:
        Tuple of (generated_text, token_info)
    """

    user_content = []

    # Add images
    for i, image_path in enumerate(image_list):
        if image_path is not None and os.path.exists(image_path):
            user_content.append({"type": "text", "text": f"Image {i}:"})
            user_content.append({"type": "image", "image": image_path})
        else:
            print(f"Warning: Image {i} not found at path: {image_path}")

    # Add text prompt
    user_content.append({"type": "text", "text": text})

    # print("User content for Qwen inference:", json.dumps(user_content, indent=4))  # For debugging

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": user_content},
    ]

    # # Display the messages for debugging
    # print(json.dumps(messages, indent=4)) # Convert to JSON format if needed

    # Track token usage (approximate)
    start_time = time.time()

    # Preparation for inference
    text_inputs = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # image_inputs, video_inputs = process_vision_info(messages)
    image_inputs, _ = process_vision_info(messages)  # Ignore video inputs for now

    # print("-"*25)
    # print("Processed text inputs:", text_inputs)  # For debugging
    # print("Processed image inputs:", image_inputs)  # For debugging
    # print("-"*25)

    inputs = processor(
        text=[text_inputs],
        images=image_inputs,
        # videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Deterministic generation (temperature=0)
        )

        # Extract only the newly generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode the generated text
        answer = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    # Create a simple token usage object (format similar to OpenAI for compatibility)
    elapsed = time.time() - start_time
    token_info = {
        "completion_tokens": len(generated_ids_trimmed[0]),
        "prompt_tokens": len(inputs.input_ids[0]),
        "total_tokens": len(inputs.input_ids[0]) + len(generated_ids_trimmed[0]),
        "elapsed_time": elapsed,
    }

    return answer, token_info
