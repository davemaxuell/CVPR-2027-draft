from google import genai
from google.genai import types
import mimetypes
import os

from tenacity import retry, wait_random_exponential, stop_after_attempt

# print(os.getenv('GEMINI_API_KEY'))

client = genai.Client(api_key='')



# a helper that retries the generate_content call
@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    reraise=True
)
def _generate_with_backoff(model: str, contents: list):
    return client.models.generate_content(
        model=model,
        contents=contents,
    )

def gemini_infer(system_prompt: str,
                 text_prompt: str,
                 image_desp: list[str],
                 image_list: list[str],
                 model="gemini-2.5-pro"):
    # Build a flat contents list
    contents: list[types.Part | str] = []
    
    # 1) system prompt (if you really need it)
    contents.append(system_prompt)
    # 2) user text
    contents.append(text_prompt)
    
    # 3) each image, prefixed by its label
    for i, img_path in enumerate(image_list):
        # skip None or empty entries
        if not img_path:
            print(f"Skipping missing image at index {i}")
            continue
        # ensure a valid path
        if not isinstance(img_path, (str, os.PathLike)) or not os.path.exists(img_path):
            print(f"Warning: Invalid or missing image path at index {i}: {img_path}")
            continue
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        contents.append(f"Description {i}: {image_desp[i]}")
        # label
        contents.append(f"Image {i}:")
        # upload and append reference
        uploaded_file = client.files.upload(file=img_path)
        contents.append(uploaded_file)
    
    response = _generate_with_backoff(model, contents)
    
    return response.text
    

if __name__ == "__main__":
    
    system_prompt = "You are a helpful assistant."
    text_prompt = "What is the content of the image?"
    image_list = ["path/to/image1.jpg", "path/to/image2.jpg"]  # Replace with actual image paths
    
   
    response = gemini_infer(system_prompt, text_prompt, image_list)
    print(response)