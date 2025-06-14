import base64
import uuid
from io import BytesIO
from openai import OpenAI
from langchain_core.tools import tool
try:
    from utils.upload1 import upload_image_to_s3
except:
    from upload1 import upload_image_to_s3
from PIL import Image

# OpenAI client globally initialized
client = OpenAI()

@tool
def generate_image_tool(prompt: str, size: str = "1024x1024") -> dict:
    """
    Generates a complex, high-quality illustration image based on a given text prompt.
    The image is uploaded to storage and a link is returned along with base64 data.

    This tool is designed to generate **non-photographic, artistic, or conceptual illustrations** — e.g. cartoon scenes, product illustrations, infographic components, icons, or abstract art. It does not generate realistic human photographs.

    ========================
    **Supported Resolutions:**
    - '1024x1024'  (Square, default)
    - '1792x1024'  (Wide landscape)
    - '1024x1792'  (Vertical portrait)

    *Other resolutions are not supported and may return errors.*

    ========================
    **Prompt Guidelines:**
    - Be descriptive and explicit. The model responds well to detailed scene descriptions.
    - Use keywords like: "cartoon", "vector style", "digital illustration", "minimalist art", "flat design", etc.
    - Avoid requesting realistic photographs or celebrity likenesses.
    - Specify color schemes or mood where relevant (e.g. "warm colors", "blue background", "isometric view").
    - Example:  
        `"A digital illustration of a rocket ship launching, cartoon style, with colorful smoke trails, blue sky, and minimal flat design."`

    ========================
    **Arguments:**
    - prompt (str): A detailed description of the image you want generated.
    - size (str): Image resolution. Defaults to '1024x1024'. Must be one of the supported resolutions listed above.

    ========================
    **Returns:**
    - link (str): The storage URL of the uploaded image.
    """
    object_id = str(uuid.uuid4())
    
    try:
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=size,
            background="transparent"  # Note: transparency not fully supported yet, may require post-processing.
        )
        
        # Decode base64 into bytes
        image_bytes = base64.b64decode(result.data[0].b64_json)

        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")

        # Upload PIL image to storage
        link = upload_image_to_s3(image, object_id + ".png")

        return link

    except Exception as e:
        return {
            "link": "",
            "base64_string": "",
            "prompt": prompt,
            "error": str(e)
        }


if __name__ == "__main__":
    result = generate_image_tool.invoke({"prompt": "a three dimentional illustration of a man with a beard and a hat", "size": "1024x1024"})
    # with open("transparent_illustration.png", "wb") as f:
    #     f.write(result["base64_string"])
    print(result["link"])