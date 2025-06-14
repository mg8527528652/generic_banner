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
    Generates PREMIUM, PROFESSIONAL-GRADE illustrations for high-end banner designs.
    Creates sophisticated, commercially-viable graphics with transparent backgrounds.

    **QUALITY MISSION**: Produce illustrations that rival top design agencies and premium brand identity systems.

    ========================
    **SUPPORTED RESOLUTIONS:**
    - '1024x1024'  (Square, default)
    - '1024x1536'  (Wide landscape) 
    - '1536x1024'  (Vertical portrait)

    ========================
    **PREMIUM ILLUSTRATION GUIDELINES:**

    **PROFESSIONAL TERMINOLOGY FOR BETTER QUALITY:**
    - "Professional vector illustration with sophisticated gradients"
    - "Contemporary graphic design with premium finish and clean lines"
    - "High-end brand identity style with mathematical precision"
    - "Sophisticated color palette with harmonious relationships"
    - "Commercial-grade illustration with market appeal"

    **STYLE SPECIFICATIONS:**
    - **Modern**: Geometric precision, sophisticated gradients, clean minimalism
    - **Luxury**: Elegant details, rich textures, premium color palettes
    - **Corporate**: Professional aesthetics, trustworthy visual language
    - **Creative**: Artistic flair with commercial viability
    - **Tech**: Futuristic elements, digital sophistication

    **QUALITY MARKERS TO INCLUDE:**
    - "Professional finish", "sophisticated gradients", "premium quality"
    - "Commercial-grade", "brand identity style", "market-ready"
    - "Mathematical precision", "balanced composition", "visual hierarchy"
    - "Contemporary design", "high-end aesthetic", "professional polish"

    **AVOID AMATEUR TERMS:**
    - Basic descriptions like "simple cartoon" or "basic design"
    - Amateur styling requests
    - Low-quality or unprofessional aesthetic descriptions

    ========================
    **ENHANCED PROMPT STRUCTURE:**
    Use this format for maximum quality:
    "Professional [style] illustration of [subject] with sophisticated gradients, premium finish, contemporary design aesthetic, balanced composition, and commercial-grade quality. Clean lines, harmonious color palette, transparent background."

    ========================
    **Arguments:**
    - prompt (str): DETAILED professional illustration description
    - size (str): Image resolution from supported options

    **Returns:**
    - link (str): Storage URL of the professional-grade illustration
    """
    object_id = str(uuid.uuid4())
    # find nearest aspect ratio to the size and supported resolutions
    width, height = map(int, size.split("x"))
    supported_resolutions = [(1024, 1024), (1024, 1536), (1536, 1024)]
    nearest_resolution = min(supported_resolutions, key=lambda x: abs(x[0] - width) + abs(x[1] - height))
    inferred_size = f"{nearest_resolution[0]}x{nearest_resolution[1]}"
    
    try:
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=inferred_size,
            background="transparent"  # Note: transparency not fully supported yet, may require post-processing.
        )
        
        # Decode base64 into bytes
        image_bytes = base64.b64decode(result.data[0].b64_json)

        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_bytes)).convert("RGBA")
        image = image.resize((width, height))
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