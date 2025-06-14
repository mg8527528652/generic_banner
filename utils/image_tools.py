import requests
from typing import Dict, Any, Optional, Annotated
import json
import random
from langchain_core.tools import tool
import fal_client
from utils.upload1 import upload_image_to_s3
import uuid
@tool
def background_replacer(
    image_url: Annotated[str, "URL of the image to replace background for"],
    prompt: Annotated[str, "Description of the new background to generate"],
    order_id: Annotated[Optional[str], "Order ID for tracking (optional)"] = None,
    user_type: Annotated[str, "User type"] = "FREE",
    image_extension: Annotated[str, "Output image format"] = "webp",
    batch_size: Annotated[int, "Number of images to generate"] = 1,
    num_inference_steps: Annotated[int, "Number of inference steps"] = 30,
    run_post_process: Annotated[bool, "Whether to run post processing"] = True,
) -> str:
    """
    Replace the background of an image with a new background based on a text prompt.
    
    Args:
        image_url: URL of the original image
        mask_url: URL of the mask image (defines what to replace)
        prompt: Text description of the desired new background
        order_id: Optional order ID for tracking
        user_type: User type (default: "FREE")
        image_extension: Output format (default: "webp")
        batch_size: Number of images to generate (default: 1)
        num_inference_steps: AI inference steps (default: 30)
        run_post_process: Whether to apply post-processing (default: True)
    
    Returns:
        URL of the image with replaced background
    """
    base_url = "https://static-aws-ml1.phot.ai/background-replacer-comfyui"
    mask_url = background_remover(image_url)
    if order_id is None:
        order_id = str(random.randint(1, 100000000))
    
    payload = {
        "prompt": prompt,
        "image_url": image_url,
        "mask_url": mask_url,
        "order_id": order_id,
        "user_type": user_type,
        "image_extension": image_extension,
        "batch_size": batch_size,
        "num_inference_steps": num_inference_steps,
        "run_post_process": run_post_process,
    }
    
    try:
        response = requests.post(
            f"{base_url}/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        result = response.json()
        output_url = result["output_doc"]['0']['without_watermark']
        
        return output_url
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Request error: {str(e)}")
    except KeyError as e:
        raise Exception(f"Invalid response format: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")


@tool
def background_remover(
    img_url: Annotated[Optional[str], "URL of the image to remove background from"] = None,
    b64_image: Annotated[Optional[str], "Base64 encoded image string"] = None,
) -> str:
    """
    Remove the background from an image and get a mask URL.
    
    You must provide either img_url OR b64_image (not both).
    
    Args:
        img_url: URL of the image to process
        b64_image: Base64 encoded image string
    
    Returns:
        URL of the generated mask image
    """
    if not img_url and not b64_image:
        raise ValueError("Either img_url or b64_image must be provided")
    
    if img_url and b64_image:
        raise ValueError("Provide either img_url or b64_image, not both")
    
    url = 'https://static-aws-ml1.phot.ai/v1/models/transparent-bgremover-model:predict'
    headers = {"Content-Type": "application/json"}
    
    # Prepare API request
    if img_url:
        data = {
            "instances": [{
                "image": {
                    "url": img_url
                }
            }]
        }
    else:
        data = {
            "instances": [{
                "image": {
                    "b64": b64_image
                }
            }]
        }
    
    try:
        # Make API call
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        result = json.loads(response.text)
        return result['output_image_url']
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")
    except KeyError as e:
        raise Exception(f"Invalid API response format: {str(e)}")
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse API response: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")


@tool
def text_to_image_generator(
    prompt: Annotated[str, "PROFESSIONAL description for premium background generation"],
    width: Annotated[int, "Width of the generated image in pixels"] = 1024,
    height: Annotated[int, "Height of the generated image in pixels"] = 1024,
    out_path: Annotated[str, "Local file path to save the generated image"] = "generated_image.png"
) -> str:
    """
    Generate PREMIUM, PROFESSIONAL-GRADE background images for high-end banner designs.
    Creates sophisticated, commercially-viable backgrounds with professional polish.

    **QUALITY MISSION**: Produce backgrounds that rival top design agencies and premium brand campaigns.

    **BACKGROUND GENERATION GUIDELINES:**

    **PROFESSIONAL TERMINOLOGY FOR MAXIMUM QUALITY:**
    - "Cinematic lighting with dramatic shadows and highlights"
    - "Professional studio photography with sophisticated color grading"
    - "High-end commercial aesthetic with atmospheric depth"
    - "Premium brand campaign style with rich visual texture"
    - "Sophisticated gradient systems with perfect tonal balance"

    **QUALITY ENHANCEMENT KEYWORDS:**
    - "Ultra-high-resolution", "professional studio setup", "cinematic composition"
    - "Sophisticated color grading", "atmospheric depth", "premium finish"
    - "Commercial-grade", "brand campaign quality", "market-ready aesthetic"
    - "Professional lighting", "dramatic atmosphere", "rich visual texture"

    **STYLE SPECIFICATIONS:**
    - **Corporate**: Clean, professional, trustworthy atmospheric backgrounds
    - **Luxury**: Rich textures, sophisticated gradients, premium aesthetics
    - **Modern**: Contemporary design, geometric precision, sophisticated minimalism
    - **Creative**: Artistic flair with commercial viability and visual impact

    **ENHANCED PROMPT STRUCTURE:**
    "Professional [style] background with cinematic lighting, sophisticated color grading, atmospheric depth, premium finish, and commercial-grade quality. Rich visual texture, dramatic highlights, [specific elements]. High-end brand campaign aesthetic."

    Args:
        prompt: DETAILED professional background description with quality keywords
        width: Width of the generated image in pixels (default: 1024)
        height: Height of the generated image in pixels (default: 1024)
        out_path: Local file path to save the generated image (default: "generated_image.png")
    
    Returns:
        URL of the professional-grade background image
    """
    try:
        
        handler = fal_client.submit(
            "fal-ai/ideogram/v3",
            arguments={
                "prompt": prompt,
                "image_size": {
                    "width": width,
                    "height": height
                }
            },
        )

        result = handler.get()
        url = result['images'][0]['url']
        
        # Save the image to a file
        with open(out_path, 'wb') as f:
            f.write(requests.get(url).content)
        
        # Load the image as PIL Image before uploading
        from PIL import Image
        with Image.open(out_path) as pil_image:
            img = upload_image_to_s3(pil_image, str(uuid.uuid4()) + '.png')
        return img
        
    except ImportError:
        raise Exception("fal_client is not installed. Please install it with: pip install fal-client")
    except Exception as e:
        raise Exception(f"Error generating image: {str(e)}")


# List of tools for easy import
bg_tools = [background_replacer, background_remover, text_to_image_generator] 