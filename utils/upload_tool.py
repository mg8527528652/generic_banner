from langchain.tools import tool
from utils.upload1 import upload_image_to_s3

@tool
def upload_image_to_s3_tool(image_src: str, object_name: str) -> str:
    """Uploads an image to S3 storage. The image can be from a URL or local file path.
    
    Args:
        image_src: Source of the image (URL or local file path)
        object_name: Name of the object to be stored in S3 (e.g., 'folder/image.webp')
        
    Returns:
        str: URL of the uploaded image
    """
    try:
        url = upload_image_to_s3(image_src, object_name)
        return url
    except Exception as e:
        return f"Error uploading image: {str(e)}" 