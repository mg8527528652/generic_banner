import base64
import requests
import socket
import uuid
import os
import json
from PIL import Image
from io import BytesIO
import numpy as np
# hostname = socket.gethostname()
# IPAddr = socket.gethostbyname(hostname)
from boto3 import resource
import dotenv

dotenv.load_dotenv()

aws_access_key_id = os.getenv("WASABI_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("WASABI_SECRET_ACCESS_KEY")
bucket_name = os.getenv("WASABI_BUCKET_NAME")
s3_base_url = os.getenv("WASABI_ENDPOINT_URL") 
# s3_cdn_url = "https://photai-s3-bucket.apyhi.com"
s3_client = resource(service_name='s3', 
                    endpoint_url=s3_base_url,
                    region_name='us-east-2',
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_session_token=None
                    )
s3_bucket = s3_client.Bucket(bucket_name)
def upload_wasabi_rest(base64_string, object_name):
    try:
        s3_client.Object(bucket_name, object_name).put(
            Body=base64.b64decode(base64_string), 
            ContentType='image/png', 
            ACL='public-read',
            ContentDisposition = "attachment; filename="+str(object_name.split("/")[-1]))
        return f"{s3_base_url}{bucket_name}/{object_name}"
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return 500


def pil_to_base64(image, format='PNG'):
    if isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return img_str
    elif isinstance(image, np.ndarray):
        # convert numpy array to PIL image
        
        image = Image.fromarray(image)
        return pil_to_base64(image, format)

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        # return encoded_image.decode('utf-8')
        return encoded_image
    
def upload_image_to_s3(image_src, object_name):
    if isinstance(image_src, str) and image_src.startswith("http"):
        image_src = Image.open(requests.get(image_src, stream=True).raw)
    encoded_image = pil_to_base64(image_src)
    url = upload_wasabi_rest(encoded_image, object_name)
    return url

if __name__ == "__main__":
    image_src = "https://th.bing.com/th/id/OSK.HEROi9emigND1FjTGZhVSpzKHxDxCNvM5l9UxChMsUOVBuU?w=472&h=280&c=1&rs=2&o=6&dpr=2&pid=SANGAM"
    url = upload_image_to_s3(image_src, "test.webp")
    print(url)


