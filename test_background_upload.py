#!/usr/bin/env python3
"""
Test script to demonstrate background image creation and upload functionality
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFilter
import base64
from io import BytesIO
import uuid

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_gradient_background(width, height, colors, style="linear"):
    """Create a gradient background image"""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)
    
    if style == "linear":
        # Create linear gradient
        for i in range(height):
            # Calculate color interpolation
            ratio = i / height
            if len(colors) >= 2:
                r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
                g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
                b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
            else:
                r, g, b = colors[0]
            draw.line([(0, i), (width, i)], fill=(r, g, b))
    
    # Add some texture/noise for aesthetic appeal
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    return img

def pil_to_base64(image, format='JPEG'):
    """Convert PIL image to base64"""
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def save_background_locally(colors, width, height, filename="test_background.jpg"):
    """Create and save background image locally for testing"""
    try:
        # Convert hex colors to RGB
        rgb_colors = [hex_to_rgb(color) for color in colors[:2]]
        
        # Create gradient background
        bg_image = create_gradient_background(width, height, rgb_colors)
        
        # Save locally
        bg_image.save(filename)
        print(f"✅ Background image saved locally: {filename}")
        
        # Also get base64 for upload simulation
        base64_image = pil_to_base64(bg_image)
        print(f"✅ Base64 image created (length: {len(base64_image)} characters)")
        
        return filename, base64_image
        
    except Exception as e:
        print(f"❌ Background creation failed: {e}")
        return None, None

if __name__ == "__main__":
    print("🎨 Testing Background Image Creation")
    print("=" * 50)
    
    # Test colors
    test_colors = ["#FF6B9D", "#45B7D1", "#96CEB4"]
    
    # Create background
    local_file, base64_data = save_background_locally(
        colors=test_colors,
        width=1200,
        height=600,
        filename="demo_background.jpg"
    )
    
    if local_file:
        print(f"\n📋 Results:")
        print(f"   Local file: {local_file}")
        print(f"   Base64 ready for upload: {'Yes' if base64_data else 'No'}")
        print(f"   Colors used: {test_colors}")
        
        # Simulate what would happen with Wasabi credentials
        print(f"\n🔧 With Wasabi credentials, this would:")
        print(f"   1. Upload {local_file} to Wasabi S3")
        print(f"   2. Return a public URL like: https://your-bucket.s3.wasabisys.com/backgrounds/gradient_{uuid.uuid4().hex[:8]}.jpg")
        print(f"   3. Use that URL in the Fabric.js JSON as background image")
        
    else:
        print("❌ Failed to create background image")
    
    print("\n" + "=" * 50)
    print("🎯 To enable full functionality, add these environment variables:")
    print("   WASABI_ACCESS_KEY_ID=your_access_key")
    print("   WASABI_SECRET_ACCESS_KEY=your_secret_key") 
    print("   WASABI_BUCKET_NAME=your_bucket_name")
    print("   WASABI_ENDPOINT_URL=https://s3.wasabisys.com/") 