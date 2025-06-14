import anthropic
import os
from typing import Optional
from dotenv import load_dotenv
import re
from langchain_core.tools import tool

load_dotenv()

def extract_svg_from_response(response: str) -> str:
    """
    Extract clean SVG code from Claude's response, handling various formatting issues.
    
    Args:
        response (str): Raw response from Claude
        
    Returns:
        str: Clean SVG code
    """
    # Remove markdown code block markers
    response = re.sub(r'^```(?:xml|svg)?\s*\n?', '', response, flags=re.MULTILINE)
    response = re.sub(r'\n?```\s*$', '', response, flags=re.MULTILINE)
    
    # Find SVG content using regex
    svg_match = re.search(r'(<\?xml.*?</svg>)', response, re.DOTALL)
    if svg_match:
        svg_code = svg_match.group(1)
    else:
        # Fallback: look for just the SVG tag without XML declaration
        svg_match = re.search(r'(<svg.*?</svg>)', response, re.DOTALL)
        if svg_match:
            svg_code = '<?xml version="1.0" encoding="UTF-8"?>\n' + svg_match.group(1)
        else:
            # If no proper SVG found, return the cleaned response
            svg_code = response.strip()
    
    # Clean up whitespace and ensure proper formatting
    svg_code = svg_code.strip()
    
    # Validate that it starts with XML declaration or SVG tag
    if not (svg_code.startswith('<?xml') or svg_code.startswith('<svg')):
        raise ValueError("Generated response does not contain valid SVG code")
    
    return svg_code

@tool
def svg_generator(
    description: str, 
    width: int = 400, 
    height: int = 300,
    style: str = "modern"
) -> str:
    """
    Advanced SVG generator with customizable dimensions and style.
    
    Generate scalable vector graphics based on natural language descriptions with control
    over dimensions and visual style.
    
    Args:
        description: Detailed description of what to draw (e.g., "minimalist logo with geometric shapes")
        width: Width of the SVG canvas in pixels (default: 400)
        height: Height of the SVG canvas in pixels (default: 300) 
        style: Visual style preference - "modern", "classic", "minimalist", "detailed" (default: "modern")
        
    Returns:
        str: Complete SVG code ready for use
        
    Examples:
        - svg_generator("abstract art with circles and triangles", 500, 500, "modern")
        - svg_generator("vintage car illustration", 600, 400, "classic")
        - svg_generator("simple house icon", 100, 100, "minimalist")
    """
    # Get API key from environment variable
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "Error: Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
    
    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Enhanced system prompt with style and dimension awareness
    system_prompt = f"""You are an expert SVG generator. Create clean, valid SVG code based on the description.

CRITICAL: Return ONLY the SVG code with no markdown blocks or explanations.

Requirements:
- Start with <?xml version="1.0" encoding="UTF-8"?>
- Use SVG namespace: <svg xmlns="http://www.w3.org/2000/svg">
- Set dimensions: width="{width}" height="{height}" viewBox="0 0 {width} {height}"
- Style preference: {style} (adjust colors, complexity, and design accordingly)
- Use self-closing tags and valid XML syntax

Style Guidelines:
- Modern: Clean lines, vibrant colors, geometric shapes
- Classic: Traditional colors, ornate details, balanced composition  
- Minimalist: Simple shapes, limited colors, lots of white space
- Detailed: Rich textures, complex paths, intricate designs"""
    
    # Enhanced user prompt
    user_prompt = f"Generate {style} style SVG code for: {description}"
    
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=20000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        
        response_text = message.content[0].text.strip()
        svg_code = extract_svg_from_response(response_text)
        return svg_code
        
    except Exception as e:
        return f"Error generating SVG: {str(e)}"


if __name__ == "__main__":
  
    # Test the advanced tool
    result2 = svg_generator.invoke({"description": "a 1080 * 1080 aesthetic red  background with gradient, radial circles", "width": 600, "height": 400, "style": "detailed"})
    with open("svg_code.svg", "w") as f:
        f.write(result2)
    print("Advanced SVG Tool Result:")
    print(result2[:200] + "..." if len(result2) > 200 else result2)
    