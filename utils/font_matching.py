import google.generativeai as genai
import json
import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Import the decorator from langchain_core
from langchain_core.tools import tool

load_dotenv()

# Configure Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("Warning: GOOGLE_API_KEY environment variable not set.")


@tool
def select_best_font_url(banner_prompt: str, json_path: str = "data/fonts_data.json") -> str:
    """Selects the single best font URL from a database to match a banner's theme.

    Use this tool when you need to choose the most appropriate font for a design based on its description.
    This tool analyzes the banner's mood, style, and typography requirements to find the best match in the font database.
    For best results, provide a rich and detailed description.

    Example 'banner_prompt': "A banner for a luxury watch brand. The theme is elegant, timeless, and exclusive. The main headline requires a classic serif font."

    Args:
        banner_prompt (str): A detailed text description of the banner's requirements, including theme (e.g., 'vintage', 'futuristic'), mood (e.g., 'playful', 'corporate'), and desired font characteristics (e.g., 'a bold sans-serif', 'a flowing script').
        json_path (str): The optional file path to the JSON font database. Defaults to 'fonts_database.json'.
    """
    # 1. Load the font data from the JSON file
    try:
        with open(json_path, 'r') as f:
            font_data = json.load(f)
    except FileNotFoundError:
        return f"Error: The font database at {json_path} was not found."
    except json.JSONDecodeError:
        return f"Error: The font database at {json_path} is not a valid JSON file."

    # 2. Prepare the font data for the API (exclude URLs)
    fonts_for_api = []
    for font in font_data:
        font_copy = font.copy()
        font_copy.pop('url', None)
        fonts_for_api.append(font_copy)

    # 3. Construct the enhanced system prompt for premium font selection
    system_prompt = f"""
    You are a PREMIUM typography director specializing in high-end brand identity and commercial design. Your expertise rivals top design agencies like Pentagram, Sagmeister & Walsh, and IDEO.

    **MISSION**: Select the perfect font that elevates the banner to SELLABLE, PROFESSIONAL-GRADE quality.

    **TYPOGRAPHY ANALYSIS FRAMEWORK:**
    - **Brand Personality**: Does the font convey luxury, trust, innovation, playfulness, sophistication?
    - **Emotional Impact**: What feelings does the typography evoke?
    - **Market Positioning**: Premium, mainstream, or accessible luxury aesthetic?
    - **Readability**: Clear hierarchy and excellent legibility across devices
    - **Contemporary Relevance**: Current design trends and timeless appeal

    **FONT SELECTION CRITERIA:**
    1. **Brand Alignment**: Font personality matches the banner's message and audience
    2. **Visual Hierarchy**: Supports clear information architecture
    3. **Aesthetic Quality**: Professional finish and sophisticated character design
    4. **Versatility**: Works well at multiple sizes and weights
    5. **Market Appeal**: Enhances commercial value and perceived quality

    **STYLE MATCHING GUIDELINES:**
    - **Luxury/Premium**: Sophisticated serifs, refined sans-serifs, elegant scripts
    - **Modern/Tech**: Clean geometrics, contemporary sans-serifs, futuristic fonts
    - **Creative/Artistic**: Unique character, artistic flair, distinctive personality
    - **Corporate/Professional**: Trustworthy, clean, excellent readability
    - **Playful/Youth**: Dynamic, energetic, approachable character
    - **Classic/Traditional**: Timeless elegance, established credibility

    **AVAILABLE FONTS DATABASE:**
    {json.dumps(fonts_for_api, indent=2)}

    **SELECTION MANDATE:**
    Analyze the banner description through the lens of premium design standards. Consider the font's ability to:
    - Create emotional resonance with the target audience
    - Support the overall brand strategy and message
    - Maintain readability and professional appearance
    - Enhance the commercial appeal of the final design

    Return ONLY the `filename` of the single best font choice. No explanations.
    """

    # 4. Initialize the model and send the request
    try:
        # Using a newer model name as an example, adjust if needed
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            system_instruction=system_prompt,
        )
        # Low temperature for deterministic, accurate selection
        generation_config = {"temperature": 0.8}
        response = model.generate_content(f"BANNER DESCRIPTION:\n{banner_prompt}", generation_config=generation_config)

        best_font_filename = response.text.strip()

        # 5. Look up the URL corresponding to the returned filename (Verification step)
        for font in font_data:
            if font.get('filename') == best_font_filename:
                return font.get('url', f"Error: URL not found for filename {best_font_filename}.")

        return f"Error: Model recommended filename '{best_font_filename}', which was not found in the font database."

    except Exception as e:
        if "API key not valid" in str(e) or "API_KEY_INVALID" in str(e):
            return "Error: Google API Key is not configured or is invalid."
        return f"An unexpected error occurred: {e}"


# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    print("✨ This script demonstrates the enhanced tool with a better docstring. ✨\n")

    # The @tool decorator reads the new, detailed docstring.
    print(select_best_font_url.invoke("A banner for a luxury watch brand. The theme is elegant, timeless, and exclusive. The main headline requires a classic serif font."))