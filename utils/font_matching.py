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

    # 3. Construct the system prompt for the Gemini model
    system_prompt = f"""
    You are a professional typography expert. Your task is to select the single best font filename from the provided JSON list to match the user's banner description.
    Analyze the banner's theme, typography requirements, and overall mood.
    Review the list of available fonts, paying close attention to their descriptions and themes.
    Return ONLY the `filename` of the one best font. Do not provide any explanation or other text.
    ---
    AVAILABLE FONTS (JSON):
    {json.dumps(fonts_for_api, indent=2)}
    ---
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