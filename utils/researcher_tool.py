import os 
import io
import base64
import requests
import json
from dotenv import load_dotenv
from PIL import Image

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

# --- Step 1: Load Environment Variables ---
# Make sure you have a .env file with your OPENAI_API_KEY and TAVILY_API_KEY
load_dotenv()

# --- Step 2: Create an In-Memory Cache ---
# This simple dictionary will hold the base64 data between tool calls,
# preventing it from cluttering the agent's chat history.
IMAGE_CACHE = {}

# --- Step 3: Define the Tools ---

@tool
def save_image_to_cache(image_url: str) -> str:
    """
    Downloads an image from a URL, converts it to base64, and saves it
    to an in-memory cache. 
    
    This tool should be called for each image URL you want to analyze.
    
    Args:
        image_url: The URL of the image to download and cache.

    Returns:
        A success message confirming the image has been cached, using the URL as its ID.
    """
    if not image_url:
        return "Error: No image URL provided."
    try:
        print(f"--- Tool Call: Caching image from URL: {image_url} ---")
        response = requests.get(image_url, timeout=20)
        response.raise_for_status()  # Raise an exception for bad status codes (like 404)

        # Use Pillow to validate it's a real image and get its format
        image_bytes = io.BytesIO(response.content)
        image = Image.open(image_bytes)
        
        # Prepare a buffer to save the image into
        buffer = io.BytesIO()
        image.save(buffer, format=image.format or "PNG")
        final_bytes = buffer.getvalue()
        
        # Encode the final bytes to base64
        base64_string = base64.b64encode(final_bytes).decode('utf-8')
        
        # Store the base64 string in the cache, using the URL as the unique key
        IMAGE_CACHE[image_url] = base64_string
        
        return f"Success: Image from URL '{image_url}' has been downloaded and stored."
    
    except requests.exceptions.RequestException as e:
        return f"Error: Network issue downloading image from {image_url}. Details: {e}"
    except Exception as e:
        return f"Error: Failed to process image from {image_url}. It may not be a valid image file. Details: {e}"
@tool
def analyze_images_from_cache(image_urls: list[str], user_query: str, resolution: list[int]) -> str:
    """
    Analyzes one or more images that have been previously saved to the cache.

    This tool makes a direct call to the multimodal LLM with the image data.
    It should be called only ONCE, after all desired images have been cached.

    Args:
        image_urls: A list of the image URLs that were successfully cached.
        user_query: The original user intent or query text.
        resolution: Desired banner resolution as [width, height].

    Returns:
        A highly detailed design description as a string (minimum 150 words).
    """
    print(f"--- Tool Call: Analyzing {len(image_urls)} images from cache ---")

    analysis_llm = ChatOpenAI(
        model="gpt-4.1-2025-04-14",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        reasoning_effort="high",
        temperature=0.3  # Lower temperature for more consistent, professional briefs
    )

    # Build the resolution context
    width, height = resolution
    resolution_context = f"The target banner resolution is {width}x{height} pixels."

    # Build the analysis instruction
    analysis_prompt_text = """
You are a PREMIUM banner design director for a high-end AI design system. Your mission is to create SELLABLE, PROFESSIONAL-GRADE design briefs that rival top-tier design agencies like Pentagram, IDEO, or Sagmeister & Walsh.

**QUALITY STANDARDS:**
- Think Dribbble "Popular" section or Behance "Featured Projects"
- Modern, sophisticated, market-ready designs worth $500-2000+
- Rich visual depth with professional polish
- Strong brand impact and commercial appeal

**DESIGN ANALYSIS FRAMEWORK:**
Analyze the reference images through these professional lenses:
- **Visual Hierarchy**: How do the best designs guide the eye?
- **Color Psychology**: What emotions and associations do colors create?
- **Typography Systems**: How do premium designs handle font relationships?
- **Composition Techniques**: Golden ratio, rule of thirds, visual balance
- **Market Positioning**: Premium, mainstream, or budget aesthetic?

**COMPREHENSIVE DESIGN BRIEF STRUCTURE:**

1. **Brand Strategy & Message:**
   - Primary message and emotional tone
   - Target audience psychographics
   - Brand personality (sophisticated, playful, trustworthy, innovative)
   - Market positioning (premium, accessible luxury, mainstream)

2. **Typography System (PREMIUM FOCUS):**
   - **Headline**: Exact font style, dramatic sizing (8-12% of canvas height)
   - **Subheadline**: Complementary font, 60-70% of headline size
   - **Body Text**: Clean, readable, 40-50% of headline
   - **Accent Text**: Special elements, 70-80% of headline
   - **Typography Mood**: Modern geometric, classic serif, artistic script, etc.
   - **Letter spacing, line height, text effects specifications**

3. **Professional Color Strategy:**
   - **Primary Colors**: 2-3 dominant colors with hex codes
   - **Secondary Palette**: 2-3 supporting colors
   - **Accent Colors**: 1-2 high-impact colors for CTAs/highlights
   - **Color Psychology**: Emotional impact and brand association
   - **Contrast Ratios**: Ensure accessibility and readability
   - **Gradient Specifications**: Advanced color transitions if needed

4. **Sophisticated Layout Architecture:**
   - **Grid System**: How elements align and relate mathematically
   - **Visual Weight Distribution**: Balanced asymmetry or centered symmetry
   - **Focal Point Strategy**: Primary, secondary, tertiary attention areas  
   - **White Space Management**: Strategic negative space for premium feel
   - **Element Positioning**: Exact placement with mathematical precision
   - **Margins and Padding**: Consistent spacing relationships

5. **Advanced Visual Elements:**
   - **Background Concept**: Atmospheric, textural, gradient, or photographic
   - **Graphic Elements**: Icons, illustrations, decorative patterns
   - **Visual Style**: Minimalist, maximalist, flat, dimensional, textural
   - **Artistic Treatment**: Photography style, illustration approach, abstraction level
   - **Depth and Dimension**: Shadows, overlays, layering, 3D effects

6. **Professional Effects & Finishing:**
   - **Lighting Concept**: Dramatic shadows, soft glows, directional lighting
   - **Texture Application**: Paper, fabric, metal, glass effects
   - **Blend Modes**: Multiply, screen, overlay applications
   - **Filter Effects**: Blur, sharp, color adjustments
   - **Visual Hierarchy Effects**: Size, color, contrast, positioning

7. **Market-Ready Specifications:**
   - **Resolution Optimization**: Pixel-perfect scaling strategies
   - **Asset Integration**: How multiple elements work together cohesively
   - **Brand Consistency**: Maintainable visual standards
   - **Conversion Focus**: Elements that drive action/engagement
   - **Platform Adaptability**: Cross-platform visual effectiveness

8. **Alignment with User Query:**
   - **Exact Match**: Align design elements directly with user query
   - **Contextual Fit**: Ensure design elements support user intent
   - **Brand Alignment**: Maintain consistent brand identity, if provided
   - **Market Fit**: Ensure design resonates with target audience

**EXECUTION MANDATES:**
- Generate EXACT text content (no placeholders like "insert here")
- Specify mathematical relationships (1.618 golden ratio, 40px margins, etc.)
- Define precise color values and combinations
- Detail sophisticated visual effects and treatments
- Create production-ready specifications for premium execution

**PREMIUM QUALITY MARKERS:**
✓ Sophisticated color harmonies with professional contrast
✓ Typography that creates emotional resonance
✓ Composition that guides attention strategically  
✓ Visual effects that add depth without clutter
✓ Brand impact that commands premium pricing
✓ Technical specifications for flawless execution

Your design brief will power a premium AI pipeline - make it extraordinary, specific, and commercially valuable.

Output ONLY the complete design specification in the format above.
"""


    # Build full content parts for LLM input
    content_parts = [
        {"type": "text", "text": f"User Query: {user_query}"},
        {"type": "text", "text": resolution_context},
        {"type": "text", "text": analysis_prompt_text}
    ]

    retrieved_count = 0
    for url in image_urls:
        if url in IMAGE_CACHE:
            base64_image = IMAGE_CACHE[url]
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
            retrieved_count += 1
        else:
            print(f"Warning: URL '{url}' not found in cache. It will be skipped.")

    if retrieved_count == 0:
        return "Error: No valid images were found in the cache. Please ensure `save_image_to_cache` was called successfully first."

    try:
        message = HumanMessage(content=content_parts)
        response = analysis_llm.invoke([message])
        # print(response.content)
        return response.content
    except Exception as e:
        return f"Error: The analysis call to the LLM failed. Details: {e}"


# --- Step 4: Initialize Tools, LLM, and Agent ---

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

tavily_search_tool = TavilySearch(api_key=TAVILY_API_KEY, max_results=15, include_images=True, search_depth="basic", include_domains=["https://www.canva.com/banners/templates", "https://www.freepik.com/", "https://in.pinterest.com/"])
agent_llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0.2)
all_tools = [tavily_search_tool, save_image_to_cache, analyze_images_from_cache]
agent = create_react_agent(agent_llm, all_tools)

# --- Step 5: Define the System Message to guide the Agent ---

system_message = (
    """You are a methodical design research agent. Your purpose is to gather visual intelligence and synthesize it into a structured design brief. You operate with a clear, stateful, three-phase process.
    **Phase 1: SEARCH**
    1.  Your first action is ALWAYS to use the `tavily_search` tool to find image URLs relevant to the user's request. Draft search queries based on the user request and the product url and logo url if provided, to extract relevant layout banners.
    2.  Your goal is to gather an inventory of at least 8 high-quality, distinct image URLs.
    3.  To achieve this, you may need to call `tavily_search` multiple times with refined search queries. For example, if the user asks for "cricket banner," you can search for "cricket tournament poster," "modern cricket graphics," etc., to find varied examples.
    4.  Once you have compiled a list of at least 8 URLs in your thoughts, you will transition to the next phase.

    **Phase 2: SAVE**
    1.  For each of the unique URLs in your inventory, you will now call the `save_image_to_cache` tool.
    2.  You must maintain a list of the URLs that were *successfully* saved.
    3.  **Error Handling**: If a `save_image_to_cache` call returns an error, you will simply disregard that URL and continue to the next one. DO NOT attempt to find a replacement URL. DO NOT stop the process.

    **Phase 3: ANALYZE & FINISH**
    1.  After attempting to save every URL from your initial inventory, your final action MUST be to call the `analyze_images_from_cache` tool.
    2.  The `image_urls` argument for this tool must be the list of URLs that you confirmed were *successfully* saved in Phase 2.
    3.  The `user_query` argument for this tool must be the original user request.
    4.  The `resolution` argument for this tool must be the desired banner resolution as (width, height).
    3.  The String output from this final tool call is your complete and final answer. You MUST output this String directly, without any additional text, formatting, or explanation. Your job is finished.

    **CRITICAL RULES:**
    - Never analyze images yourself; you lack this capability.
    - Strictly follow the SEARCH -> SAVE -> ANALYZE sequence.
    - Your final output must only be the valid String object from the `analyze_images_from_cache` tool.
    """
)

# --- Step 6: Define the main execution function with streaming ---

@tool
def banner_design_researcher(user_request: str, resolution: list[int], product_url_provided: bool = False, logo_url_provided : bool = False) -> str:
    """
    Researches and analyzes banner designs to create a detailed design brief.

    Args:
        user_request: The original user request.
        resolution: The desired banner resolution as (width, height).
        product_url_provided: Whether a product URL was provided.
        logo_url_provided: Whether a logo URL was provided.

    Returns:
        A detailed design brief as a string.
    """
    try:
        IMAGE_CACHE.clear()
        tavily_search_tool = TavilySearch(api_key=TAVILY_API_KEY, max_results=15, include_images=True, search_depth="basic", include_domains=["https://www.canva.com/banners/templates", "https://www.freepik.com/", "https://in.pinterest.com/"])
        agent_llm = ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0.2)
        all_tools = [tavily_search_tool, save_image_to_cache, analyze_images_from_cache]
        agent = create_react_agent(agent_llm, all_tools)
        # combine user request with product url and logo url if provided
        user_content = user_request
        if product_url_provided:
            user_content += f"\nProduct URL: {product_url_provided}"
        if logo_url_provided:
            user_content += f"\nLogo URL: {logo_url_provided}"
        if resolution:
            user_content += f"\nResolution: {str(resolution[0])}x{str(resolution[1])}"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        
        # MODIFICATION: Reverted to invoke() for direct execution
        result = agent.invoke({"messages": messages}, {"recursion_limit": 35})
        # Extract the final response from the result dictionary
        if result and "messages" in result:
            last_message = result["messages"][-1]
            if hasattr(last_message, 'content'):
                print(last_message.content)
                return last_message.content
        
        return "Error: Could not generate design brief"
        
    except Exception as e:
        return f"Error: Banner design research failed. Details: {e}"

if __name__ == "__main__":
    user_request = (
        "create an awareness banner that motivates people to become  instagram influencers. create a 1080x1080 banner.  it would be good if background is an image of a rich woman counting money and sipping wine"
    )
    
    creative_brief = banner_design_researcher.invoke({"user_request": user_request})
    
    print("\n--- Final Creative Brief (JSON) ---")

    print(creative_brief)
