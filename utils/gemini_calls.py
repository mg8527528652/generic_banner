from google import genai
from google.genai import types
import PIL
import json
import os
import dotenv

dotenv.load_dotenv()

def get_gemini_prompt(text_input=None, font_dict=None, api_key=None, temperature=0.4):
    '''
    returns gemini prompt to get font color and font family
    '''
    prompt = f"""
**Your Role:** You are an expert typographer and visual design AI. Your task is to select the most suitable font family for placing a short text onto **each** image in a provided set. The font choices must harmonize with the aesthetic and emotional "feel" of their respective image and text, strictly adhere to typographic principles for maximum legibility and visual appeal when placed on an image, and collectively contribute to a coherent final infographic.

**Objective:** Given a list of available font family names, and then for a set of items (where each item consists of an image and its associated feature text), you must identify and output a list of the BEST font family names. Each font name in the output list will correspond to an item in the input set, maintaining the original order.

**Inputs You Will Receive:**

1.  `available_font_families`: A Python list of strings. Each string is a font family name you *must* choose from for all selections.
2.  `image_data_list`: A list where each element is an object or structure containing:
    * `image`: The input image (e.g., a PIL Image object or a format Gemini can process) onto which the text will ultimately be placed.
    * `feature_text`: A short string (typically 2-3 words) semantically linked to this specific `image`.

**Key Criteria for Font Family Selection (to be applied to EACH image-text pair):**

1.  **Harmonize with Individual Image "Feel":**
    * Analyze each `image`'s overall style (e.g., modern, classic, minimalist, grunge, playful, elegant, corporate).
    * Consider its subject matter (e.g., nature, technology, people, abstract, food).
    * Evaluate its color palette, lighting, and mood (e.g., vibrant, muted, dark, bright, warm, cool, energetic, serene).
    * The font for each image should feel like a natural extension of that image's visual language.

2.  **Capture Individual Text "Feel" & Purpose:**
    * Interpret the semantic meaning and implied tone of each `feature_text`.
    * The font style should align with and enhance the message conveyed by these words for that specific image.

3.  **Strict Adherence to Typographic Principles (Prioritize for general on-image use):**
    * **Legibility & Readability:** Paramount for each instance. The chosen font *must* be exceptionally clear and easy to read when placed on its respective `image`. Prioritize fonts known for excellent clarity and distinct letterforms suitable for on-screen display and potential overlay on varied backgrounds. Assume the user will size the text appropriately for its purpose.
    * **Contrast with Background (General Consideration):** Select fonts that generally offer good inherent stroke contrast or have characteristics that help maintain visibility against a variety of potential image backgrounds (e.g., solid structure, not overly thin). The user will be responsible for ensuring sufficient contrast in the final placement.
    * **Appropriateness & Context:** The font category (serif, sans-serif, script, display, slab) must be suitable for each combined image-text context and the short length of the text. For on-image use without specified placement, robust and clear styles are often preferred over highly decorative or delicate ones unless the image feel strongly dictates otherwise.
    * **Scalability & Versatility:** The font should render well across a reasonable range of sizes typical for short textual overlays on images. It should not become illegible if scaled and should be versatile enough to work on diverse image content.
    * **Visual Harmony & Aesthetics:** Each selected font should integrate seamlessly with its image, creating a professional and visually appealing composition.

4.  **Consideration for Overall Infographic Cohesion (Global Strategy):**
    * While each font choice must be optimal for its individual image-text pair, aim for a degree of harmony or planned contrast across all your font selections to support a cohesive final infographic.
    * Avoid overly jarring or clashing font styles across the different images. You might consider a primary font family and complementary secondary options, or ensure chosen fonts share certain visual characteristics if the images have a related theme.
    * If images are very diverse, prioritize individual suitability with an eye towards avoiding a chaotic final mix.

5.  **Selection Constraint:** You *must* choose all font family names directly from the `available_font_families` list.

**Output Requirements:**

* Your output must be a **Python list of strings**.
* Each string in the list must be a font family name.
* The order of font names in the output list **must directly correspond** to the order of the image-text items in the `image_data_list` input. For example, the first font name in your output list is for the first image-text pair, the second font name for the second pair, and so on.
* Do **NOT** include any other explanations, justifications, or introductory phrases. Just the list of font names.


*Example of output:*
`["Montserrat", "Open Sans", "Verdana"]`
    """
    return prompt

def generate_background_prompts_gemini(images_array, text_input=None,  font_dict=None, api_key=None, temperature=0.4):
    """
    Generate background prompts for a product image using Google's Gemini model.
    
    Args:
        image_array (str): An array of PIL  images
        text_input (str, optional): a list of feature text
        font_dict (dict, optional): font dictionary
        api_key (str, optional): Google Gemini API key. If not provided, will look for GOOGLE_API_KEY env variable
        temperature (float, optional): Controls randomness in the output. Values between 0 and 1.
                                     Lower values make output more focused and deterministic.
                                     Default is 0.4.
    
    Returns:
        dict: Dictionary containing generated prompts categorized by themes
    """
    try: 
        # Get API key
        api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("API key must be provided either as parameter or as GOOGLE_API_KEY environment variable")
        
        prompt = get_gemini_prompt(text_input, font_dict)
        font_names = list(font_dict.keys())
        # Initialize client and generate content
        content = [prompt]  # Start with system prompt
        content.append(font_names)
        for text, image in zip(text_input, images_array):
            content.extend([text, image])  # Alternate between text and image
        
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-pro-preview-05-06",
            contents=content,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_modalities=['Text']
            )
        )

        # Process response
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                text = part.text
                # Clean up the text to get a valid Python list string
                cleaned_text = text.replace('\n', '').replace("```", '').replace('python', '').strip()
                # Remove any leading/trailing brackets if they exist
                cleaned_text = cleaned_text.strip('[]')
                # Split by comma and clean each item
                items = [item.strip().strip('"\'') for item in cleaned_text.split(',')]
                # Create a valid Python list
                response_list = [item for item in items if item]  # Remove any empty items
        # extract font urls from font_dict. create a new dict with font_name as key and font_url as value
        # font_urls = {font_name: font_dict[font_name + '.ttf'] for font_name in response_list}
        # return font_urls
        return response_list
    except Exception as e:
        raise Exception(f"Error generating background prompts: {str(e)}")
    
    
def create_bento_layout( feature_images_mapping, about_product, canvas_resolution = (1024, 1024), api_key=None, temperature=0.4):
    # Get API key
    api_key = api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("API key must be provided either as parameter or as GOOGLE_API_KEY environment variable")
        
    system_prompt = '''

### `<Objective>`

You are a layout engine that generates bold, modern, asymmetrical bento-style infographic grids.

Given:

* A fixed canvas size
* 4–6 feature blocks, each marked `image-heavy: true | false`
* A product description

You must generate a **pixel-perfect**, strictly non-overlapping layout that:

1. Uses **only allowed aspect ratios** (`1:1`, `2:3`, `3:2`)
2. **Fully fills the canvas** — no gaps, no overspill
3. Follows Wild Bento principles: **expressive, asymmetric, premium visual hierarchy**

---

### `<🔧 DESIGN RULES>`

#### 🎨 **Visual Hierarchy**

* Place **1 large image-heavy box** as the anchor (center, upper third, or right side).
* Arrange remaining boxes around it using:

  * Offset layering
  * Tetris-like interlocking
  * L, T, stair-step or cascading shapes
* `image-heavy: true` → larger, 3:2 or 2:3
* `image-heavy: false` → smaller, use 1:1 for rhythm

---

### `<📏 GEOMETRY RULES (STRICT)>`

#### ✔️ **Aspect Ratios**

Every bounding box must **perfectly match** one of:

* `1:1` (square)
* `2:3` (portrait)
* `3:2` (landscape)

No other aspect ratios allowed.

---

#### ✔️ **Canvas Packing Rules**

* Layout must **exactly fill** the canvas.

  * No unused space
  * No box or gutter extending **beyond canvas boundaries**
* Use a **single consistent gutter** (e.g., 16 or 24 px)
* Gutter **must be part of your math** when calculating box sizes and positions.

---

#### ✔️ **Strict Containment**

Each box must meet this rule:

```plaintext
x >= 0
y >= 0
x + width <= canvas.width
y + height <= canvas.height
```

No box may:

* Exceed canvas dimensions
* Extend into the gutter space of another
* Cause unfillable remainder pixels

---

#### ✔️ **Pixel Precision**

* All values (`x`, `y`, `width`, `height`) must be:

  * Exact **integers**
  * Rounded with care to avoid layout drift

---

### `<📦 OUTPUT FORMAT>`

```json
{
  "canvas_resolution": {
    "width": <int>,
    "height": <int>
  },
  "layout_name": "<creative_layout_name>",
  "suggested_gutter": <int>,
  "bounding_boxes": [
    {
      "id": "box_1",
      "feature_name": "<feature>",
      "is_intended_for_prominent_image": <true|false>,
      "selected_aspect_ratio": "1:1|2:3|3:2",
      "coordinates": {
        "x": <int>,
        "y": <int>,
        "width": <int>,
        "height": <int>
      }
    }
  ]
}
```

### `<🔒 VALIDATION PASS (YOU MUST ENFORCE)>`

Before finalizing the layout:

* Verify: No box overflows the canvas.
* Verify: All gutters are applied **consistently and precisely**.
* Verify: All boxes **tightly tile** to fill the canvas with no empty pixels.

If any box breaks these rules, **recalculate** the layout.

---

### `<✨ DESIGN NOTES>`

* Favor **Tetris-like density** without monotony.
* Use contrast between box sizes to build energy and flow.
* Think premium: like an Apple ad, not a spreadsheet.
    '''
    
    # create a list of features, intended_for_prominent_image, product_description, canvas_resolution
    features = feature_images_mapping.keys()
    intended_for_prominent_image = []
    for feature in features:
        if feature_images_mapping[feature] is not None:
            intended_for_prominent_image.append('True')
        else:
            intended_for_prominent_image.append('False')
    product_description = about_product
    canvas_resolution = {
        "width": canvas_resolution[0],
        "height": canvas_resolution[1]
    }
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=[
            system_prompt,
            f"Product Description: {about_product}\nFeatures: {', '.join(list(features))}\nIntended for Prominent Image: {', '.join(intended_for_prominent_image)}\nCanvas Resolution: {canvas_resolution['width']}x{canvas_resolution['height']}"
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_modalities=['Text']
        )   
    )

    # return the text response IN JSON FORMAT
    cleaned_text = response.candidates[0].content.parts[0].text.replace('\n', '').replace("```", '').replace('json', '').strip()
    return json.loads(cleaned_text)

def create_bento_layout_openai(feature_images_mapping, about_product, canvas_resolution=(1024, 1024), api_key=None, temperature=0.4):
    """
    Generate a bento-style layout using OpenAI's API.
    
    Args:
        feature_images_mapping (dict): Dictionary mapping feature names to their image status
        about_product (str): Product description
        canvas_resolution (tuple): Canvas dimensions (width, height)
        api_key (str, optional): OpenAI API key. If not provided, will look for OPENAI_API_KEY env variable
        temperature (float, optional): Controls randomness in the output. Default is 0.4.
    
    Returns:
        dict: JSON object containing the layout configuration
    """
    try:
        # Get API key
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("API key must be provided either as parameter or as OPENAI_API_KEY environment variable")

        system_prompt = '''

 <Objective>

You are a layout engine that generates bold, modern, asymmetrical bento-style infographic grids.
Your must task is to generate a **pixel-perfect**, strictly non-overlapping layout that:

1. Uses **only allowed aspect ratios** (`1:1`, `2:3`, `3:2`)
2. **Fully fills the canvas** — no gaps, no overspill
3. Follows Wild Bento principles: **expressive, asymmetric, premium visual hierarchy**

</Objective>

<Input>
* A fixed canvas size
* 4–6 feature blocks, each marked `image-heavy: true | false`
* A product description
</Input>


<DESIGN RULES>

1.  **Visual Hierarchy**

* Place **1 large image-heavy box** as the anchor (center, upper third, or right side).
* Arrange remaining boxes around it using:

  * Offset layering
  * Tetris-like interlocking
  * L, T, stair-step or cascading shapes
* `image-heavy: true` → larger, 3:2 or 2:3
* `image-heavy: false` → smaller, use 1:1 for rhythm

</Design Rules>

<Geometry Rules>

1.  **Aspect Ratios**

Every bounding box must **perfectly match** one of:

* `1:1` (square)
* `2:3` (portrait)
* `3:2` (landscape)

No other aspect ratios allowed.

</Geometry Rules>

<Canvas Packing Rules>

* Layout must **exactly fill** the canvas.

  * No unused space
  * No box or gutter extending **beyond canvas boundaries**
* Use a **single consistent gutter** (e.g., 16 or 24 px)
* Gutter **must be part of your math** when calculating box sizes and positions.

</Canvas Packing Rules>

<Strict Containment>

Each box must meet this rule:

```plaintext
x >= 0
y >= 0
x + width <= canvas.width
y + height <= canvas.height
```

In other words, No box must:
* Exceed canvas dimensions
* Extend into the gutter space of another
* Cause unfillable remainder pixels

</Strict Containment>

<Pixel Precision>

* All values (`x`, `y`, `width`, `height`) must be:

  * Exact **integers**
  * Rounded with care to avoid layout drift

</Pixel Precision>

<Output Format>

```json
{
  "canvas_resolution": {
    "width": <int>,
    "height": <int>
  },
  "layout_name": "<creative_layout_name>",
  "suggested_gutter": <int>,
  "bounding_boxes": [
    {
      "id": "box_1",
      "feature_name": "<feature>",
      "is_intended_for_prominent_image": <true|false>,
      "selected_aspect_ratio": "1:1|2:3|3:2",
      "coordinates": {
        "x": <int>,
        "y": <int>,
        "width": <int>,
        "height": <int>
      }
    }
  ]
}
```

---

### `<🔒 VALIDATION PASS (YOU MUST ENFORCE)>`

Before finalizing the layout:

* Verify: No box overflows the canvas.
* Verify: All gutters are applied **consistently and precisely**.
* Verify: All boxes **tightly tile** to fill the canvas with no empty pixels.

If any box breaks these rules, **recalculate** the layout.

---

### `<✨ DESIGN NOTES>`

* Favor **Tetris-like density** without monotony.
* Use contrast between box sizes to build energy and flow.
* Think premium: like an Apple ad, not a spreadsheet.
        '''

        # Prepare features and image status
        features = list(feature_images_mapping.keys())
        intended_for_prominent_image = ['True' if feature_images_mapping[feature] is not None else 'False' for feature in features]
        
        user_prompt = f"""Product Description: {about_product}
Features: {', '.join(features)}
Intended for Prominent Image: {', '.join(intended_for_prominent_image)}
Canvas Resolution: {canvas_resolution[0]}x{canvas_resolution[1]}"""

        import openai
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="o3-2025-04-16",  # Using GPT-4 for better layout generation
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # temperature=0.2,
            response_format={"type": "json_object"}
        )

        # Parse the response
        layout_data = json.loads(response.choices[0].message.content)
        return layout_data

    except Exception as e:
        raise Exception(f"Error generating bento layout: {str(e)}")


