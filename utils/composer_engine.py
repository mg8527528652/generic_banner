import os
import json
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# --- Simplified Composer System Prompt ---
COMPOSER_SYSTEM_PROMPT = """
You are a professional banner designer. Create a Fabric.js v5.3.0 JSON for a banner.

CRITICAL FABRIC.JS v5.3.0 SYNTAX RULES:
1. ALL elements must stay within canvas boundaries (0 to width, 0 to height)
2. Gradient colorStops MUST use array format: [{"offset": 0, "color": "#hex"}, {"offset": 1, "color": "#hex"}]
3. Text objects use "textbox" type, NOT "text"
4. CTA text goes INSIDE CTA buttons/shapes using group objects
5. All colors must be valid hex codes or rgba() values
6. originX/originY must be "left", "center", or "right" / "top", "center", "bottom"
7. All numeric values must be numbers, not strings

FABRIC.JS v5.3.0 GRADIENT SYNTAX (MANDATORY):
```json
"fill": {
  "type": "linear",
  "x1": 0, "y1": 0, "x2": 100, "y2": 0,
  "colorStops": [
    {"offset": 0, "color": "#FF0000"},
    {"offset": 1, "color": "#0000FF"}
  ]
}
```

NEVER USE: "colorStops": {"0": "#color", "1": "#color"} - THIS IS INVALID!

Return ONLY valid Fabric.js v5.3.0 JSON - no explanations.
"""

# --- Validator System Prompt ---
VALIDATOR_SYSTEM_PROMPT = """
You are a Fabric.js v5.3.0 syntax validator. Check JSON for specific syntax errors.

CRITICAL VALIDATION CHECKS:
1. Elements within canvas boundaries (check left+width ≤ canvas.width, top+height ≤ canvas.height)
2. Gradient colorStops MUST be array format: [{"offset": number, "color": "string"}]
3. Text objects must use "textbox" type
4. CTA text properly grouped with buttons
5. All colors are valid hex (#RRGGBB) or rgba() format
6. All required Fabric.js properties present
7. No invalid property values

COMMON FABRIC.JS v5.3.0 ERRORS TO CATCH:
- ❌ "colorStops": {"0": "#color"} 
- ✅ "colorStops": [{"offset": 0, "color": "#color"}]
- ❌ "type": "text"
- ✅ "type": "textbox"
- ❌ Elements extending beyond canvas
- ❌ Missing required properties like version, originX, originY

RESPOND WITH:
- "PASS" if JSON is valid Fabric.js v5.3.0 syntax
- "CONTINUE: [specific syntax errors]" if issues found

Be specific about exact syntax problems.
"""

# --- Apply Feedback System Prompt ---
FEEDBACK_SYSTEM_PROMPT = """
You are a Fabric.js v5.3.0 syntax fixer. Apply feedback to fix specific JSON syntax errors.

FABRIC.JS v5.3.0 SYNTAX FIXES:
1. Fix gradient colorStops to array format:
   ❌ "colorStops": {"0": "#color", "1": "#color"}
   ✅ "colorStops": [{"offset": 0, "color": "#color"}, {"offset": 1, "color": "#color"}]

2. Fix text objects:
   ❌ "type": "text"
   ✅ "type": "textbox"

3. Fix canvas boundaries:
   - Ensure left + width ≤ canvas width
   - Ensure top + height ≤ canvas height

4. Fix invalid properties:
   - Ensure all colors are valid hex or rgba
   - Ensure numeric values are numbers not strings
   - Add missing required properties

Apply the exact feedback while:
1. Keeping ALL existing assets and content
2. Maintaining design quality and layout
3. Only fixing the specific syntax errors mentioned

Return ONLY the corrected Fabric.js v5.3.0 JSON - no explanations.
"""

def create_composer_llm():
    """Create o3 LLM for composition"""
    return ChatOpenAI(
        model="o3-2025-04-16",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=50000,
        reasoning_effort="high"
    )

def create_validator_llm():
    """Create GPT-4.1 LLM for validation"""
    return ChatOpenAI(
        model="gpt-4",  # Using GPT-4 as requested (GPT-4.1 not available in API)
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=2000,
        temperature=0.1
    )

def create_feedback_llm():
    """Create o3 LLM for applying feedback"""
    return ChatOpenAI(
        model="o3-2025-04-16",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=50000,
        reasoning_effort="high"
    )

def validate_banner(fabric_json: str, banner_prompt: str, resolution: list) -> str:
    """
    Validate the banner JSON and return feedback.
    Returns: "PASS" or "CONTINUE: [feedback]"
    """
    try:
        validator_llm = create_validator_llm()
        
        validation_prompt = f"""
BANNER REQUIREMENTS:
{banner_prompt}

CANVAS: {resolution[0]}x{resolution[1]} pixels

FABRIC.JS JSON TO VALIDATE:
{fabric_json}

Check if this banner meets all requirements. Focus on:
- Elements staying within {resolution[0]}x{resolution[1]} bounds
- Text placement and readability
- CTA text inside buttons
- Good asset utilization
- Canvas coverage

Respond with "PASS" or "CONTINUE: [specific issues to fix]"
"""
        
        messages = [
            {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
            {"role": "user", "content": validation_prompt}
        ]
        
        response = validator_llm.invoke(messages)
        return response.content.strip()
        
    except Exception as e:
        print(f"Validation error: {e}")
        return "CONTINUE: Validation failed, please review banner composition"

def apply_feedback(fabric_json: str, feedback: str, banner_prompt: str, assets: list, resolution: list) -> str:
    """
    Apply validator feedback to improve the banner JSON.
    """
    try:
        feedback_llm = create_feedback_llm()
        
        feedback_prompt = f"""
ORIGINAL REQUIREMENTS:
{banner_prompt}

AVAILABLE ASSETS:
{json.dumps(assets, indent=2)}

CANVAS: {resolution[0]}x{resolution[1]} pixels

CURRENT FABRIC.JS JSON:
{fabric_json}

VALIDATOR FEEDBACK:
{feedback}

Apply the feedback to fix the issues. Keep all assets and maintain design quality.
Return ONLY the improved Fabric.js JSON.
"""
        
        messages = [
            {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
            {"role": "user", "content": feedback_prompt}
        ]
        
        response = feedback_llm.invoke(messages)
        return response.content.strip()
        
    except Exception as e:
        print(f"Feedback application error: {e}")
        return fabric_json  # Return original if error

@tool
def compose_fabric_banner(
    banner_prompt: str,
    assets: list,
    resolution: list = [1080, 1080]
) -> str:
    """
    Compose a Fabric.js v5.3.0 JSON banner with validation feedback loop.
    Args:
        banner_prompt (str): The design brief from banner_design_researcher tool.
        assets (list): List of asset dicts (type, url, description, etc).
        resolution (list): [width, height] of the banner.
    Returns:
        str: Fabric.js v5.3.0 JSON string.
    """
    print(f'🎨 Composing banner with validation loop...')
    print(f'📐 Resolution: {resolution[0]}x{resolution[1]}')
    print(f'🎯 Assets available: {len(assets)}')
    
    try:
        composer_llm = create_composer_llm()
        
        # Initial composition
        composition_prompt = f"""
CREATE BANNER FOR:
{banner_prompt}

CANVAS: {resolution[0]}x{resolution[1]} pixels

AVAILABLE ASSETS:
{json.dumps(assets, indent=2)}

Create a professional Fabric.js banner that:
- Uses all provided assets effectively
- Keeps ALL elements within {resolution[0]}x{resolution[1]} bounds
- Places CTA text inside CTA buttons
- Fills the canvas well
- Has readable text with good contrast

Return ONLY valid Fabric.js v5.3.0 JSON.
"""
        
        messages = [
            {"role": "system", "content": COMPOSER_SYSTEM_PROMPT},
            {"role": "user", "content": composition_prompt}
        ]
        
        print("🔄 Step 1: Creating initial composition...")
        response = composer_llm.invoke(messages)
        current_json = response.content.strip()
        
        # Clean JSON (remove markdown if present)
        if current_json.startswith("```"):
            current_json = current_json.split("\n", 1)[1]
        if current_json.endswith("```"):
            current_json = current_json.rsplit("\n", 1)[0]
        current_json = current_json.strip()
        
        # Validation and feedback loop (max 5 iterations)
        max_iterations = 5
        for iteration in range(max_iterations):
            print(f"🔍 Step {iteration + 2}: Validating banner (iteration {iteration + 1}/{max_iterations})...")
            
            validation_result = validate_banner(current_json, banner_prompt, resolution)
            
            if validation_result.startswith("PASS"):
                print(f"✅ Banner validated successfully after {iteration + 1} iteration(s)!")
                return current_json
            
            elif validation_result.startswith("CONTINUE:"):
                feedback = validation_result.replace("CONTINUE:", "").strip()
                print(f"⚠️  Feedback received: {feedback}")
                
                if iteration < max_iterations - 1:  # Don't apply feedback on last iteration
                    print(f"🔧 Applying feedback...")
                    improved_json = apply_feedback(current_json, feedback, banner_prompt, assets, resolution)
                    
                    # Clean improved JSON
                    if improved_json.startswith("```"):
                        improved_json = improved_json.split("\n", 1)[1]
                    if improved_json.endswith("```"):
                        improved_json = improved_json.rsplit("\n", 1)[0]
                    improved_json = improved_json.strip()
                    
                    current_json = improved_json
                else:
                    print(f"🔄 Max iterations reached, returning current version")
                    break
            else:
                print(f"⚠️  Unexpected validation response: {validation_result}")
                break
        
        print(f"🏁 Composition complete after {max_iterations} iterations")
        return current_json
        
    except Exception as e:
        print(f"❌ Composer error: {e}")
        return f"Error: {e}"

# --- Main function for direct execution ---
if __name__ == "__main__":
    # Test the simplified composer
    banner_prompt = """
    Create a 'Grand Opening' banner for a new coffee shop called 'The Daily Grind.'
    
    Requirements:
    - Main headline: "GRAND OPENING"
    - Subheadline: "THE DAILY GRIND"
    - Description: "Join us for craft coffee, community, and celebration!"
    - Date: "Monday, May 1, 2024"
    - Address: "123 Main Street"
    - CTA button: "JOIN US"
    - Professional coffee shop aesthetic
    """
    
    assets = [
        {
            "type": "background",
            "url": "https://example.com/coffee-background.jpg",
            "description": "Coffee shop interior background"
        },
        {
            "type": "font",
            "url": "https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700",
            "description": "Modern sans-serif font"
        }
    ]
    
    resolution = [1080, 1080]
    
    # Test 1: Normal banner generation
    print("🧪 TEST 1: Normal Banner Generation")
    fabric_json = compose_fabric_banner.invoke({
        "banner_prompt": banner_prompt,
        "assets": assets,
        "resolution": resolution
    })
    
    print("\n" + "="*60)
    print("FINAL FABRIC.JS JSON:")
    print("="*60)
    print(fabric_json[:500] + "..." if len(fabric_json) > 500 else fabric_json)
    
    # Save to file
    with open("test_banner_output.json", "w") as f:
        f.write(fabric_json)
    print(f"\n💾 Output saved to test_banner_output.json")
    
    # Test 2: Validate problematic JSON with wrong gradient syntax
    print("\n" + "="*60)
    print("🧪 TEST 2: Gradient Syntax Validation")
    print("="*60)
    
    problematic_json = '''
    {
      "version": "5.3.0",
      "width": 1024,
      "height": 1024,
      "objects": [
        {
          "type": "textbox",
          "text": "Test Gradient",
          "left": 100,
          "top": 100,
          "width": 300,
          "height": 50,
          "originX": "left",
          "originY": "top",
          "fill": {
            "type": "linear",
            "x1": 0, "y1": 0, "x2": 0, "y2": 85,
            "colorStops": {
              "0": "#1641A6",
              "0.5": "#48BCC7", 
              "1": "#36D399"
            }
          }
        }
      ]
    }
    '''
    
    validation_result = validate_banner(problematic_json, "Simple gradient test", [1024, 1024])
    print(f"Validation Result: {validation_result}")
    
    if validation_result.startswith("CONTINUE:") and "colorStops" in validation_result:
        print("✅ Validator correctly caught gradient syntax error!")
        
        # Test feedback application
        print("\n🔧 Applying feedback...")
        feedback = validation_result.replace("CONTINUE:", "").strip()
        corrected_json = apply_feedback(problematic_json, feedback, "Simple gradient test", [], [1024, 1024])
        
        print(f"Corrected JSON:")
        print(corrected_json)
        
        # Check if the corrected JSON has proper array format
        if '"colorStops": [' in corrected_json and '"offset":' in corrected_json and '"color":' in corrected_json:
            print("✅ Gradient syntax successfully fixed to array format!")
        else:
            print("❌ Gradient syntax not properly corrected")
            
        # Re-validate
        revalidation = validate_banner(corrected_json, "Simple gradient test", [1024, 1024])
        print(f"\nRe-validation Result: {revalidation}")
        
        if revalidation.startswith("PASS"):
            print("✅ Complete success: Validation → Fix → Re-validation passed!")
        else:
            print("⚠️  Still has some issues, but gradient syntax should be fixed")
    else:
        print("❌ Validator failed to catch gradient syntax error")
