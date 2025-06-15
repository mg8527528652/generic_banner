import os
import json
import re
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# --- Enhanced Composer System Prompt ---
COMPOSER_SYSTEM_PROMPT = """
You are a PREMIUM banner designer creating Fabric.js v5.3.0 JSON. Your goal is to PRECISELY FOLLOW the detailed design brief's layout specifications.

CRITICAL DESIGN BRIEF INTERPRETATION:
1. **EXACT TEXT PLACEMENT**: Follow the design brief's "Element Positioning" and "Layout Architecture" specifications precisely
2. **MATHEMATICAL PRECISION**: Implement specified margins, padding, and grid relationships exactly as described
3. **TYPOGRAPHY HIERARCHY**: Use exact font sizes, positioning, and relationships specified in the brief
4. **VISUAL FLOW**: Follow the "Focal Point Strategy" and "Visual Weight Distribution" instructions
5. **SPACING SYSTEM**: Implement the "White Space Management" and spacing relationships precisely

LAYOUT IMPLEMENTATION RULES:
- Parse design brief for SPECIFIC positioning instructions (e.g., "40px margins", "golden ratio proportions")
- Follow EXACT text placement and alignment specifications from the brief
- Implement the described "Grid System" with mathematical precision
- Respect "Visual Hierarchy" order: Primary headline → Secondary text → Body → CTA
- Use specified "Typography System" sizing ratios (e.g., "60-70% of headline size")
- Follow "Color Strategy" for text contrast and readability

FABRIC.JS v5.3.0 TECHNICAL REQUIREMENTS:
1. MUST include canvas dimensions: {"version": "5.3.0", "width": X, "height": Y, "objects": [...]}
2. ALL elements must stay within canvas boundaries (0 to width, 0 to height)
3. Gradient colorStops MUST use array format: [{"offset": 0, "color": "#hex"}, {"offset": 1, "color": "#hex"}]
4. Text objects use "textbox" type, NOT "text"
5. CTA text goes INSIDE CTA buttons/shapes using group objects
6. All colors must be valid hex codes or rgba() values
7. originX/originY must be "left", "center", or "right" / "top", "center", "bottom"
8. All numeric values must be numbers, not strings

TEXT POSITIONING BEST PRACTICES (CRITICAL):
- Use consistent originX/originY for predictable positioning ("left", "top" recommended)
- Calculate text positions based on canvas grid system from design brief
- **PREVENT TEXT OVERLAPS**: Ensure minimum 40px vertical spacing between text elements
- **CALCULATE TEXT HEIGHTS**: Font size × line height × text lines = actual text height
- **VERIFY POSITIONING**: Check that text elements don't overlap with other elements
- Consider text width and line height in positioning calculations
- Position text to avoid background image focal points unless intentionally overlaying
- **STACK TEXT VERTICALLY**: Place text elements in clear vertical sequence with adequate spacing

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

**EXECUTION PRIORITY**: Design brief specifications override generic design rules. Follow the brief's layout instructions EXACTLY.

Return ONLY valid Fabric.js v5.3.0 JSON - no explanations.
"""

# --- Enhanced Validator System Prompt ---
VALIDATOR_SYSTEM_PROMPT = """
You are a Fabric.js v5.3.0 design quality validator. Focus on DESIGN BRIEF ADHERENCE and LAYOUT QUALITY.

The JSON has already passed programmatic validation for:
- Canvas dimensions and boundaries
- Gradient syntax
- Required properties

Your job is to validate DESIGN BRIEF COMPLIANCE and QUALITY:

1. **DESIGN BRIEF ADHERENCE**:
   - Does text placement follow the specified "Element Positioning" guidelines?
   - Are margins and spacing consistent with the "Layout Architecture" specifications?
   - Does typography hierarchy match the "Typography System" requirements?
   - Are grid system and mathematical relationships properly implemented?

2. **TEXT POSITIONING QUALITY**:
   - Are text elements properly spaced and non-overlapping?
   - Do text positions respect the visual hierarchy described in the brief?
   - Is text positioned to avoid background image conflicts?
   - Are text elements within safe zones and properly aligned?

3. **DESIGN QUALITY**:
   - Text readability and contrast against backgrounds
   - Visual hierarchy and professional balance
   - Asset placement and utilization effectiveness
   - CTA button effectiveness and visibility
   - Overall premium design execution

RESPOND WITH:
- "PASS" if the banner properly follows the design brief specifications and has good design quality
- "CONTINUE: [specific improvements needed]" if the banner doesn't follow the brief's layout specifications or has design issues

Focus on ensuring the banner MATCHES the detailed design brief instructions, especially for text placement, spacing, and layout architecture.
"""

# --- Enhanced Feedback System Prompt ---
FEEDBACK_SYSTEM_PROMPT = """
You are a Fabric.js v5.3.0 design improver. Apply specific design feedback to enhance banner quality and ENSURE DESIGN BRIEF COMPLIANCE.

The JSON is syntactically correct. Apply DESIGN IMPROVEMENTS with focus on DESIGN BRIEF ADHERENCE:

1. **LAYOUT CORRECTIONS**:
   - Fix text positioning to match design brief's "Element Positioning" specifications
   - Implement proper margins and spacing per "Layout Architecture" guidelines
   - Correct typography hierarchy per "Typography System" requirements
   - Apply grid system and mathematical relationships from the brief

2. **TEXT PLACEMENT FIXES**:
   - Eliminate text overlapping and positioning conflicts
   - Ensure proper spacing between text elements (minimum 20-40px)
   - Position text elements according to visual hierarchy specifications
   - Align text to avoid background image focal points when appropriate

3. **DESIGN QUALITY ENHANCEMENTS**:
   - Improve text contrast and readability
   - Better visual hierarchy and balanced spacing
   - Enhanced asset placement and sizing
   - More effective CTA design and positioning
   - Professional polish and premium execution

CONSTRAINTS:
- Keep ALL existing assets and content from the original design
- Maintain valid Fabric.js v5.3.0 syntax at all times
- Stay within canvas boundaries (all elements must fit)
- PRIORITIZE following the design brief's specific layout instructions
- Only apply the specific feedback provided

**EXECUTION PRIORITY**: Focus on making the banner match the design brief's layout specifications EXACTLY, especially for text positioning and spacing.

Return ONLY the improved Fabric.js v5.3.0 JSON - no explanations.
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
    """Create GPT-4 LLM for design validation"""
    return ChatOpenAI(
        model="gpt-4",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=2000,
        temperature=0.1
    )

def create_feedback_llm():
    """Create o3 LLM for applying design feedback"""
    return ChatOpenAI(
        model="o3-2025-04-16",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=50000,
        reasoning_effort="high"
    )

# --- ENHANCED PROGRAMMATIC VALIDATION FUNCTIONS ---

def validate_json_structure(fabric_json: str) -> Tuple[bool, List[str]]:
    """Validate basic JSON structure and parsing"""
    try:
        data = json.loads(fabric_json)
        errors = []
        
        # Check required root properties
        if not isinstance(data, dict):
            errors.append("Root JSON must be an object")
        
        if 'version' not in data:
            errors.append("Missing 'version' property")
        elif data['version'] != "5.3.0":
            errors.append(f"Version must be '5.3.0', got '{data.get('version')}'")
        
        if 'objects' not in data:
            errors.append("Missing 'objects' array")
        elif not isinstance(data['objects'], list):
            errors.append("'objects' must be an array")
        
        return len(errors) == 0, errors
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {str(e)}"]
    except Exception as e:
        return False, [f"JSON structure error: {str(e)}"]

def validate_canvas_dimensions(data: Dict, resolution: List[int]) -> Tuple[bool, List[str]]:
    """Validate canvas dimensions are present and correct"""
    errors = []
    
    if 'width' not in data:
        errors.append("Missing canvas 'width' property")
    elif data['width'] != resolution[0]:
        errors.append(f"Canvas width mismatch: expected {resolution[0]}, got {data['width']}")
    
    if 'height' not in data:
        errors.append("Missing canvas 'height' property")  
    elif data['height'] != resolution[1]:
        errors.append(f"Canvas height mismatch: expected {resolution[1]}, got {data['height']}")
    
    return len(errors) == 0, errors

def validate_element_boundaries(data: Dict, resolution: List[int]) -> Tuple[bool, List[str]]:
    """Validate all elements stay within canvas boundaries"""
    errors = []
    canvas_width, canvas_height = resolution
    
    def check_object_bounds(obj: Dict, obj_index: int, parent_name: str = ""):
        obj_name = f"{parent_name}object[{obj_index}]" if parent_name else f"object[{obj_index}]"
        
        left = obj.get('left', 0)
        top = obj.get('top', 0)
        width = obj.get('width', 0)
        height = obj.get('height', 0)
        
        # Handle scaling
        scale_x = obj.get('scaleX', 1)
        scale_y = obj.get('scaleY', 1)
        effective_width = width * scale_x
        effective_height = height * scale_y
        
        # Check right boundary
        if left + effective_width > canvas_width:
            errors.append(f"{obj_name} extends beyond right boundary: {left} + {effective_width} = {left + effective_width} > {canvas_width}")
        
        # Check bottom boundary  
        if top + effective_height > canvas_height:
            errors.append(f"{obj_name} extends beyond bottom boundary: {top} + {effective_height} = {top + effective_height} > {canvas_height}")
        
        # Check negative positions
        if left < 0:
            errors.append(f"{obj_name} has negative left position: {left}")
        if top < 0:
            errors.append(f"{obj_name} has negative top position: {top}")
        
        # Check grouped objects
        if obj.get('type') == 'group' and 'objects' in obj:
            for i, sub_obj in enumerate(obj['objects']):
                check_object_bounds(sub_obj, i, f"{obj_name}.group.")
    
    # Check all objects
    for i, obj in enumerate(data.get('objects', [])):
        check_object_bounds(obj, i)
    
    return len(errors) == 0, errors

def validate_gradient_syntax(data: Dict) -> Tuple[bool, List[str]]:
    """Validate gradient colorStops use correct array format"""
    errors = []
    
    def check_gradient(obj: Any, path: str = ""):
        if isinstance(obj, dict):
            # Check if this is a gradient with colorStops
            if obj.get('type') in ['linear', 'radial'] and 'colorStops' in obj:
                color_stops = obj['colorStops']
                
                # Check if using invalid object format
                if isinstance(color_stops, dict):
                    errors.append(f"Invalid gradient colorStops format at {path}: must be array format [{{\"offset\": 0, \"color\": \"#hex\"}}], not object format")
                
                # Check if using correct array format
                elif isinstance(color_stops, list):
                    for i, stop in enumerate(color_stops):
                        if not isinstance(stop, dict):
                            errors.append(f"Gradient colorStop[{i}] at {path} must be an object")
                        else:
                            if 'offset' not in stop:
                                errors.append(f"Gradient colorStop[{i}] at {path} missing 'offset' property")
                            if 'color' not in stop:
                                errors.append(f"Gradient colorStop[{i}] at {path} missing 'color' property")
            
            # Recursively check nested objects
            for key, value in obj.items():
                check_gradient(value, f"{path}.{key}" if path else key)
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_gradient(item, f"{path}[{i}]" if path else f"[{i}]")
    
    check_gradient(data)
    return len(errors) == 0, errors

def validate_text_objects(data: Dict) -> Tuple[bool, List[str]]:
    """Validate text objects use correct 'textbox' type"""
    errors = []
    
    def check_text_type(obj: Dict, path: str = ""):
        if obj.get('type') == 'text':
            errors.append(f"Text object at {path} uses deprecated 'text' type, should use 'textbox'")
        
        # Check grouped objects
        if obj.get('type') == 'group' and 'objects' in obj:
            for i, sub_obj in enumerate(obj['objects']):
                check_text_type(sub_obj, f"{path}.group[{i}]")
    
    for i, obj in enumerate(data.get('objects', [])):
        check_text_type(obj, f"objects[{i}]")
    
    return len(errors) == 0, errors

def validate_color_format(data: Dict) -> Tuple[bool, List[str]]:
    """Validate color values are in correct format"""
    errors = []
    
    def is_valid_color(color: str) -> bool:
        if isinstance(color, str):
            # Check hex format
            if re.match(r'^#[0-9A-Fa-f]{6}$', color):
                return True
            # Check rgba format
            if re.match(r'^rgba?\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*(,\s*[01]?\.?\d*)?\s*\)$', color):
                return True
        return False
    
    def check_colors(obj: Any, path: str = ""):
        if isinstance(obj, dict):
            # Check fill colors
            if 'fill' in obj:
                fill = obj['fill']
                if isinstance(fill, str) and not is_valid_color(fill):
                    errors.append(f"Invalid color format at {path}.fill: '{fill}'")
            
            # Check stroke colors
            if 'stroke' in obj:
                stroke = obj['stroke'] 
                if isinstance(stroke, str) and not is_valid_color(stroke):
                    errors.append(f"Invalid color format at {path}.stroke: '{stroke}'")
            
            # Recursively check nested objects
            for key, value in obj.items():
                check_colors(value, f"{path}.{key}" if path else key)
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_colors(item, f"{path}[{i}]" if path else f"[{i}]")
    
    check_colors(data)
    return len(errors) == 0, errors

def validate_text_overlaps(data: Dict) -> Tuple[bool, List[str]]:
    """Validate that text elements don't overlap with each other or other elements"""
    errors = []
    
    def get_element_bounds(obj: Dict, parent_left: int = 0, parent_top: int = 0) -> Dict:
        """Calculate the actual bounding box of an element"""
        left = obj.get('left', 0) + parent_left
        top = obj.get('top', 0) + parent_top
        width = obj.get('width', 0)
        height = obj.get('height', 0)
        
        # Handle scaling
        scale_x = obj.get('scaleX', 1)
        scale_y = obj.get('scaleY', 1)
        effective_width = width * scale_x
        effective_height = height * scale_y
        
        # Handle text elements - estimate height based on fontSize and lineHeight
        if obj.get('type') == 'textbox':
            font_size = obj.get('fontSize', 16)
            line_height = obj.get('lineHeight', 1.2)
            text_lines = len(obj.get('text', '').split('\n'))
            estimated_height = font_size * line_height * text_lines
            effective_height = max(effective_height, estimated_height)
        
        return {
            'left': left,
            'top': top,
            'right': left + effective_width,
            'bottom': top + effective_height,
            'width': effective_width,
            'height': effective_height
        }
    
    def boxes_overlap(box1: Dict, box2: Dict, min_spacing: int = 20) -> bool:
        """Check if two bounding boxes overlap (with minimum spacing buffer)"""
        return not (
            box1['right'] + min_spacing <= box2['left'] or  # box1 is to the left of box2
            box1['left'] >= box2['right'] + min_spacing or  # box1 is to the right of box2
            box1['bottom'] + min_spacing <= box2['top'] or  # box1 is above box2
            box1['top'] >= box2['bottom'] + min_spacing     # box1 is below box2
        )
    
    # Collect all elements with their bounds
    all_elements = []
    
    def collect_elements(objects: List[Dict], parent_left: int = 0, parent_top: int = 0, parent_name: str = ""):
        for i, obj in enumerate(objects):
            obj_name = f"{parent_name}object[{i}]" if parent_name else f"object[{i}]"
            bounds = get_element_bounds(obj, parent_left, parent_top)
            
            all_elements.append({
                'name': obj_name,
                'type': obj.get('type'),
                'bounds': bounds,
                'obj': obj
            })
            
            # Handle grouped objects
            if obj.get('type') == 'group' and 'objects' in obj:
                group_left = obj.get('left', 0) + parent_left
                group_top = obj.get('top', 0) + parent_top
                collect_elements(obj['objects'], group_left, group_top, f"{obj_name}.group.")
    
    collect_elements(data.get('objects', []))
    
    # Check for overlaps between text elements and other elements
    for i, elem1 in enumerate(all_elements):
        for j, elem2 in enumerate(all_elements):
            if i >= j:  # Avoid checking the same pair twice
                continue
            
            # Check if elements overlap
            if boxes_overlap(elem1['bounds'], elem2['bounds']):
                # Special cases for acceptable overlaps
                if (elem1['type'] == 'image' and elem2['type'] == 'image'):
                    continue  # Images can overlap for layering effects
                if (elem1['type'] == 'rect' and elem2['type'] in ['image', 'textbox']):
                    continue  # Background rects can overlap content
                if (elem1['type'] == 'image' and elem2['type'] == 'rect'):
                    continue  # Images can overlap background rects
                if (elem1['type'] == 'image' and elem2['type'] == 'textbox'):
                    continue  # Background images can have text over them
                if (elem1['type'] == 'textbox' and elem2['type'] == 'image'):
                    continue  # Text can be over background images
                
                errors.append(
                    f"Element overlap detected: {elem1['name']} ({elem1['type']}) "
                    f"overlaps with {elem2['name']} ({elem2['type']}) - "
                    f"{elem1['name']} bounds: {elem1['bounds']}, "
                    f"{elem2['name']} bounds: {elem2['bounds']}"
                )
    
    # Check for text elements that are too close vertically (specific rule for text)
    text_elements = [elem for elem in all_elements if elem['type'] == 'textbox']
    text_elements.sort(key=lambda x: x['bounds']['top'])  # Sort by vertical position
    
    for i in range(len(text_elements) - 1):
        current = text_elements[i]
        next_elem = text_elements[i + 1]
        
        vertical_gap = next_elem['bounds']['top'] - current['bounds']['bottom']
        if vertical_gap < 30:  # Minimum 30px gap between text elements
            errors.append(
                f"Text elements too close: {current['name']} and {next_elem['name']} "
                f"have only {vertical_gap}px vertical gap (minimum 30px required)"
            )
    
    return len(errors) == 0, errors

def programmatic_validation(fabric_json: str, resolution: List[int]) -> Tuple[bool, List[str]]:
    """
    Comprehensive programmatic validation of Fabric.js JSON
    Returns: (is_valid, list_of_errors)
    """
    print("🔍 Running programmatic validation...")
    
    # Step 1: JSON structure validation
    is_valid_json, json_errors = validate_json_structure(fabric_json)
    if not is_valid_json:
        return False, json_errors
    
    # Parse JSON for further validation
    try:
        data = json.loads(fabric_json)
    except Exception as e:
        return False, [f"JSON parsing failed: {str(e)}"]
    
    all_errors = []
    
    # Step 2: Canvas dimensions validation
    is_valid_canvas, canvas_errors = validate_canvas_dimensions(data, resolution)
    all_errors.extend(canvas_errors)
    
    # Step 3: Element boundary validation  
    is_valid_bounds, bounds_errors = validate_element_boundaries(data, resolution)
    all_errors.extend(bounds_errors)
    
    # Step 4: Gradient syntax validation
    is_valid_gradients, gradient_errors = validate_gradient_syntax(data)
    all_errors.extend(gradient_errors)
    
    # Step 5: Text object validation
    is_valid_text, text_errors = validate_text_objects(data)
    all_errors.extend(text_errors)
    
    # Step 6: Color format validation
    is_valid_colors, color_errors = validate_color_format(data)
    all_errors.extend(color_errors)
    
    # Step 7: Text overlaps validation
    is_valid_overlaps, overlaps_errors = validate_text_overlaps(data)
    all_errors.extend(overlaps_errors)
    
    is_valid = len(all_errors) == 0
    
    if is_valid:
        print("✅ Programmatic validation passed!")
    else:
        print(f"❌ Programmatic validation failed with {len(all_errors)} errors")
        for error in all_errors:
            print(f"   - {error}")
    
    return is_valid, all_errors

def fix_programmatic_errors(fabric_json: str, errors: List[str], resolution: List[int]) -> str:
    """
    Automatically fix common programmatic errors
    """
    print("🔧 Auto-fixing programmatic errors...")
    
    try:
        data = json.loads(fabric_json)
    except:
        return fabric_json  # Can't fix if JSON is invalid
    
    canvas_width, canvas_height = resolution
    
    # Fix missing canvas dimensions
    if 'width' not in data or data['width'] != resolution[0]:
        data['width'] = resolution[0]
        print(f"   ✓ Fixed canvas width to {resolution[0]}")
    
    if 'height' not in data or data['height'] != resolution[1]:
        data['height'] = resolution[1]
        print(f"   ✓ Fixed canvas height to {resolution[1]}")
    
    # Fix version if missing
    if 'version' not in data:
        data['version'] = "5.3.0"
        print("   ✓ Added version 5.3.0")
    
    # Fix gradient colorStops format
    def fix_gradients(obj):
        if isinstance(obj, dict):
            if obj.get('type') in ['linear', 'radial'] and 'colorStops' in obj:
                color_stops = obj['colorStops']
                if isinstance(color_stops, dict):
                    # Convert object format to array format
                    new_stops = [
                        {"offset": float(k), "color": v}
                        for k, v in color_stops.items()
                    ]
                    obj['colorStops'] = new_stops
                    print("   ✓ Fixed gradient colorStops to array format")
            
            # Recursively fix nested objects
            for value in obj.values():
                if isinstance(value, (dict, list)):
                    fix_gradients(value)
        
        elif isinstance(obj, list):
            for item in obj:
                fix_gradients(item)
    
    fix_gradients(data)
    
    # Fix text objects (convert 'text' to 'textbox')
    def fix_text_types(obj):
        if isinstance(obj, dict):
            if obj.get('type') == 'text':
                obj['type'] = 'textbox'
                print("   ✓ Fixed text type to textbox")
            
            # Check grouped objects
            if obj.get('type') == 'group' and 'objects' in obj:
                for sub_obj in obj['objects']:
                    fix_text_types(sub_obj)
        
        elif isinstance(obj, list):
            for item in obj:
                fix_text_types(item)
    
    for obj in data.get('objects', []):
        fix_text_types(obj)
    
    # Fix element boundaries
    def fix_element_bounds(obj: Dict, obj_index: int, parent_name: str = ""):
        obj_name = f"{parent_name}object[{obj_index}]" if parent_name else f"object[{obj_index}]"
        
        left = obj.get('left', 0)
        top = obj.get('top', 0)
        width = obj.get('width', 0)
        height = obj.get('height', 0)
        
        # Handle scaling
        scale_x = obj.get('scaleX', 1)
        scale_y = obj.get('scaleY', 1)
        effective_width = width * scale_x
        effective_height = height * scale_y
        
        # Fix right boundary overflow
        if left + effective_width > canvas_width:
            if effective_width <= canvas_width:
                # Can fit by adjusting position
                new_left = canvas_width - effective_width
                obj['left'] = max(0, new_left)  # Don't go negative
                print(f"   ✓ Fixed {obj_name} left position from {left} to {obj['left']}")
            else:
                # Need to resize element
                if scale_x > 1:
                    # First try reducing scale
                    new_scale = canvas_width / width
                    obj['scaleX'] = min(new_scale, 1)
                    print(f"   ✓ Fixed {obj_name} scaleX from {scale_x} to {obj['scaleX']}")
                else:
                    # Resize the width directly
                    new_width = canvas_width - left
                    obj['width'] = max(new_width, 50)  # Minimum width of 50
                    if obj['width'] < new_width:  # If we had to enforce minimum
                        obj['left'] = canvas_width - obj['width']
                    print(f"   ✓ Fixed {obj_name} width from {width} to {obj['width']}")
        
        # Fix bottom boundary overflow
        if top + effective_height > canvas_height:
            if effective_height <= canvas_height:
                # Can fit by adjusting position
                new_top = canvas_height - effective_height
                obj['top'] = max(0, new_top)  # Don't go negative
                print(f"   ✓ Fixed {obj_name} top position from {top} to {obj['top']}")
            else:
                # Need to resize element
                if scale_y > 1:
                    # First try reducing scale
                    new_scale = canvas_height / height
                    obj['scaleY'] = min(new_scale, 1)
                    print(f"   ✓ Fixed {obj_name} scaleY from {scale_y} to {obj['scaleY']}")
                else:
                    # Resize the height directly
                    new_height = canvas_height - top
                    obj['height'] = max(new_height, 20)  # Minimum height of 20
                    if obj['height'] < new_height:  # If we had to enforce minimum
                        obj['top'] = canvas_height - obj['height']
                    print(f"   ✓ Fixed {obj_name} height from {height} to {obj['height']}")
        
        # Fix negative positions
        if left < 0:
            obj['left'] = 0
            print(f"   ✓ Fixed {obj_name} negative left position to 0")
        if top < 0:
            obj['top'] = 0
            print(f"   ✓ Fixed {obj_name} negative top position to 0")
        
        # Fix grouped objects
        if obj.get('type') == 'group' and 'objects' in obj:
            for i, sub_obj in enumerate(obj['objects']):
                fix_element_bounds(sub_obj, i, f"{obj_name}.group.")
    
    # Apply boundary fixes to all objects
    for i, obj in enumerate(data.get('objects', [])):
        fix_element_bounds(obj, i)
    
    # Fix text overlaps if overlap errors detected
    if any("overlap" in error.lower() for error in errors):
        print("🔧 Fixing text overlaps...")
        data = fix_text_overlaps(data, resolution)
    
    return json.dumps(data)

def fix_text_overlaps(data: Dict, resolution: List[int]) -> Dict:
    """Automatically fix text overlaps by repositioning elements"""
    
    def get_element_bounds(obj: Dict, parent_left: int = 0, parent_top: int = 0) -> Dict:
        """Calculate the actual bounding box of an element"""
        left = obj.get('left', 0) + parent_left
        top = obj.get('top', 0) + parent_top
        width = obj.get('width', 0)
        height = obj.get('height', 0)
        
        # Handle scaling
        scale_x = obj.get('scaleX', 1)
        scale_y = obj.get('scaleY', 1)
        effective_width = width * scale_x
        effective_height = height * scale_y
        
        # Handle text elements - estimate height based on fontSize and lineHeight
        if obj.get('type') == 'textbox':
            font_size = obj.get('fontSize', 16)
            line_height = obj.get('lineHeight', 1.2)
            text_lines = len(obj.get('text', '').split('\n'))
            estimated_height = font_size * line_height * text_lines
            effective_height = max(effective_height, estimated_height)
        
        return {
            'left': left,
            'top': top,
            'right': left + effective_width,
            'bottom': top + effective_height,
            'width': effective_width,
            'height': effective_height
        }
    
    # Collect all text elements
    text_elements = []
    
    def collect_text_elements(objects: List[Dict], parent_left: int = 0, parent_top: int = 0):
        for obj in objects:
            if obj.get('type') == 'textbox':
                bounds = get_element_bounds(obj, parent_left, parent_top)
                text_elements.append({
                    'obj': obj,
                    'bounds': bounds,
                    'parent_left': parent_left,
                    'parent_top': parent_top
                })
            elif obj.get('type') == 'group' and 'objects' in obj:
                group_left = obj.get('left', 0) + parent_left
                group_top = obj.get('top', 0) + parent_top
                collect_text_elements(obj['objects'], group_left, group_top)
    
    collect_text_elements(data.get('objects', []))
    
    # Sort text elements by vertical position
    text_elements.sort(key=lambda x: x['bounds']['top'])
    
    # Reposition overlapping text elements
    canvas_height = resolution[1]
    margin = 40  # Minimum margin from edges
    min_spacing = 40  # Minimum spacing between text elements
    
    for i in range(len(text_elements)):
        current = text_elements[i]
        current_bounds = current['bounds']
        
        # Check if current element overlaps with any previous element
        for j in range(i):
            prev = text_elements[j]
            prev_bounds = prev['bounds']
            
            # Check for vertical overlap
            if (current_bounds['top'] < prev_bounds['bottom'] + min_spacing):
                # Move current element below the previous one
                new_top = prev_bounds['bottom'] + min_spacing - current['parent_top']
                current['obj']['top'] = max(margin, new_top)
                
                # Update bounds for future checks
                current['bounds']['top'] = current['obj']['top'] + current['parent_top']
                current['bounds']['bottom'] = current['bounds']['top'] + current['bounds']['height']
                
                # Ensure element doesn't go off canvas
                if current['bounds']['bottom'] > canvas_height - margin:
                    # If it would go off canvas, try to reduce font size instead
                    if current['obj'].get('fontSize', 16) > 24:
                        current['obj']['fontSize'] = int(current['obj']['fontSize'] * 0.8)
                        # Recalculate bounds with new font size
                        font_size = current['obj']['fontSize']
                        line_height = current['obj'].get('lineHeight', 1.2)
                        text_lines = len(current['obj'].get('text', '').split('\n'))
                        new_height = font_size * line_height * text_lines
                        current['bounds']['height'] = new_height
                        current['bounds']['bottom'] = current['bounds']['top'] + new_height
                break
    
    return data

# --- ENHANCED VALIDATION FUNCTION ---
def validate_banner(fabric_json: str, banner_prompt: str, resolution: list) -> str:
    """
    Enhanced hybrid validation: Programmatic + LLM validation
    Returns: "PASS" or "CONTINUE: [feedback]"
    """
    try:
        # Step 1: Programmatic validation (syntax, boundaries, structure)
        is_programmatically_valid, prog_errors = programmatic_validation(fabric_json, resolution)
        
        if not is_programmatically_valid:
            # Return programmatic errors for fixing
            error_summary = "; ".join(prog_errors[:3])  # Limit to first 3 errors
            return f"CONTINUE: PROGRAMMATIC_ERRORS: {error_summary}"
        
        # Step 2: LLM validation for design quality (only if programmatically valid)
        print("🎨 Running design quality validation...")
        validator_llm = create_validator_llm()
        
        validation_prompt = f"""
BANNER REQUIREMENTS:
{banner_prompt}

CANVAS: {resolution[0]}x{resolution[1]} pixels

FABRIC.JS JSON (already validated for syntax):
{fabric_json}

This JSON has passed all programmatic validation. Evaluate DESIGN QUALITY only:
- Text readability and contrast against backgrounds
- Visual hierarchy and professional appearance
- Asset placement and utilization
- CTA effectiveness and visibility
- Overall design coherence

Respond with "PASS" if design quality is good, or "CONTINUE: [specific design improvements]" if issues found.
"""
        
        messages = [
            {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
            {"role": "user", "content": validation_prompt}
        ]
        
        response = validator_llm.invoke(messages)
        result = response.content.strip()
        
        print(f"🎨 Design validation result: {result[:100]}...")
        return result
        
    except Exception as e:
        print(f"Validation error: {e}")
        return "CONTINUE: Validation failed, please review banner composition"

# --- ENHANCED FEEDBACK APPLICATION ---
def apply_feedback(fabric_json: str, feedback: str, banner_prompt: str, assets: list, resolution: list) -> str:
    """
    Enhanced feedback application with programmatic fixes
    """
    try:
        # Handle programmatic errors first
        if feedback.startswith("PROGRAMMATIC_ERRORS:"):
            print("🔧 Applying programmatic fixes...")
            error_details = feedback.replace("PROGRAMMATIC_ERRORS:", "").strip()
            
            # Get detailed errors for fixing
            _, detailed_errors = programmatic_validation(fabric_json, resolution)
            
            # Apply automatic fixes
            fixed_json = fix_programmatic_errors(fabric_json, detailed_errors, resolution)
            
            # Validate the fix
            is_fixed, remaining_errors = programmatic_validation(fixed_json, resolution)
            
            if is_fixed:
                print("✅ Programmatic errors fixed successfully!")
                return fixed_json
            else:
                print(f"⚠️ Some programmatic errors remain: {len(remaining_errors)}")
                # If auto-fix doesn't work, fall back to LLM
                
        # Handle design feedback with LLM
        print("🎨 Applying design feedback...")
        feedback_llm = create_feedback_llm()
        
        feedback_prompt = f"""
ORIGINAL REQUIREMENTS:
{banner_prompt}

AVAILABLE ASSETS:
{json.dumps(assets, indent=2)}

CANVAS: {resolution[0]}x{resolution[1]} pixels

CURRENT FABRIC.JS JSON:
{fabric_json}

DESIGN FEEDBACK TO APPLY:
{feedback}

Apply the specific feedback to improve the design. The JSON is syntactically correct.
Focus on design improvements only:
- Enhance text contrast and readability
- Improve visual hierarchy and spacing  
- Better asset placement and sizing
- More effective CTA design

Keep all assets and maintain valid syntax. Return ONLY the improved JSON.
"""
        
        messages = [
            {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
            {"role": "user", "content": feedback_prompt}
        ]
        
        response = feedback_llm.invoke(messages)
        improved_json = response.content.strip()
        
        # Clean the response
        if improved_json.startswith("```"):
            improved_json = improved_json.split("\n", 1)[1]
        if improved_json.endswith("```"):
            improved_json = improved_json.rsplit("\n", 1)[0]
        improved_json = improved_json.strip()
        
        # Verify the improved JSON is still valid
        is_valid, errors = programmatic_validation(improved_json, resolution)
        if not is_valid:
            print(f"⚠️ Feedback application broke validation, returning original")
            return fabric_json
        
        print("✅ Design feedback applied successfully!")
        return improved_json
        
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
    Compose a Fabric.js v5.3.0 JSON banner with enhanced validation feedback loop.
    Args:
        banner_prompt (str): The design brief from banner_design_researcher tool.
        assets (list): List of asset dicts (type, url, description, etc).
        resolution (list): [width, height] of the banner.
    Returns:
        str: Fabric.js v5.3.0 JSON string.
    """
    print(f'🎨 Composing banner with enhanced validation loop...')
    print(f'📐 Resolution: {resolution[0]}x{resolution[1]}')
    print(f'🎯 Assets available: {len(assets)}')
    
    try:
        composer_llm = create_composer_llm()
        
        # Initial composition
        composition_prompt = f"""
DESIGN BRIEF TO IMPLEMENT:
{banner_prompt}

CANVAS: {resolution[0]}x{resolution[1]} pixels

AVAILABLE ASSETS:
{json.dumps(assets, indent=2)}

Create a professional Fabric.js banner that PRECISELY FOLLOWS the design brief specifications:

DESIGN BRIEF COMPLIANCE (CRITICAL):
- Follow EXACT text placement and positioning specified in the design brief
- Implement the Layout Architecture with mathematical precision as described
- Use Typography System sizing and hierarchy exactly as specified
- Apply Grid System and spacing relationships from the brief
- Position elements according to Visual Weight Distribution and Focal Point Strategy

TEXT OVERLAP PREVENTION (MANDATORY):
- Calculate actual text heights: fontSize × lineHeight × number of text lines
- Ensure minimum 40px vertical spacing between ALL text elements
- Stack text elements vertically without overlapping
- Check that no text overlaps with images, buttons, or other elements
- Position text elements in clear, readable sequence from top to bottom

TECHNICAL REQUIREMENTS:
- INCLUDES canvas dimensions in root: {{"version": "5.3.0", "width": {resolution[0]}, "height": {resolution[1]}, "objects": [...]}}
- Uses all provided assets effectively and as described in the brief
- Keeps ALL elements within {resolution[0]}x{resolution[1]} bounds
- Places CTA text inside CTA buttons per brief specifications
- Ensures readable text with contrast ratios specified in Color Strategy
- Uses proper gradient syntax with array format colorStops

**EXECUTION PRIORITY**: The design brief contains detailed layout specifications - follow them EXACTLY for text positioning, margins, spacing, and element relationships.

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
        
        # Enhanced validation and feedback loop (max 5 iterations)
        max_iterations = 5
        for iteration in range(max_iterations):
            print(f"🔍 Step {iteration + 2}: Enhanced validation (iteration {iteration + 1}/{max_iterations})...")
            
            validation_result = validate_banner(current_json, banner_prompt, resolution)
            
            if validation_result.startswith("PASS"):
                print(f"✅ Banner validated successfully after {iteration + 1} iteration(s)!")
                return current_json
            
            elif validation_result.startswith("CONTINUE:"):
                feedback = validation_result.replace("CONTINUE:", "").strip()
                print(f"⚠️  Feedback received: {feedback[:100]}...")
                
                if iteration < max_iterations - 1:  # Don't apply feedback on last iteration
                    print(f"🔧 Applying enhanced feedback...")
                    improved_json = apply_feedback(current_json, feedback, banner_prompt, assets, resolution)
                    current_json = improved_json
                else:
                    print(f"🔄 Max iterations reached, returning current version")
                    break
            else:
                print(f"⚠️  Unexpected validation response: {validation_result}")
                break
        
        print(f"🏁 Enhanced composition complete after {max_iterations} iterations")
        return current_json
        
    except Exception as e:
        print(f"❌ Composer error: {e}")
        return f"Error: {e}"

# --- Main function for direct execution ---
if __name__ == "__main__":
    # Test the enhanced composer with programmatic validation
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
    
    # Test 1: Enhanced banner generation
    print("🧪 TEST 1: Enhanced Banner Generation with Programmatic Validation")
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
    with open("test_enhanced_banner_output.json", "w") as f:
        f.write(fabric_json)
    print(f"\n💾 Enhanced output saved to test_enhanced_banner_output.json")
    
    # Test 2: Programmatic validation with problematic JSON
    print("\n" + "="*60)
    print("🧪 TEST 2: Programmatic Validation with Multiple Issues")
    print("="*60)
    
    problematic_json = '''
    {
      "version": "5.3.0",
      "objects": [
        {
          "type": "text",
          "text": "Test Text",
          "left": 900,
          "top": 100,
          "width": 300,
          "height": 50,
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
    
    print("Testing problematic JSON with issues:")
    print("- Missing canvas width/height")  
    print("- Element extending beyond bounds (left: 900, width: 300 = 1200 > 1080)")
    print("- Wrong gradient colorStops format (object instead of array)")
    print("- Wrong text type ('text' instead of 'textbox')")
    
    is_valid, errors = programmatic_validation(problematic_json, [1080, 1080])
    print(f"\nProgrammatic validation result: {'PASS' if is_valid else 'FAIL'}")
    
    if not is_valid:
        print(f"Errors found ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")
        
        print("\n🔧 Testing automatic fixes...")
        fixed_json = fix_programmatic_errors(problematic_json, errors, [1080, 1080])
        
        print("Re-validating fixed JSON...")
        is_fixed, remaining_errors = programmatic_validation(fixed_json, [1080, 1080])
        
        if is_fixed:
            print("✅ All programmatic errors successfully fixed!")
        else:
            print(f"⚠️ {len(remaining_errors)} errors remain:")
            for error in remaining_errors:
                print(f"  - {error}")
        
        print(f"\nFixed JSON preview:")
        print(fixed_json[:300] + "..." if len(fixed_json) > 300 else fixed_json)
    
    print("\n" + "="*60)
    print("🎉 ENHANCED VALIDATION SYSTEM TESTS COMPLETE!")
    print("="*60)
