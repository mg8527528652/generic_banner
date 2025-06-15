import os
import json
import time
from typing import Dict, Any, List, Optional, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import all the specialized tools
from utils.researcher_tool import banner_design_researcher
from utils.image_tools import background_replacer, text_to_image_generator
from utils.svg_genrator import svg_generator
from utils.transparent_illustration_tool import generate_image_tool
from utils.font_matching import select_best_font_url
from utils.composer_engine import compose_fabric_banner

# Load environment variables
load_dotenv()

# --- State Definition ---
class BannerState(TypedDict):
    user_prompt: str
    product_image_url: Optional[str]
    logo: Optional[str]
    resolution: List[int]
    design_brief: str
    generated_assets: List[Dict[str, Any]]
    font_url: str
    fabric_json: str
    messages: List[Any]
    current_step: str
    error: Optional[str]
    execution_plan: Dict[str, Any]

# --- System Prompts ---
MASTER_PLANNER_SYSTEM_PROMPT = """
You are the MASTER PLANNER for a PREMIUM AI banner generation pipeline. Your goal is to create SELLABLE, PROFESSIONAL-GRADE banners that rival high-end design agencies.

**QUALITY STANDARDS:**
- Think Dribbble/Behance premium level quality
- Modern, sophisticated, market-ready designs
- Rich visual depth and professional polish
- Strong visual hierarchy and composition

**CRITICAL RULES:**
1. For visual assets: NEVER include text, words, or letters in prompts
2. Text will be handled separately by Fabric.js composition
3. Focus on creating RICH, LAYERED visual experiences

**PROFESSIONAL DESIGN ANALYSIS:**

1. **DESIGN BRIEF PARSING** - Extract and elevate:
   - Visual mood: luxury, modern, vintage, bold, minimalist, artistic, corporate
   - Color psychology: warm/cool, high/low contrast, monochromatic/complementary
   - Target audience: premium/budget, young/mature, creative/corporate
   - Visual complexity: clean/detailed, geometric/organic, flat/dimensional

2. **PREMIUM ASSET STRATEGY:**
   - `text_to_image_generator`: For RICH, ATMOSPHERIC backgrounds with professional lighting, depth, and composition
   - `generate_image_tool`: For SOPHISTICATED illustrations with artistic flair and transparent backgrounds
   - `svg_generator`: For ELEGANT decorative elements with mathematical precision
   - `select_best_font_url`: For TYPOGRAPHY that matches the premium aesthetic
   - Multiple asset layers for visual richness and depth

3. **EXECUTION PLAN FORMAT:**
```json
{
  "assets_to_generate": [
    {
      "type": "background|illustration|decoration|font",
      "tool": "tool_name", 
      "prompt": "RICH, DETAILED visual description with professional terminology",
      "description": "specific role in creating premium visual impact",
      "dimensions": {"width": X, "height": Y},
      "style_keywords": ["keyword1", "keyword2", "keyword3"]
    }
  ],
  "design_theme": "overall aesthetic direction",
  "color_palette": ["primary", "secondary", "accent colors"],
  "reasoning": "why this creates a sellable, premium banner"
}
```

**PREMIUM PROMPT GUIDELINES:**

**BACKGROUND PROMPTS - Use professional photography terms:**
- "Cinematic lighting with dramatic shadows and highlights"
- "Professional studio setup with gradient backdrop"  
- "Atmospheric depth with bokeh effects and layered composition"
- "High-end commercial photography aesthetic"
- "Sophisticated color grading and tonal balance"

**ILLUSTRATION PROMPTS - Use artistic terminology:**
- "Vector illustration with sophisticated gradients and geometric precision"
- "Artistic composition with balanced asymmetry and visual flow"
- "Contemporary graphic design with premium finish and subtle details"
- "Modern illustration style with clean lines and rich color palette"

**DECORATION PROMPTS - Use design principles:**
- "Elegant geometric patterns following golden ratio proportions"
- "Sophisticated decorative elements with mathematical precision"
- "Premium accent graphics with subtle gradients and perfect spacing"
- "Modern ornamental design with balanced visual weight"

**ALWAYS GENERATE 3-5 ASSETS** for visual richness:
1. Primary background (atmospheric, rich)
2. Secondary graphic elements (illustrations/icons)
3. Decorative accents (patterns, shapes, textures)
4. Typography selection
5. Optional: Additional overlay/texture elements

Return ONLY the JSON execution plan with no markdown code blocks, explanations, or additional text. Pure JSON only.
"""

# --- Agent Functions ---
def create_master_planner():
    """Create the master planner LLM"""
    return ChatOpenAI(
        model="o3-2025-04-16",  # Using GPT-4O as requested
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        reasoning_effort="high"
    )

def is_tool_success(result) -> bool:
    """
    Check if a tool result indicates success.
    
    Handles different tool response formats:
    - String responses (most tools): Success if not starting with "Error"
    - Dict responses (generate_image_tool): Success if no "error" key or result is a string
    """
    if isinstance(result, str):
        return result and not result.startswith("Error")
    elif isinstance(result, dict):
        return not result.get("error")
    else:
        return bool(result)

def extract_tool_result(result):
    """
    Extract the actual result/URL from tool response.
    
    Handles different response formats:
    - String: Return as-is
    - Dict: Return the main result (usually 'link' or 'url')
    """
    if isinstance(result, str):
        return result
    elif isinstance(result, dict):
        # For generate_image_tool error responses
        if result.get("error"):
            return result.get("error", "Unknown error")
        # For successful responses, we should have gotten a string
        return result.get("link", result.get("url", str(result)))
    else:
        return str(result)

def research_phase(state: BannerState) -> BannerState:
    """Phase 1: Research and get detailed design brief"""
    try:
        print("🔍 Phase 1: Researching banner design...")
        
        # Call the banner design researcher
        design_brief = banner_design_researcher.invoke({
            "user_request": state["user_prompt"],
            "resolution": state["resolution"],
            "product_url_provided": bool(state.get("product_image_url")),
            "logo_url_provided": bool(state.get("logo"))
        })
        
        state["design_brief"] = design_brief
        state["current_step"] = "planning"
        state["messages"].append(AIMessage(content=f"Design brief generated: {design_brief[:200]}..."))
        
        print(f"✅ Design brief generated ({len(design_brief)} characters)")
        return state
        
    except Exception as e:
        state["error"] = f"Research phase failed: {str(e)}"
        state["current_step"] = "error"
        return state

def planning_phase(state: BannerState) -> BannerState:
    """Phase 2: Plan asset generation strategy"""
    try:
        print("🎯 Phase 2: Planning asset generation strategy...")
        
        llm = create_master_planner()
        
        # Prepare context for the planner
        context = f"""
DESIGN BRIEF TO ANALYZE:
{state['design_brief']}

AVAILABLE RESOURCES:
- Product Image URL: {'Yes' if state.get('product_image_url') else 'No'}
- Logo: {'Yes' if state.get('logo') else 'No'}
- Target Resolution: {state['resolution'][0]}x{state['resolution'][1]}

Create a concrete execution plan with specific assets, tools, and prompts (without text for visual assets).
"""
        
        messages = [
            SystemMessage(content=MASTER_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=context)
        ]
        
        response = llm.invoke(messages)
        
        # Parse the JSON response, handling markdown code blocks
        try:
            import json
            import re
            
            # Clean the response by removing markdown code blocks
            content = response.content.strip()
            
            # Remove markdown code block markers
            content = re.sub(r'^```(?:json)?\s*\n?', '', content, flags=re.MULTILINE)
            content = re.sub(r'\n?```\s*$', '', content, flags=re.MULTILINE)
            content = content.strip()
            
            execution_plan = json.loads(content)
            
            # Store the execution plan in state
            state["execution_plan"] = execution_plan
            state["messages"].append(AIMessage(content=f"Execution plan created with {len(execution_plan.get('assets_to_generate', []))} assets"))
            state["current_step"] = "generation"
            
            print(f"✅ Execution plan created:")
            print(f"   - Assets to generate: {len(execution_plan.get('assets_to_generate', []))}")
            for i, asset in enumerate(execution_plan.get('assets_to_generate', []), 1):
                print(f"   {i}. {asset.get('type', 'unknown')} via {asset.get('tool', 'unknown')}")
            
            return state
            
        except json.JSONDecodeError as e:
            state["error"] = f"Failed to parse execution plan JSON: {str(e)}\nResponse: {response.content[:500]}"
            state["current_step"] = "error"
            return state
        
    except Exception as e:
        state["error"] = f"Planning phase failed: {str(e)}"
        state["current_step"] = "error"
        return state

def generate_single_asset(asset_plan: Dict[str, Any], state: BannerState, asset_index: int) -> Optional[Dict[str, Any]]:
    """Generate a single asset - used for parallel processing"""
    asset_type = asset_plan.get("type", "unknown")
    tool_name = asset_plan.get("tool", "")
    prompt = asset_plan.get("prompt", "")
    description = asset_plan.get("description", "")
    dimensions = asset_plan.get("dimensions", {"width": state["resolution"][0], "height": state["resolution"][1]})
    
    print(f"  {asset_index}. Generating {asset_type} using {tool_name}...")
    print(f"     Prompt: {prompt}")
    
    try:
        # Route to the appropriate tool based on the plan
        if tool_name == "select_best_font_url":
            result = select_best_font_url.invoke({
                "banner_prompt": state["design_brief"]
            })
            
            if is_tool_success(result):
                font_url = extract_tool_result(result)
                print(f"     ✅ Font selected: {font_url}")
                return {
                    "type": "font",
                    "url": font_url,
                    "description": description,
                    "is_font": True  # Special flag for font assets
                }
            else:
                error_msg = extract_tool_result(result)
                print(f"     ❌ Font selection failed: {error_msg}")
                return None
        
        elif tool_name == "text_to_image_generator":
            result = text_to_image_generator.invoke({
                "prompt": prompt,
                "width": dimensions["width"],
                "height": dimensions["height"]
            })
            
            if is_tool_success(result):
                image_url = extract_tool_result(result)
                print(f"     ✅ Background image generated")
                return {
                    "type": asset_type,
                    "url": image_url,
                    "description": description
                }
            else:
                error_msg = extract_tool_result(result)
                print(f"     ❌ Background generation failed: {error_msg}")
                return None
        
        elif tool_name == "svg_generator":
            result = svg_generator.invoke({
                "description": prompt,
                "width": dimensions.get("width", 200),
                "height": dimensions.get("height", 200),
                "style": "modern"
            })
            
            if is_tool_success(result):
                svg_content = extract_tool_result(result)
                print(f"     ✅ SVG asset generated")
                return {
                    "type": asset_type,
                    "content": svg_content,
                    "description": description
                }
            else:
                error_msg = extract_tool_result(result)
                print(f"     ❌ SVG generation failed: {error_msg}")
                return None
        
        elif tool_name == "generate_image_tool":
            result = generate_image_tool.invoke({
                "prompt": prompt,
                "size": f"{dimensions['width']}x{dimensions['height']}"
            })
            
            if is_tool_success(result):
                image_url = extract_tool_result(result)
                print(f"     ✅ Illustration generated")
                return {
                    "type": asset_type,
                    "url": image_url,
                    "description": description
                }
            else:
                error_msg = extract_tool_result(result)
                print(f"     ❌ Illustration generation failed: {error_msg}")
                return None
        
        elif tool_name == "background_replacer":
            if state.get("product_image_url"):
                result = background_replacer.invoke({
                    "image_url": state["product_image_url"],
                    "prompt": prompt
                })
                
                if is_tool_success(result):
                    image_url = extract_tool_result(result)
                    print(f"     ✅ Background replaced successfully")
                    return {
                        "type": asset_type,
                        "url": image_url,
                        "description": description
                    }
                else:
                    error_msg = extract_tool_result(result)
                    print(f"     ❌ Background replacement failed: {error_msg}")
                    return None
            else:
                print(f"     ⚠️ Background replacement requires product image URL")
                return None
        
        else:
            print(f"     ❌ Unknown tool: {tool_name}")
            return None
            
    except Exception as e:
        print(f"     ❌ Error generating {asset_type}: {str(e)}")
        return None

def generation_phase(state: BannerState) -> BannerState:
    """Phase 3: Generate assets according to execution plan - with parallel processing"""
    try:
        print("🎨 Phase 3: Generating banner assets in parallel...")
        
        execution_plan = state.get("execution_plan", {})
        assets_to_generate = execution_plan.get("assets_to_generate", [])
        
        if not assets_to_generate:
            print("⚠️  No assets specified in execution plan")
            state["generated_assets"] = []
            state["current_step"] = "composition"
            return state
        
        print(f"📋 Following execution plan with {len(assets_to_generate)} assets")
        
        generated_assets = []
        
        # Process assets in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(assets_to_generate), 4)) as executor:
            # Submit all asset generation tasks
            future_to_asset = {
                executor.submit(generate_single_asset, asset_plan, state, i+1): (asset_plan, i+1)
                for i, asset_plan in enumerate(assets_to_generate)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_asset):
                asset_plan, asset_index = future_to_asset[future]
                try:
                    result = future.result()
                    if result:
                        # Handle font assets specially
                        if result.get("is_font"):
                            state["font_url"] = result["url"]
                            result.pop("is_font", None)  # Remove the flag before adding to assets
                        
                        generated_assets.append(result)
                except Exception as e:
                    asset_type = asset_plan.get("type", "unknown")
                    print(f"     ❌ Parallel execution error for {asset_type}: {str(e)}")
        
        state["generated_assets"] = generated_assets
        state["current_step"] = "composition"
        
        print(f"✅ Parallel asset generation complete - {len(generated_assets)} assets successfully created")
        return state
        
    except Exception as e:
        state["error"] = f"Generation phase failed: {str(e)}"
        state["current_step"] = "error"
        return state

def composition_phase(state: BannerState) -> BannerState:
    """Phase 4: Compose final Fabric.js banner"""
    try:
        print("🎼 Phase 4: Composing final banner...")
        
        # Compose the final banner
        fabric_json = compose_fabric_banner.invoke({
            "banner_prompt": state["design_brief"],
            "assets": state["generated_assets"],
            "resolution": state["resolution"]
        })
        
        if fabric_json and not fabric_json.startswith("Error"):
            state["fabric_json"] = fabric_json
            state["current_step"] = "complete"
            print("✅ Banner composition complete!")
        else:
            state["error"] = f"Composition failed: {fabric_json}"
            state["current_step"] = "error"
        
        return state
        
    except Exception as e:
        state["error"] = f"Composition phase failed: {str(e)}"
        state["current_step"] = "error"
        return state

def should_continue_research(state: BannerState) -> str:
    """Determine next step after research phase"""
    if state.get("error"):
        print(f"🔄 Research phase: error detected -> finish")
        return "finish"
    return "planning"

def should_continue_planning(state: BannerState) -> str:
    """Determine next step after planning phase"""
    if state.get("error"):
        print(f"🔄 Planning phase: error detected -> finish")
        return "finish"
    return "generation"

def should_continue_generation(state: BannerState) -> str:
    """Determine next step after generation phase"""
    if state.get("error"):
        print(f"🔄 Generation phase: error detected -> finish")
        return "finish"
    return "composition"

def should_continue_composition(state: BannerState) -> str:
    """Determine next step after composition phase"""
    if state.get("error"):
        print(f"🔄 Composition phase: error detected -> finish")
        return "finish"
    return "finish"

def finish_workflow(state: BannerState) -> BannerState:
    """Final step - just return the state"""
    if state.get("error"):
        print(f"❌ Workflow completed with error: {state['error']}")
    else:
        print("✅ Workflow completed successfully!")
    return state

# Error handling is now done by returning END in the conditional functions

# --- Main Workflow ---
def create_banner_workflow():
    """Create the LangGraph workflow"""
    
    workflow = StateGraph(BannerState)
    
    # Add nodes
    workflow.add_node("research", research_phase)
    workflow.add_node("planning", planning_phase)
    workflow.add_node("generation", generation_phase)
    workflow.add_node("composition", composition_phase)
    workflow.add_node("finish", finish_workflow)
    
    # Set entry point
    workflow.set_entry_point("research")
    
    # Add edges
    workflow.add_conditional_edges(
        "research",
        should_continue_research,
        {
            "planning": "planning",
            "finish": "finish"
        }
    )
    
    workflow.add_conditional_edges(
        "planning",
        should_continue_planning,
        {
            "generation": "generation",
            "finish": "finish"
        }
    )
    
    workflow.add_conditional_edges(
        "generation",
        should_continue_generation,
        {
            "composition": "composition",
            "finish": "finish"
        }
    )
    
    workflow.add_conditional_edges(
        "composition",
        should_continue_composition,
        {
            "finish": "finish"
        }
    )
    
    workflow.add_edge("finish", END)
    
    return workflow.compile()

# --- Main API Function ---
def generate_banner(
    user_prompt: str,
    resolution: List[int],
    product_image_url: Optional[str] = None,
    logo: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main function to generate a banner using the AI Director workflow.
    
    Args:
        user_prompt: Text description of the desired banner
        resolution: [width, height] of the banner
        product_image_url: Optional URL of product image
        logo: Optional logo SVG or URL
    
    Returns:
        Dictionary containing the generated Fabric.js JSON and metadata
    """
    
    print("🚀 Starting AI Director Banner Generation Pipeline...")
    print(f"📝 Prompt: {user_prompt}")
    print(f"📐 Resolution: {resolution[0]}x{resolution[1]}")
    print(f"🖼️ Product Image: {'Yes' if product_image_url else 'No'}")
    print(f"🏷️ Logo: {'Yes' if logo else 'No'}")
    print("-" * 60)
    
    # Initialize state
    initial_state = BannerState(
        user_prompt=user_prompt,
        product_image_url=product_image_url,
        logo=logo,
        resolution=resolution,
        design_brief="",
        generated_assets=[],
        font_url="",
        fabric_json="",
        messages=[],
        current_step="research",
        error=None,
        execution_plan={}
    )
    
    # Create and run workflow
    workflow = create_banner_workflow()
    
    try:
        # Execute the workflow
        print("🔧 Invoking workflow...")
        final_state = workflow.invoke(initial_state)
        print(f"🔧 Workflow completed. Final state keys: {list(final_state.keys())}")
        print(f"🔧 Final current_step: {final_state.get('current_step')}")
        
        if final_state.get("error"):
            print(f"🔧 Error in final state: {final_state['error']}")
            return f"Error: {final_state['error']}"
        
        # Return just the Fabric.js JSON string for direct use
        return final_state["fabric_json"]
        
    except Exception as e:
        print(f"🔧 Exception details: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

# --- Example Usage ---
if __name__ == "__main__":
    """
    EXAMPLE EXECUTION PLANS:
    
    1. FITNESS BANNER WITH PERSON:
    {
      "assets_to_generate": [
        {
          "type": "background",
          "tool": "text_to_image_generator", 
          "prompt": "Dynamic fitness gym environment with modern equipment, dramatic lighting, energetic atmosphere",
          "description": "Main background showing gym setting",
          "dimensions": {"width": 1080, "height": 1080}
        },
        {
          "type": "decoration",
          "tool": "svg_generator",
          "prompt": "Abstract geometric swoosh patterns and energy burst effects",
          "description": "Decorative overlay elements",
          "dimensions": {"width": 200, "height": 200}
        },
        {
          "type": "font",
          "tool": "select_best_font_url",
          "prompt": "",
          "description": "Typography for motivational fitness theme"
        }
      ],
      "reasoning": "Need realistic gym background, simple decorative elements, and bold typography. No text in visual assets."
    }
    
    2. BUSINESS BANNER:
    {
      "assets_to_generate": [
        {
          "type": "background",
          "tool": "text_to_image_generator",
          "prompt": "Modern office environment with natural lighting, clean professional aesthetic",
          "description": "Corporate background setting",
          "dimensions": {"width": 1080, "height": 1080}
        },
        {
          "type": "font",
          "tool": "select_best_font_url", 
          "prompt": "",
          "description": "Professional typography"
        }
      ],
      "reasoning": "Simple professional background, elegant typography. Fabric.js will handle geometric elements."
    }
    """
    simple_banner_queries = [
    "Create a 'Grand Opening' banner for a new coffee shop called 'The Daily Grind.'"
    "Design a motivational banner for Instagram with the quote 'Dream big, work hard, stay focused.'",
    "Generate a '40% Off Flash Sale' banner for a clothing store, valid this weekend only.",
    "Make a 'We're Hiring!' banner for a software developer position at a tech startup.",
    "Create a professional 'Happy Diwali' greeting banner for a corporate client.",
    "Design an informational banner with the title '5 Tips for Better Time Management.'",
    "Make an 'Open House this Saturday, 1 PM - 4 PM' banner for a modern suburban home.",
    "Create a YouTube thumbnail for a new podcast episode titled 'The Secrets of Ancient Rome.'",
    "Design a simple 'Save the Date' banner for Jessica and Tom's wedding on October 18th, 2025.",
    "Generate a vibrant 'Taco Tuesday' promotional banner for a Mexican restaurant."
    ]
    def generate_single_banner(query_data):
        """Generate a single banner - used for parallel processing"""
        i, query = query_data
        st = time.time()
        
        print(f"🔧 [Thread-{threading.current_thread().name}] Generating banner {i+1}: {query[:50]}...")
        
        try:
            result = generate_banner(
                user_prompt=query,
                resolution=[1024, 1024],
                product_image_url='',
                logo=None
            )
            
            if result and not result.startswith("Error"):
                print(f"\n🎉 [Thread-{threading.current_thread().name}] Banner {i+1} generated successfully!")
                print(f"📄 Fabric JSON Length: {len(result)} characters")
                
                # Clean and parse result
                result = result.strip().replace("```json", "").replace("```", "")
                result_dict = json.loads(result)
                
                # Save the Fabric.js JSON directly
                filename = f"generated_banner_{str(i)}.json"
                save_path =  "generated_banners"    
                os.makedirs(save_path, exist_ok=True)
                with open(save_path + "/" + filename, "w") as f:
                    f.write(result)
                
                print(f"💾 Fabric.js JSON saved to {filename}")
                print(f"🔧 Time taken: {time.time() - st:.2f} seconds")
                
                return {
                    "index": i,
                    "query": query,
                    "success": True,
                    "filename": filename,
                    "time_taken": time.time() - st,
                    "json_length": len(result)
                }
            else:
                print(f"\n❌ [Thread-{threading.current_thread().name}] Banner {i+1} generation failed: {result}")
                return {
                    "index": i,
                    "query": query,
                    "success": False,
                    "error": result,
                    "time_taken": time.time() - st
                }
                
        except Exception as e:
            print(f"\n❌ [Thread-{threading.current_thread().name}] Banner {i+1} generation error: {str(e)}")
            return {
                "index": i,
                "query": query,
                "success": False,
                "error": str(e),
                "time_taken": time.time() - st
            }
    
    # Process banners in parallel
    total_start_time = time.time()
    print(f"🚀 Starting parallel processing of {len(simple_banner_queries)} banner queries...")
    print(f"🔧 Using ThreadPoolExecutor with max_workers={min(len(simple_banner_queries), 3)}")
    
    results = []
    with ThreadPoolExecutor(max_workers=min(len(simple_banner_queries), 9)) as executor:
        # Submit all banner generation tasks
        query_data = [(i, query) for i, query in enumerate(simple_banner_queries)]
        future_to_query = {
            executor.submit(generate_single_banner, data): data
            for data in query_data
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_query):
            query_data = future_to_query[future]
            try:
                result = future.result()
                results.append(result)
                print(f"✅ Completed banner {result['index']+1}/{len(simple_banner_queries)}")
            except Exception as e:
                print(f"❌ Error processing banner {query_data[0]+1}: {str(e)}")
                results.append({
                    "index": query_data[0],
                    "query": query_data[1],
                    "success": False,
                    "error": str(e),
                    "time_taken": 0
                })
    
    # Sort results by index to maintain order
    results.sort(key=lambda x: x['index'])
    
    # Print summary
    total_time = time.time() - total_start_time
    successful_banners = sum(1 for r in results if r['success'])
    failed_banners = len(results) - successful_banners
    
    print(f"\n🏁 PARALLEL PROCESSING COMPLETE!")
    print(f"📊 Results Summary:")
    print(f"   ✅ Successful banners: {successful_banners}/{len(simple_banner_queries)}")
    print(f"   ❌ Failed banners: {failed_banners}/{len(simple_banner_queries)}")
    print(f"   ⏱️  Total parallel time: {total_time:.2f} seconds")
    print(f"   ⚡ Average time per banner: {total_time/len(simple_banner_queries):.2f} seconds")
    
    if successful_banners > 0:
        avg_individual_time = sum(r['time_taken'] for r in results if r['success']) / successful_banners
        speedup_factor = avg_individual_time * len(simple_banner_queries) / total_time
        print(f"   🔥 Estimated speedup factor: {speedup_factor:.1f}x")
    
    print(f"\n📁 Generated files:")
    for result in results:
        if result['success']:
            print(f"   ✅ {result['filename']} - {result['json_length']} chars - {result['time_taken']:.2f}s")
        else:
            print(f"   ❌ Banner {result['index']+1} failed: {result.get('error', 'Unknown error')}")
