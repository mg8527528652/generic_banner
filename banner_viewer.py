import json
import os

# The HTML template. The {json_data}, {width}, {height}, and {filename} placeholders
# will be replaced by the script.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fabric.js Renderer</title>
    <!-- 1. Fabric.js Library -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/fabric.js/5.3.0/fabric.min.js"></script>
    <style>
        body {{
            background-color: #f0f0f0;
            color: #333;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        .info-box {{
            background-color: #fffbe6;
            border: 1px solid #ffe58f;
            border-radius: 8px;
            padding: 15px 20px;
            margin-bottom: 20px;
            max-width: {width}px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        .info-box strong {{
            color: #d46b08;
        }}
        .info-box code {{
            background-color: #e8e8e8;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: "Courier New", Courier, monospace;
        }}
        canvas {{
            border: 1px solid #ccc;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        /*
         * FONT LOADING:
         * Your Fabric.js JSON may use custom fonts (e.g., 'Montserrat', 'Inter').
         * For them to render correctly, you must load them here.
         *
         * Example for Google Fonts:
         * @import url('https://fonts.googleapis.com/css2?family=Montserrat&family=Inter&display=swap');
        */
    </style>
</head>
<body>

    <div class="info-box">
        <strong>Important:</strong> If images are not showing up, it is likely due to a browser security policy (CORS). You cannot open this HTML file directly from your computer (e.g., using a <code>file:///...</code> path).
        <br><br>
        <strong>Solution:</strong> You must serve it from a local web server. If you have Python installed, open a terminal in this directory and run: <code>python -m http.server</code> then open <a href="http://localhost:8000/{filename}" target="_blank">http://localhost:8000/{filename}</a> in your browser.
    </div>

    <!-- 2. The Canvas Element -->
    <canvas id="canvas"></canvas>

    <script>
        // 3. The Fabric.js JSON data from your file
        const fabricJSON = {json_data};

        // 4. Initialization and Rendering Logic
        (function() {{
            // This self-running function handles the rendering process.

            // AUTOMATIC GRADIENT FIX: This part checks for the old gradient format
            // and converts it to the array format that Fabric.js v5 expects.
            if (fabricJSON.objects) {{
                fabricJSON.objects.forEach(obj => {{
                    const fill = obj.fill;
                    if (fill && typeof fill === 'object' && fill.colorStops && !Array.isArray(fill.colorStops)) {{
                        console.warn("Found and fixing legacy gradient format for object:", obj);
                        try {{
                            const newColorStops = Object.entries(fill.colorStops).map(([offset, color]) => ({{ offset: parseFloat(offset), color }}));
                            fill.colorStops = newColorStops;
                        }} catch (e) {{
                            console.error("Failed to automatically fix gradient format. The canvas may not render correctly.", e);
                        }}
                    }}
                }});
            }}

            // Get canvas dimensions from JSON, with fallback default values
            const canvasWidth = fabricJSON.width || 1024;
            const canvasHeight = fabricJSON.height || 1024;

            const canvasEl = document.getElementById('canvas');
            canvasEl.width = canvasWidth;
            canvasEl.height = canvasHeight;

            const canvas = new fabric.Canvas('canvas');

            console.log("Loading Fabric.js canvas from JSON...");
            canvas.loadFromJSON(fabricJSON, function() {{
                canvas.renderAll();
                console.log('Canvas loaded successfully!');
            }});
        }})();
    </script>

</body>
</html>
"""

def create_html_from_fabric_json(json_path, output_path):
    """
    Reads a Fabric.js JSON file and generates a self-contained HTML file
    to render it on a canvas.

    Args:
        json_path (str): The path to the input Fabric.js JSON file.
        output_path (str): The path where the output HTML file will be saved.
    """
    print(f"Reading Fabric JSON from: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_content = f.read()
        
        # Load the JSON to check for validity and get dimensions
        json_data = json.loads(json_content)
        canvas_width = json_data.get('width', 1024)
        canvas_height = json_data.get('height', 1024)

    except FileNotFoundError:
        print(f"Error: The file '{json_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{json_path}' is not a valid JSON file.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    # Populate the HTML template with the JSON data and canvas dimensions
    output_html = HTML_TEMPLATE.format(
        json_data=json_content,
        width=canvas_width,
        height=canvas_height,
        filename=os.path.basename(output_path)
    )

    # Write the populated template to the output file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_html)
        print(f"-> Successfully generated HTML file: {output_path}")
    except Exception as e:
        print(f"Error writing to output file '{output_path}': {e}")


def main(input_folder, output_folder):
    """
    Processes all JSON files in an input folder and saves them as HTML
    renderings in the output folder.
    """
    # 1. Ensure the output directory exists, create it if it doesn't
    if not os.path.exists(output_folder):
        print(f"Output folder '{output_folder}' not found. Creating it.")
        os.makedirs(output_folder)

    # 2. Check if the input directory exists
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' not found or is not a directory.")
        return

    # 3. Process each file in the input folder
    print(f"\nScanning for JSON files in '{input_folder}'...")
    json_files_found = 0
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".json"):
            json_files_found += 1
            json_path = os.path.join(input_folder, filename)
            
            # Create a corresponding output path with an .html extension
            html_filename = os.path.splitext(filename)[0] + ".html"
            output_path = os.path.join(output_folder, html_filename)
            
            # Call the main conversion function for the current file
            create_html_from_fabric_json(json_path, output_path)
    
    if json_files_found == 0:
        print("No JSON files were found in the specified input folder.")
    else:
        print(f"\nProcessed {json_files_found} JSON file(s).")


if __name__ == "__main__":
    # --- HOW TO USE THIS SCRIPT ---
    # 1. Set the path to the folder containing your Fabric.js JSON files.
    # 2. Set the path to the folder where you want to save the generated HTML files.
    
    # --- IMPORTANT: UPDATE THESE PATHS ---
    input_directory = "generated_banners"
    output_directory = "data/generated_banners_html"
    
    # --- Run the script ---
    # This check prevents the script from running with the default placeholder paths.
    if "path/to/your" in input_directory or "path/to/your" in output_directory:
        print("="*60)
        print("!! PLEASE UPDATE THE 'input_directory' and 'output_directory' !!")
        print("   variables in the script before running.")
        print("="*60)
    else:
        main(input_directory, output_directory)
