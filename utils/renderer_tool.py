import json
import asyncio
import os
from pathlib import Path
from playwright.async_api import async_playwright

# Fabric JSON (sample)
fabric_json = {
    "version": "4.6.0",
    "objects": [
        {
            "type": "rect",
            "left": 100,
            "top": 100,
            "width": 200,
            "height": 200,
            "fill": "red"
        },
        {
            "type": "text",
            "left": 150,
            "top": 150,
            "text": "Hello World",
            "fontSize": 30,
            "fontFamily": "Roboto"
        }
    ]
}

# Generate HTML file with embedded fabric JSON
def create_html(fabric_json, html_file, width=800, height=600):
    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Fabric Render</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/fabric.js/4.6.0/fabric.min.js"></script>
        <style>
            html, body {{ margin: 0; padding: 0; }}
            canvas {{ display: block; }}
        </style>
    </head>
    <body>
        <canvas id="c" width="{width}" height="{height}"></canvas>
        <script>
            async function loadFonts() {{
                const font = new FontFace('Roboto', 'url(https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu4mxP.ttf)');
                await font.load();
                document.fonts.add(font);
            }}

            async function load() {{
                await loadFonts();
                const canvas = new fabric.Canvas('c');
                const json = {json.dumps(fabric_json)};
                await new Promise(resolve => {{
                    canvas.loadFromJSON(json, () => {{
                        canvas.renderAll();
                        requestAnimationFrame(resolve);
                    }});
                }});
                window.renderDone = true;
            }}

            load();
        </script>
    </body>
    </html>
    """
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

async def render_and_screenshot(html_file, output_file, width=800, height=600):
    html_path = Path(html_file).absolute().as_uri()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": width, "height": height})
        await page.goto(html_path)

        # Wait until JS finishes rendering
        await page.wait_for_function("() => window.renderDone === true")

        await page.screenshot(path=output_file)

        await browser.close()

if __name__ == "__main__":
    html_file = "fabric_render.html"
    output_file = "output.png"
    canvas_width, canvas_height = 800, 600

    # Step 1: generate HTML
    create_html(fabric_json, html_file, canvas_width, canvas_height)

    # Step 2: open HTML and take screenshot
    asyncio.run(render_and_screenshot(html_file, output_file, canvas_width, canvas_height))

    print("Screenshot saved:", output_file)
