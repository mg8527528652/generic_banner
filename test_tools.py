from utils.image_tools import background_remover, background_replacer, text_to_image_generator

# Test image URL
test_image = "https://images.pexels.com/photos/106399/pexels-photo-106399.jpeg"

# 1. Test background remover
print("Testing background remover...")
mask_url = background_remover.run({"img_url": test_image})
print(f"Mask URL: {mask_url}")

# 2. Test background replacer
print("\nTesting background replacer...")
new_bg_url = background_replacer.run({
    "image_url": test_image,
    "mask_url": mask_url,
    "prompt": "A beautiful beach with palm trees",
    "batch_size": 1
})
print(f"New background image URL: {new_bg_url}")

# 3. Test text to image generator
print("\nTesting text to image generator...")
generated_image_url = text_to_image_generator.run({
    "prompt": "A serene mountain landscape at sunset",
    "width": 1024,
    "height": 1024,
    "out_path": "test_generated.png"
})
print(f"Generated image URL: {generated_image_url}") 