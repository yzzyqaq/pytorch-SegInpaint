from PIL import Image, ImageDraw
import os
import random

# Output directory to store generated mask images
output_dir = 'test/mask512256'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Image size (assuming the image size is 256x256)
image_size = (512, 256)

# Generate 5000 mask images
for i in range(5000):
    # Create an all-white image
    mask = Image.new('L', image_size, color=255)

    # Randomly generate the top-left coordinates of the masked region
    x = random.randint(0, image_size[0] // 2)
    y = random.randint(0, image_size[1] // 2)

    # Randomly generate the width and height of the masked region
    width = random.randint(image_size[0] // 4, image_size[0] // 2)
    height = random.randint(image_size[1] // 4, image_size[1] // 2)

    # Draw a black rectangle on the mask image to represent the masked region
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x, y, x + width, y + height], fill=0)

    # Save the mask to the output directory
    mask_path = os.path.join(output_dir, f'mask_{i + 1}.png')
    mask.save(mask_path)

print("Generation complete.")
