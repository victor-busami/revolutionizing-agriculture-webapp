from PIL import Image, ImageDraw, ImageFont
import os

# Create a new image with green background
img_width = 800
img_height = 500
background_color = (46, 139, 87)  # Sea green - agricultural theme
image = Image.new('RGB', (img_width, img_height), background_color)

# Get a drawing context
draw = ImageDraw.Draw(image)

# Draw lighter green squares and circles for decorative effect
for i in range(20):
    x = i * 40
    y = i * 25
    size = 100 - i * 3
    lighter_green = (min(46 + i * 5, 255), min(139 + i * 3, 255), min(87 + i * 4, 255))
    draw.rectangle((x, y, x + size, y + size), fill=lighter_green, outline=None)
    
    # Draw some circular elements
    cx = img_width - i * 30
    cy = i * 25
    radius = 40 - i
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), 
                 fill=(255, 255, 255, 128), outline=None)

# Draw a stylized leaf shape
leaf_points = [
    (400, 150),  # top
    (450, 200),  # right
    (400, 350),  # bottom
    (350, 200)   # left
]
draw.polygon(leaf_points, fill=(35, 110, 70))

# Add a stem to the leaf
draw.rectangle((395, 320, 405, 400), fill=(35, 110, 70))

# Save the image
image_path = os.path.join('app', 'static', 'img', 'agriculture_hero.png')
image.save(image_path)

print(f"Image saved to {image_path}")
