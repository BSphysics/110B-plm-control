# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 16:30:08 2025

@author: Ben
"""

from PIL import Image, ImageDraw, ImageFont
import glob
import os

def gifMaker(image_folder):

    # img_folder = '2025_02_18_17_53_12 Basler frames'
    
    # input_folder = os.path.join(os.getcwd(), img_folder)
    
    # Parameters
    
    output_gif = os.path.join(image_folder, 'animation.gif')
    duration = 100  # Duration per frame in milliseconds (2 seconds per frame)
    
    # Get all PNG images sorted by name
    image_files = sorted(glob.glob(f"{image_folder}/*.png"))
    
    # Ensure there are images to process
    if not image_files:
        raise ValueError("No PNG images found in the specified directory.")
    
    # Load a font (default PIL font if no specific font is available)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()
    
    # Function to add text to image
    def add_title(image, text):
        draw = ImageDraw.Draw(image)
        text_position = (10, 10)
        text_color = "white"
        draw.text(text_position, text, fill=text_color, font=font)
        return image
    
    # Open images, add titles, and convert to RGB (required for GIF compatibility)
    images = []
    for img_path in image_files:
        img = Image.open(img_path).convert("L")
        title = os.path.basename(img_path)
        img = add_title(img, title[6:])
        images.append(img)
    
    # Save as GIF
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration,  # Control animation speed
        loop=0  # Loop indefinitely
    )
    
    print(f"GIF saved successfully as {output_gif}")