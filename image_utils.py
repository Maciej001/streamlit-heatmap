import numpy as np
import cv2
import PIL.Image
from typing import Tuple, List
from data_schemas import AOIS
from data_schemas import AttentionAnalysis
from data_schemas import BoundingBox
from PIL import Image, ImageDraw

def normalize_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    """
    Normalizes bounding box coordinates to pixel values.
    1. Normalize width and height from 1000x1000 to actual width x height
    2. Convert coordinates [y1, x1, y2, x2] to [x1, y1, x2, y2]

    Args:
        bounding_box: Tuple of int coordinates (y1, x1, y2, x2)
        width: Width of the original image in pixels
        height: Height of the original image in pixels

    Returns:
        Tuple of coordinates()
    """
    y1, x1, y2, x2 = bbox
    # Convert normalized coordinates to absolute coordinates
    abs_y1 = int(y1/1000 * height)
    abs_x1 = int(x1/1000 * width)
    abs_y2 = int(y2/1000 * height)
    abs_x2 = int(x2/1000 * width)

    if abs_x1 > abs_x2:
      abs_x1, abs_x2 = abs_x2, abs_x1

    if abs_y1 > abs_y2:
      abs_y1, abs_y2 = abs_y2, abs_y1

    return [abs_x1, abs_y1, abs_x2, abs_y2]
  

def overlay_bounding_boxes(image: PIL.Image.Image, aois) -> PIL.Image.Image:
    """
    Overlays bounding boxes on an image.

    Args:
        image: Original image
        aois: AttentionAnalysis object containing the elements
    """
    try:
        # Convert image to RGBA if it isn't already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
            
        width, height = image.size

        # Create a transparent overlay image
        overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)

        # Draw boxes and labels
        for aoi in aois:
            bbox = aoi["bounding_box"]
            label = aoi["label"]
            score = aoi["attention_score"]

            # Draw rectangle
            draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline="green", width=2)

            # Add label and score
            text_pos = (bbox[0], bbox[1] - 15) 
            draw.text(text_pos, f"{label}: {score:.1f}", fill="green")

        # Combine images and save
        combined_image = Image.alpha_composite(image, overlay)
        return combined_image

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def create_color_mapping(value):
    """
    Create RGBA color values for a given intensity value (0-1).
    Color transitions with varying transparency levels:
    - 0.0-0.33: Transparent to Green (alpha: 0 to 0.5)
    - 0.33-0.66: Green to Yellow (alpha: 0.5)
    - 0.66-0.85: Yellow to Orange (alpha: 0.5 to 0.8)
    - 0.85-1.0: Orange to Red (alpha: 0.8)
    """
    if value < 0.25:  # First transition
        # Transparent to Green
        ratio = value / 0.25
        alpha = ratio * 0.5
        return (0, 1, 0, alpha)
    elif value < 0.5:  # Second transition
        # Green to Yellow
        ratio = (value - 0.25) / 0.25
        return (ratio, 1, 0, 0.5)
    elif value < 0.65:  # Start orange transition earlier
        # Yellow to Orange
        ratio = (value - 0.5) / 0.15
        alpha = 0.5 + (ratio * 0.2)  # Smaller alpha change
        return (1, 1 - ratio/3, 0, alpha)  # Slower green reduction
    else:
        # Orange to Red with smoother transition
        ratio = (value - 0.65) / 0.35  
        green_component = max(0.6 - (ratio * 0.6), 0)  # Slower green reduction
        alpha = min(0.7 + (ratio * 0.2), 0.9)  # Gradual alpha increase
        return (1, green_component, 0, alpha)
      
def create_heatmap(image: PIL.Image.Image, aois: List[AOIS]) -> PIL.Image.Image:
    """
    Create a heatmap overlay with smooth circular gradients and transparency.
    """
    # Read image using PIL first to handle PNG properly
    
    # Sigma Clamping
    SIGMA_MIN = 5
    SIGMA_MAX = 80
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    original_image = np.array(image)
    height, width = original_image.shape[:2]
    
    # Create coordinate arrays
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    X, Y = np.meshgrid(x, y)
    
    # Initialize heatmap array
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Generate gaussian heatmap
    for element in aois:
        x1, y1, x2, y2 = element['bounding_box']
        box_width = x2 - x1
        box_height = y2 - y1
        
        score = element['attention_score']
        
        # scale the score by a factor inversely proportional 
        # to its area, so smaller bounding boxes get a gentle boost, while bigger ones do not dominate.
        area = box_width * box_height
        area_factor = 1.0 / (np.sqrt(area) + 1e-8)
        scaled_score = score 
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate separate x and y standard deviations based on box dimensions
        # Add clamping to prevent extreme values
        sigma_x = max(SIGMA_MIN, min(SIGMA_MAX, box_width / 6))
        sigma_y = max(SIGMA_MIN, min(SIGMA_MAX, box_height / 6))

        
        # Calculate separate squared distances for x and y
        squared_dist_x = (X - center_x)**2 / (2 * sigma_x**2)
        squared_dist_y = (Y - center_y)**2 / (2 * sigma_y**2)
        
        # Combine for oval-shaped gaussian with increased center intensity
        gaussian = scaled_score * np.exp(-(squared_dist_x + squared_dist_y))
        
        # Apply power function to increase contrast
        gaussian = gaussian ** 0.7  # Values less than 1 will increase intensity
        
        # Accumulate in heatmap
        heatmap += gaussian
    
    # Manual min-max normalization across entire array
    heatmap_min = np.min(heatmap)
    heatmap_max = np.max(heatmap)
    heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
    
    # Apply gaussian blur to smooth transitions
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=5, sigmaY=5)
    
    # Create colored heatmap with alpha channel
    heatmap_colored = np.zeros((height, width, 4), dtype=np.float32)
    
    # Vectorized color mapping with alpha channel
    for i in range(height):
        for j in range(width):
            color = create_color_mapping(heatmap[i, j])
            heatmap_colored[i, j] = color
    
    # Convert to uint8
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Create PIL images for blending with alpha
    original_pil = Image.fromarray(original_image)
    heatmap_pil = Image.fromarray(heatmap_colored, 'RGBA')
    
    # Blend images using PIL's alpha compositing
    blended = Image.alpha_composite(original_pil.convert('RGBA'), heatmap_pil)
    
    # return np.array(blended)
    return blended