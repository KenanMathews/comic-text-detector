import cv2
import numpy as np
from PIL import Image
import pytesseract
import logging
from typing import List, Tuple, Set
from pydantic import BaseModel

# Define the ViewInfo model
class ViewInfo(BaseModel):
    x: int  # Click x coordinate
    y: int  # Click y coordinate
    view_width: int  # Width of the current view
    view_height: int  # Height of the current view
    view_x: int  # X offset of the view
    view_y: int  # Y offset of the view
    scale: float  # Current scale factor of the view
    image_data: str  # Base64 image data

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess image to prepare for text detection."""
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Normalize image
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Enhanced text detection with multiple thresholds
        binary1 = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15,  # Block size
            2    # C constant
        )
        
        binary2 = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            25,  # Larger block size
            5    # Larger C constant
        )
        
        # Combine both thresholds
        binary = cv2.bitwise_or(binary1, binary2)
        
        # Enhanced morphological operations for better text connection
        # Horizontal connection
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        # Vertical connection
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        # General dilation kernel
        kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # First connect horizontally
        text_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
        # Then connect vertically (less aggressively)
        text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel_v)
        # Finally dilate slightly to connect remaining components
        text_mask = cv2.dilate(text_mask, kernel_d, iterations=1)
        
        return gray, binary, text_mask
        
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        raise

def recursive_region_grow(
    image: np.ndarray,
    binary: np.ndarray,
    start_x: int,
    start_y: int,
    visited: Set[Tuple[int, int]],
    max_size: int = 1000
) -> np.ndarray:
    """
    Recursively grow region from click point to find text boundaries.
    
    Args:
        image: Grayscale input image
        binary: Binary image with text regions
        start_x, start_y: Starting coordinates
        visited: Set of visited pixels
        max_size: Maximum region size to prevent unbounded growth
    
    Returns:
        region_mask: Binary mask of detected region
    """
    height, width = image.shape[:2]
    region_mask = np.zeros((height, width), dtype=np.uint8)
    
    def is_valid(x: int, y: int) -> bool:
        return 0 <= x < width and 0 <= y < height and (x,y) not in visited
    
    def is_boundary(x: int, y: int) -> bool:
        # Check for both edge strength and text density
        if x > 0 and x < width-1 and y > 0 and y < height-1:
            # Edge strength check with lower threshold
            grad_x = cv2.Sobel(image[y-1:y+2, x-1:x+2], cv2.CV_64F, 1, 0)
            grad_y = cv2.Sobel(image[y-1:y+2, x-1:x+2], cv2.CV_64F, 0, 1)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Text density check in larger neighborhood
            neighborhood = binary[max(0, y-10):min(height, y+11), 
                                max(0, x-10):min(width, x+11)]
            text_density = np.sum(neighborhood > 0) / neighborhood.size
            
            # Return true if either condition is met with adjusted thresholds
            edge_threshold = 30  # Lowered from 50
            density_threshold = 0.02  # Lowered from 0.05
            return np.max(magnitude) > edge_threshold or text_density < density_threshold
        return False
    
    def grow_region(x: int, y: int, current_size: int) -> int:
        if not is_valid(x, y) or current_size > max_size:
            return current_size
            
        visited.add((x,y))
        region_mask[y,x] = 255
        current_size += 1
        
        # Check for boundary
        if is_boundary(x, y):
            return current_size
            
        # Recursive growth in 8 directions
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dx, dy in directions:
            if is_valid(x+dx, y+dy):
                current_size = grow_region(x+dx, y+dy, current_size)
                
        return current_size
    
    grow_region(start_x, start_y, 0)
    return region_mask

def detect_text_boundaries(
    image: np.ndarray,
    view_info: ViewInfo
) -> Tuple[List[dict], np.ndarray]:
    """
    Detect text boundaries using flood fill and bubble detection.
    Takes into account the current view information.
    """
    try:
        logger.debug(f"Starting text boundary detection with view info: {view_info}")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        height, width = gray.shape[:2]
        
        # Calculate click coordinates in image space
        image_x = int(view_info.x * view_info.scale + view_info.view_x)
        image_y = int(view_info.y * view_info.scale + view_info.view_y)
        
        logger.debug(f"Converted click coordinates: ({image_x}, {image_y})")
        
        # Calculate ROI around click point
        roi_size = int(300 * view_info.scale)  # Scale ROI size based on view scale
        x1 = max(0, image_x - roi_size // 2)
        y1 = max(0, image_y - roi_size // 2)
        x2 = min(width, image_x + roi_size // 2)
        y2 = min(height, image_y + roi_size // 2)
        
        roi = gray[y1:y2, x1:x2]
        roi_color = image[y1:y2, x1:x2].copy()
        
        # Threshold to get white regions (speech bubbles)
        _, binary_bubbles = cv2.threshold(roi, 240, 255, cv2.THRESH_BINARY)
        
        # Find the speech bubble containing the click point
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_bubbles)
        
        # Adjust click point to ROI coordinates
        roi_click_x = image_x - x1
        roi_click_y = image_y - y1
        
        # Find which component contains the click point
        click_label = labels[roi_click_y, roi_click_x] if 0 <= roi_click_y < labels.shape[0] and 0 <= roi_click_x < labels.shape[1] else 0
        
        if click_label == 0:  # If click is on dark pixel, search neighborhood
            logger.debug("Click point on dark pixel, searching neighborhood")
            neighborhood_size = 20
            nx1 = max(0, roi_click_x - neighborhood_size)
            ny1 = max(0, roi_click_y - neighborhood_size)
            nx2 = min(labels.shape[1], roi_click_x + neighborhood_size)
            ny2 = min(labels.shape[0], roi_click_y + neighborhood_size)
            
            neighborhood = labels[ny1:ny2, nx1:nx2]
            unique_labels = np.unique(neighborhood)
            unique_labels = unique_labels[unique_labels != 0]  # Remove background
            
            if len(unique_labels) > 0:
                click_label = unique_labels[0]
        
        if click_label > 0:
            logger.debug(f"Found containing bubble with label {click_label}")
            # Get the bounding box of the bubble
            bubble_x = stats[click_label, cv2.CC_STAT_LEFT]
            bubble_y = stats[click_label, cv2.CC_STAT_TOP]
            bubble_w = stats[click_label, cv2.CC_STAT_WIDTH]
            bubble_h = stats[click_label, cv2.CC_STAT_HEIGHT]
            
            # Create a mask for the bubble
            bubble_mask = (labels == click_label).astype(np.uint8) * 255
            
            # Find text within the bubble
            # Threshold for text (black pixels)
            _, binary_text = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
            binary_text = cv2.bitwise_and(binary_text, bubble_mask)
            
            # Clean up text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            binary_text = cv2.morphologyEx(binary_text, cv2.MORPH_CLOSE, kernel)
            
            # Find text contours
            contours, _ = cv2.findContours(
                binary_text,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            logger.debug(f"Found {len(contours)} text contours")
            
            # Create output image
            output_img = image.copy()
            boxes = []
            
            if contours:
                # Create a mask for all text
                text_mask = np.zeros_like(binary_text)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 50:  # Filter very small contours
                        cv2.drawContours(text_mask, [cnt], -1, 255, -1)
                
                # Merge nearby text
                kernel_merge = cv2.getStructuringElement(cv2.MORPH_RECT, (20,10))
                text_mask = cv2.dilate(text_mask, kernel_merge)
                text_mask = cv2.erode(text_mask, kernel_merge)
                
                # Find contours of merged text
                merged_contours, _ = cv2.findContours(
                    text_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                for cnt in merged_contours:
                    rx, ry, rw, rh = cv2.boundingRect(cnt)
                    # Convert back to original image coordinates
                    box = {
                        'x': int(x1 + rx),
                        'y': int(y1 + ry),
                        'width': int(rw),
                        'height': int(rh)
                    }
                    boxes.append(box)
                    cv2.rectangle(
                        output_img,
                        (box['x'], box['y']),
                        (box['x'] + box['width'], box['y'] + box['height']),
                        (0, 255, 0),
                        2
                    )
            
            # If no text regions found, use the bubble bounds
            if not boxes:
                box = {
                    'x': int(x1 + bubble_x),
                    'y': int(y1 + bubble_y),
                    'width': int(bubble_w),
                    'height': int(bubble_h)
                }
                boxes.append(box)
                cv2.rectangle(
                    output_img,
                    (box['x'], box['y']),
                    (box['x'] + box['width'], box['y'] + box['height']),
                    (0, 255, 0),
                    2
                )
            
            logger.debug(f"Returning {len(boxes)} boxes")
            return boxes, output_img
            
        # Fallback: return a region around the click point
        logger.debug("No bubble found, using fallback region")
        fallback_size = int(200 * view_info.scale)
        box = {
            'x': max(0, image_x - fallback_size // 2),
            'y': max(0, image_y - fallback_size // 2),
            'width': min(width - x1, fallback_size),
            'height': min(height - y1, fallback_size)
        }
        
        output_img = image.copy()
        cv2.rectangle(
            output_img,
            (box['x'], box['y']),
            (box['x'] + box['width'], box['y'] + box['height']),
            (0, 255, 0),
            2
        )
        
        return [box], output_img
            
    except Exception as e:
        logger.error(f"Error in detect_text_boundaries: {str(e)}")
        raise

def extract_text(image: np.ndarray, box: dict) -> str:
    """
    Extract text from a given region using OCR.
    
    Args:
        image: Input image
        box: Bounding box dictionary with x, y, width, height
    
    Returns:
        text: Extracted text string
    """
    try:
        # Extract coordinates from box dictionary
        x = box['x']
        y = box['y']
        w = box['width']
        h = box['height']
        
        # Extract ROI
        roi = image[y:y+h, x:x+w]
        
        # Convert to PIL Image for Tesseract
        pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        
        # Extract text with Tesseract
        text = pytesseract.image_to_string(pil_img)
        logger.debug(f"Extracted text: {text.strip()}")
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Error in extract_text: {str(e)}")
        raise