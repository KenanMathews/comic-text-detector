from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import logging
from typing import List, Dict, Any
from detector import detect_text_boundaries, extract_text

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

app = FastAPI(title="Comic Text Detector")

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Comic Text Detector API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/detect-at-click")
async def detect_text_at_click(view_info: ViewInfo) -> Dict[str, Any]:
    """
    Detect text boundaries at the clicked point in the image, considering view information.
    
    Args:
        view_info: ViewInfo object containing click coordinates, view dimensions, and image data
    
    Returns:
        JSON response with detected boxes, processed image, and original size
    """
    try:
        logger.debug(f"Received click at coordinates: ({view_info.x}, {view_info.y})")
        logger.debug(f"View dimensions: {view_info.view_width}x{view_info.view_height}")
        logger.debug(f"View offset: ({view_info.view_x}, {view_info.view_y})")
        logger.debug(f"Scale factor: {view_info.scale}")
        
        # Remove data URL prefix if present
        image_data = view_info.image_data
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            logger.debug("Removed data URL prefix from image data")
        
        try:
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            logger.debug("Successfully decoded base64 image data")
            
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Could not decode image")
            
            logger.debug(f"Successfully decoded image of shape: {image.shape}")
            
            # Detect text boundaries
            boxes, processed_image = detect_text_boundaries(
                image,
                view_info
            )
            
            logger.debug(f"Detected {len(boxes)} text boxes")
            
            # Extract text from each box
            texts = []
            for box in boxes:
                text = extract_text(image, box)
                texts.append(text)
            
            # Format response with boxes in view coordinates
            formatted_boxes = [
                {
                    'x': int((box['x'] - view_info.view_x) / view_info.scale),
                    'y': int((box['y'] - view_info.view_y) / view_info.scale),
                    'width': int(box['width'] / view_info.scale),
                    'height': int(box['height'] / view_info.scale),
                    'text': text
                }
                for box, text in zip(boxes, texts)
            ]
            
            # Encode processed image
            success, encoded_img = cv2.imencode(
                '.png',
                processed_image,
                [cv2.IMWRITE_PNG_COMPRESSION, 9]
            )
            
            if not success:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to encode processed image"
                )
            
            # Convert processed image to base64
            base64_image = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
            logger.debug("Successfully encoded processed image")
            
            return JSONResponse({
                "boxes": formatted_boxes,
                "processed_image": base64_image,
                "original_size": {
                    "width": image.shape[1],
                    "height": image.shape[0]
                }
            })
            
        except ValueError as ve:
            logger.error(f"Base64 decoding error: {str(ve)}")
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
            
    except Exception as e:
        logger.error(f"Error in detect_text_at_click: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )