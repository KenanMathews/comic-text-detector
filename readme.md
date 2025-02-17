Comic Text Detector
An interactive web application for detecting and extracting text from comic images. The application provides a user-friendly interface for uploading images and detecting text regions with a simple click.

Features
Image upload and display
Click-based text detection
Support for speech bubbles and text regions
Real-time text detection and extraction
Drawing tools for annotations
Undo/Redo functionality
Responsive design
Technology Stack
Frontend
React
Tailwind CSS
Axios for API calls
Canvas for drawing
SVG for box rendering
Backend
FastAPI
OpenCV for image processing
Tesseract OCR for text extraction
NumPy for numerical operations
Python 3.8+
Prerequisites
Before running the application, ensure you have the following installed:

Python 3.8 or higher
Node.js 14 or higher
Tesseract OCR
Installation
Backend Setup
Create and activate a virtual environment:
bash

Copy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Python dependencies:
bash

Copy
pip install -r requirements.txt
Frontend Setup
Navigate to the frontend directory:
bash

Copy
cd frontend
Install Node dependencies:
bash

Copy
npm install
Running the Application
Start the Backend Server
Activate the virtual environment if not already activated
Run the FastAPI server:
bash

Copy
uvicorn main:app --reload --host 0.0.0.0 --port 8000
Start the Frontend Development Server
In a separate terminal, navigate to the frontend directory
Start the React development server:
bash

Copy
npm run dev
The application will be available at http://localhost:5173

Usage
Upload an image using the upload button
Select the "Detect Text" tool from the toolbar
Click on any text region in the image
The application will detect and highlight the text region
Use the drawing tools to make additional annotations if needed
Use undo/redo for any mistakes
API Endpoints
GET /: Root endpoint
GET /health: Health check endpoint
POST /api/detect-at-click: Text detection endpoint
Accepts click coordinates and image data
Returns detected text regions and processed image
Contributing
Fork the repository
Create a new branch for your feature
Commit your changes
Push to your branch
Create a Pull Request
License
[Your chosen license]

Acknowledgments
OpenCV for image processing capabilities
Tesseract OCR for text extraction
FastAPI for the efficient backend framework
React for the responsive frontend