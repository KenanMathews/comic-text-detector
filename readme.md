# Comic Text Detector

An interactive web application for detecting and extracting text from comic images. The application provides a user-friendly interface for uploading images and detecting text regions with a simple click.

## Features

- Image upload and display
- Click-based text detection
- Support for speech bubbles and text regions
- Real-time text detection and extraction
- Drawing tools for annotations
- Undo/Redo functionality
- Responsive design

## Technology Stack

### Frontend
- React
- Tailwind CSS
- Axios for API calls
- Canvas for drawing
- SVG for box rendering

### Backend
- FastAPI
- OpenCV for image processing
- Tesseract OCR for text extraction
- NumPy for numerical operations
- Python 3.8+

## Prerequisites

Before running the application, ensure you have the following installed:
- Python 3.8 or higher
- Node.js 14 or higher
- Tesseract OCR

## Installation

### Backend Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node dependencies:
```bash
npm install
```

## Running the Application

### Start the Backend Server

1. Activate the virtual environment if not already activated
2. Run the FastAPI server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Start the Frontend Development Server

1. In a separate terminal, navigate to the frontend directory
2. Start the React development server:
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

## Usage

1. Upload an image using the upload button
2. Select the "Detect Text" tool from the toolbar
3. Click on any text region in the image
4. The application will detect and highlight the text region
5. Use the drawing tools to make additional annotations if needed
6. Use undo/redo for any mistakes

## API Endpoints

- `GET /`: Root endpoint
- `GET /health`: Health check endpoint
- `POST /api/detect-at-click`: Text detection endpoint
  - Accepts click coordinates and image data
  - Returns detected text regions and processed image

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to your branch
5. Create a Pull Request


## Acknowledgments

- OpenCV for image processing capabilities
- Tesseract OCR for text extraction
- FastAPI for the efficient backend framework
- React for the responsive frontend