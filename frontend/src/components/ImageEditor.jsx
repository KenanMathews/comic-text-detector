import React, { useState, useRef, useEffect } from "react";
import {
  Upload,
  Pencil,
  MousePointer,
  Eraser,
  Square,
  Circle,
  Undo,
  Redo,
  Type,
} from "lucide-react";
import axios from "axios";

const ImageEditor = () => {
  const [image, setImage] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [boxes, setBoxes] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [tool, setTool] = useState("select");
  const [drawings, setDrawings] = useState([]);
  const [currentPath, setCurrentPath] = useState([]);
  const [undoStack, setUndoStack] = useState([]);
  const [redoStack, setRedoStack] = useState([]);
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });

  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const containerRef = useRef(null);
  const isDrawing = useRef(false);

  const tools = [
    { id: "select", icon: MousePointer, name: "Select" },
    { id: "draw", icon: Pencil, name: "Draw" },
    { id: "erase", icon: Eraser, name: "Erase" },
    { id: "rectangle", icon: Square, name: "Rectangle" },
    { id: "circle", icon: Circle, name: "Circle" },
    { id: "textdetect", icon: Type, name: "Detect Text" },
  ];

  // Update canvas size when image loads
  useEffect(() => {
    if (imageRef.current) {
      const updateCanvasSize = () => {
        const { width, height } = imageRef.current.getBoundingClientRect();
        setCanvasSize({ width, height });
      };

      updateCanvasSize();
      window.addEventListener("resize", updateCanvasSize);
      return () => window.removeEventListener("resize", updateCanvasSize);
    }
  }, [image]);

  // Update canvas size
  useEffect(() => {
    if (canvasRef.current && canvasSize.width && canvasSize.height) {
      const canvas = canvasRef.current;
      canvas.width = canvasSize.width;
      canvas.height = canvasSize.height;

      // Set up canvas context
      const ctx = canvas.getContext("2d");
      ctx.strokeStyle = tool === "erase" ? "#fff" : "#00ff00";
      ctx.lineWidth = 2;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
    }
  }, [canvasSize, tool]);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    try {
      const reader = new FileReader();
      reader.onload = (e) => {
        setOriginalImage(e.target.result);
        setImage(e.target.result);
        setBoxes([]);
        setDrawings([]);
        setUndoStack([]);
        setRedoStack([]);
        setCurrentPath([]);
      };
      reader.readAsDataURL(file);
    } catch (error) {
      console.error("Error loading image:", error);
      setError("Error loading image");
    }
  };

  const handleClick = async (e) => {
    if (tool !== 'textdetect' || !image || isLoading) return;
  
    const imageElement = imageRef.current;
    const rect = imageElement.getBoundingClientRect();
    
    // Get click coordinates relative to the viewport
    const viewX = e.clientX - rect.left;
    const viewY = e.clientY - rect.top;
    
    // Get the natural dimensions of the image
    const naturalWidth = imageElement.naturalWidth;
    const naturalHeight = imageElement.naturalHeight;
    
    // Calculate scaling factors
    const scaleX = naturalWidth / rect.width;
    const scaleY = naturalHeight / rect.height;
    
    // Get scroll position if image is in a scrollable container
    const container = containerRef.current;
    const scrollX = container ? container.scrollLeft : 0;
    const scrollY = container ? container.scrollTop : 0;
  
    setIsLoading(true);
    setError(null);
  
    try {
      const response = await axios.post('http://localhost:8000/api/detect-at-click', {
        x: Math.round(viewX),
        y: Math.round(viewY),
        view_width: Math.round(rect.width),
        view_height: Math.round(rect.height),
        view_x: Math.round(scrollX),
        view_y: Math.round(scrollY),
        scale: scaleX, // Assuming uniform scaling
        image_data: originalImage
      });
  
      if (response.data.boxes && response.data.boxes.length > 0) {
        // Scale the boxes to view coordinates
        const scaledBoxes = response.data.boxes.map(box => ({
          x: Math.round(box.x / scaleX),
          y: Math.round(box.y / scaleY),
          width: Math.round(box.width / scaleX),
          height: Math.round(box.height / scaleY),
          text: box.text
        }));
  
        setBoxes(prevBoxes => [...prevBoxes, ...scaledBoxes]);
        setUndoStack(prev => [...prev, { type: 'textbox', boxes: scaledBoxes }]);
        setRedoStack([]);
      }
    } catch (error) {
      console.error('Error detecting text:', error);
      setError(error.response?.data?.detail || 'Error detecting text');
    } finally {
      setIsLoading(false);
    }
  };

  const getPointerPosition = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    };
  };

  const startDrawing = (e) => {
    if (!image || tool === "select" || tool === "textdetect") return;

    isDrawing.current = true;
    const pos = getPointerPosition(e);
    setCurrentPath([pos]);

    // Start new path in canvas
    const ctx = canvasRef.current.getContext("2d");
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  };

  const draw = (e) => {
    if (!isDrawing.current || !image) return;

    const pos = getPointerPosition(e);
    setCurrentPath((prev) => [...prev, pos]);

    // Draw on canvas
    const ctx = canvasRef.current.getContext("2d");
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    if (!isDrawing.current) return;

    isDrawing.current = false;
    if (currentPath.length > 0) {
      const newDrawing = { tool, points: currentPath };
      setDrawings((prev) => [...prev, newDrawing]);
      setUndoStack((prev) => [
        ...prev,
        { type: "drawing", drawing: newDrawing },
      ]);
      setRedoStack([]);

      // Clear canvas for next drawing
      const ctx = canvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);
    }
    setCurrentPath([]);
  };

  const redrawCanvas = () => {
    if (!canvasRef.current) return;

    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);

    drawings.forEach((drawing) => {
      ctx.beginPath();
      const points = drawing.points;
      if (points.length > 0) {
        ctx.moveTo(points[0].x, points[0].y);
        points.slice(1).forEach((point) => {
          ctx.lineTo(point.x, point.y);
        });
        ctx.stroke();
      }
    });
  };

  // Redraw canvas when drawings change
  useEffect(() => {
    redrawCanvas();
  }, [drawings, canvasSize]);

  const undo = () => {
    if (undoStack.length === 0) return;

    const lastAction = undoStack[undoStack.length - 1];
    setUndoStack((prev) => prev.slice(0, -1));
    setRedoStack((prev) => [...prev, lastAction]);

    if (lastAction.type === "drawing") {
      setDrawings((prev) => prev.slice(0, -1));
    } else if (lastAction.type === "textbox") {
      setBoxes((prev) => prev.slice(0, -lastAction.boxes.length));
    }
  };

  const redo = () => {
    if (redoStack.length === 0) return;

    const nextAction = redoStack[redoStack.length - 1];
    setRedoStack((prev) => prev.slice(0, -1));
    setUndoStack((prev) => [...prev, nextAction]);

    if (nextAction.type === "drawing") {
      setDrawings((prev) => [...prev, nextAction.drawing]);
    } else if (nextAction.type === "textbox") {
      setBoxes((prev) => [...prev, ...nextAction.boxes]);
    }
  };

  const clearAll = () => {
    setBoxes([]);
    setDrawings([]);
    setUndoStack([]);
    setRedoStack([]);
    setImage(originalImage);
    setCurrentPath([]);
    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, canvasSize.width, canvasSize.height);
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-4">
      {/* Toolbar */}
      <div className="mb-4 flex items-center gap-4">
        <input
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          className="hidden"
          id="image-input"
        />
        <label
          htmlFor="image-input"
          className="flex items-center justify-center px-4 py-2 bg-blue-500 text-white rounded-lg cursor-pointer hover:bg-blue-600"
        >
          <Upload className="w-4 h-4 mr-2" />
          <span>Upload Image</span>
        </label>

        <div className="flex gap-2 bg-gray-100 p-1 rounded-lg">
          {tools.map(({ id, icon: Icon, name }) => (
            <button
              key={id}
              onClick={() => setTool(id)}
              className={`p-2 rounded ${
                tool === id ? "bg-white shadow" : "hover:bg-white/50"
              }`}
              title={name}
            >
              <Icon className="w-5 h-5" />
            </button>
          ))}
        </div>

        <div className="flex gap-2">
          <button
            onClick={undo}
            disabled={undoStack.length === 0}
            className="p-2 rounded hover:bg-gray-100 disabled:opacity-50"
            title="Undo"
          >
            <Undo className="w-5 h-5" />
          </button>
          <button
            onClick={redo}
            disabled={redoStack.length === 0}
            className="p-2 rounded hover:bg-gray-100 disabled:opacity-50"
            title="Redo"
          >
            <Redo className="w-5 h-5" />
          </button>
        </div>

        {(boxes.length > 0 || drawings.length > 0) && (
          <button
            onClick={clearAll}
            className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600"
          >
            Clear All
          </button>
        )}
      </div>

      {error && (
        <div className="mb-4 p-4 bg-red-100 text-red-700 rounded-lg">
          {error}
        </div>
      )}

      <div
        className="relative rounded-lg overflow-hidden bg-gray-50 border"
        ref={containerRef}
      >
        {image ? (
          <div
            className={`relative ${
              tool === "textdetect"
                ? "cursor-crosshair"
                : tool === "draw"
                ? "cursor-pencil"
                : "cursor-default"
            }`}
          >
            <img
              ref={imageRef}
              src={image}
              alt="Edited"
              className="w-full h-auto"
              onClick={handleClick}
            />

            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0"
              style={{
                width: canvasSize.width,
                height: canvasSize.height,
                pointerEvents: tool === "textdetect" ? "none" : "auto",
              }}
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
            />

            {/* SVG overlay for boxes */}
            <svg
              className="absolute top-0 left-0 pointer-events-none"
              style={{
                width: "100%",
                height: "100%",
              }}
              viewBox={`0 0 ${canvasSize.width} ${canvasSize.height}`}
              preserveAspectRatio="none"
            >
              {boxes.map((box, index) => {
                // Validate box dimensions
                const isValid =
                  box &&
                  typeof box.x === "number" &&
                  typeof box.y === "number" &&
                  typeof box.width === "number" &&
                  typeof box.height === "number" &&
                  !isNaN(box.x) &&
                  !isNaN(box.y) &&
                  !isNaN(box.width) &&
                  !isNaN(box.height);

                if (!isValid) {
                  console.warn("Invalid box data:", box);
                  return null;
                }

                return (
                  <rect
                    key={`box-${index}`}
                    x={box.x}
                    y={box.y}
                    width={box.width}
                    height={box.height}
                    fill="none"
                    stroke="red"
                    strokeWidth="2"
                    vectorEffect="non-scaling-stroke"
                  />
                );
              })}
            </svg>
          </div>
        ) : (
          <div className="h-96 flex items-center justify-center text-gray-400">
            Upload an image to start editing
          </div>
        )}

        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 text-white">
            Processing...
          </div>
        )}
      </div>

      {(boxes.length > 0 || drawings.length > 0) && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h3 className="font-semibold mb-2">Editor Status</h3>
          <div className="text-sm text-gray-600">
            <p>Detected Text Boxes: {boxes.length}</p>
            <p>Drawings: {drawings.length}</p>
            <p className="mt-2">
              {tool === "textdetect"
                ? "Click on text areas to detect boxes"
                : tool === "draw"
                ? "Click and drag to draw"
                : `Current tool: ${tools.find((t) => t.id === tool)?.name}`}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageEditor;
