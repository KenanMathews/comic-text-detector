// src/App.jsx
import React from 'react';
import ImageEditor from './components/ImageEditor';

function App() {
  return (
    <div className="min-h-screen bg-gray-100 py-8">
      <div className="container mx-auto">
        <h1 className="text-3xl font-bold text-center mb-8">
          Comic Text Detector
        </h1>
        <ImageEditor />
      </div>
    </div>
  );
}

export default App;