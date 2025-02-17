// src/services/api.js
import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const detectText = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await axios.post(
    `${API_URL}/api/detect`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return response.data;
};