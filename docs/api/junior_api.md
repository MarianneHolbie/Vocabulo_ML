# Vocabulo Junior API Documentation

## Overview

The Vocabulo Junior API provides an endpoint for processing images, performing Optical Character Recognition (OCR), 
and extracting meaningful information from the text. This API is built using FastAPI and integrates various image and 
text processing modules.

## Base URL

`http://localhost:8000`

## Endpoints

### 1. Process Image

Processes an uploaded image file, performs OCR, and extracts meaningful information.

- **URL**: `/process-image/`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Request Body**:
  - `file`: The image file to be processed (File upload)

- **Response**:
  ```json
  {
    "original_text": "string",
    "processed_results": [
      {
        "sentence": "string",
        "words": [
          {
            "word": "string",
            "lemma": "string",
            "pos": "string",
            "function": "string",
            "definition": "string",
            "url": "string",
            "confidence": "float",
            "difficulty": "integer (optional)"
          }
        ]
      }
    ]
  }
  ```

- **Description**: This endpoint takes an uploaded image file, performs OCR to extract text, cleans and normalizes the 
text, and then processes it to extract meaningful information.

- **Error Responses**:
  - 400 Bad Request: If there's an issue with the image format or processing
  - 500 Internal Server Error: If there's an unexpected error during processing

## Image Processing Details

- Supported image formats: JPEG, PNG, HEIF
- HEIF (High Efficiency Image Format) images are automatically converted to a supported format
- The app includes a feature to add a line to the image, helping to ensure the text is straight for OCR processing

## Text Processing Steps

1. **OCR**: Utilizes a custom OCR model (based on Tesseract) to extract text from the image
2. **Text Cleaning**: Removes extra spaces, preserves specific expressions, and standardizes punctuation
3. **Text Normalization**: Converts text to NFC form and replaces specific character sequences
4. **Linguistic Processing**:
   - Lemmatization
   - Part-of-speech tagging
   - Grammatical function identification
   - Word definition retrieval
   - Difficulty level assessment

## Usage Example

### Python

```python
import requests

url = "http://localhost:8000/process-image/"
files = {"file": open("path/to/your/image.jpg", "rb")}

response = requests.post(url, files=files)
print(response.json())
```

### cURL

```bash
curl -X POST "http://localhost:8000/process-image/" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
```

## Dependencies

The API relies on several custom modules:
- `core.image_processing`: Handles image format conversions and validations
- `core.ocr`: Performs OCR on images using a custom Tesseract-based model
- `core.text_processing`: Cleans, normalizes, and processes text
- `models.nlp_models`: Provides NLP functionalities using CamemBERT and spaCy

## Performance Considerations

- Image processing and OCR can be resource-intensive. Consider implementing queuing or background processing for large 
images or high traffic.
- The API currently processes one image at a time. For batch processing, consider implementing a separate endpoint.

## Security Considerations

- Implement file type validation to ensure only image files are processed
- Consider adding authentication to secure the API
- Implement rate limiting to prevent abuse

## Future Enhancements

1. Integrate with a content moderation service for uploaded images
2. Provide more detailed error messages and processing status updates
3.Implement endpoint versioning for future updates