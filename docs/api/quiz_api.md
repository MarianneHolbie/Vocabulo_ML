# Vocabulo Quiz API Documentation

## Overview

The Vocabulo Quiz API provides endpoints for generating word recommendations and checking the health status
of the service. This API is built using FastAPI and integrates with machine learning models to provide personalized 
learning experiences.

## Base URL

`http://localhost:8000`

## Endpoints

### 1. Get Recommendations

Generates word recommendations for a user.

- **URL**: `/get_recommendations`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "user_id": "string (UUID format)"
  }
  ```

- **Response**:
  ```json
  {
    "recommendations": [
      {
        "mot_id": "integer",
        "mot": "string",
        "category": "string",
        "subcategory": "string",
        "niv_diff_id": "integer",
        "url_sign": "string",
        "url_def": "string",
        "recommendation_score": "float"
      }
    ]
  }
  ```

- **Description**: This endpoint takes a user ID (in UUID format) and returns a list of recommended words based on 
the user's learning history and performance.

- **Error Responses**:
  - 400 Bad Request: If the user ID format is invalid
  - 404 Not Found: If the user does not exist
  - 500 Internal Server Error: If there's an error generating recommendations

### 2. Health Check

Checks the health status of the API.

- **URL**: `/health`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "status": "healthy"
  }
  ```

- **Description**: This endpoint can be used to verify if the service is running and responding to requests.

## Error Handling

The API uses standard HTTP status codes to indicate the success or failure of requests. In case of errors, a JSON 
response with an error message is returned.

Example error response:
```json
{
  "detail": "Error message describing the problem"
}
```

## Authorization

Currently, the API does not implement authorization, because you should be logged before getting word recommendation.
It's recommended to add authorization with token in the future for security purposes.

## Data Models

### UserRequest

- `user_id` (string, required): The ID of the user making the request. Must be a valid UUID.

## Usage Examples

### Python

```python
import requests

url = "http://localhost:8000/get_recommendations"
data = {"user_id": "27f23ffc-a449-460b-a216-86b4dd41f6ef"}

response = requests.post(url, json=data)
print(response.json())
```

### cURL

```bash
curl -X POST "http://localhost:8000/get_recommendations" \
     -H "Content-Type: application/json" \
     -d '{"user_id": "27f23ffc-a449-460b-a216-86b4dd41f6ef"}'
```

## Logging

The API implements logging to track requests and potential issues. Logs are configured at the INFO level.

## Future Enhancements

1. Implement user authorization
2. Add endpoints for user progress tracking and quiz submission
3. Implement versioning for the API
4. Add more detailed error responses