Here’s an API documentation for the Flask application you provided. This documentation covers the endpoints, their methods, parameters, and expected responses.

# API Documentation for Model and Trajectory Management

## Base URL
```
http://<server_host>:2790
```

## Endpoints

### 1. Get Model Version
- **Endpoint**: `/info/mdlversion`
- **Method**: `GET`
- **Description**: Retrieves the current model version.
- **Response**:
    - **Status Code**: `200 OK`
    - **Content**:
        ```json
        {
            "version": "x.x.x"
        }
        ```
    - **Example**:
        ```json
        {
            "version": "1.0.0"
        }
        ```

### 2. Get Trajectory Version
- **Endpoint**: `/info/trajversion`
- **Method**: `GET`
- **Description**: Retrieves the current trajectory version.
- **Response**:
    - **Status Code**: `200 OK`
    - **Content**:
        ```json
        {
            "version": "x.x.x"
        }
        ```
    - **Example**:
        ```json
        {
            "version": "1.0.0"
        }
        ```

### 3. List Model Files
- **Endpoint**: `/info/mdllist`
- **Method**: `GET`
- **Description**: Lists all available model version files in CSV format.
- **Response**:
    - **Status Code**: `200 OK`
    - **Content**:
        ```json
        {
            "model_files": ["model_v1.0.0.onnx", "model_v1.0.1.onnx"]
        }
        ```
    - **Example**:
        ```json
        {
            "model_files": ["1.0.0.onnx", "1.0.1.onnx"]
        }
        ```

### 4. List Trajectory Files
- **Endpoint**: `/info/trajlist`
- **Method**: `GET`
- **Description**: Lists all available trajectory files in CSV format.
- **Response**:
    - **Status Code**: `200 OK`
    - **Content**:
        ```json
        {
            "traj_files": ["traj_v1.0.0.idx.csv", "traj_v1.0.1.idx.csv"]
        }
        ```
    - **Example**:
        ```json
        {
            "traj_files": ["1.0.0.idx.csv", "1.0.1.idx.csv"]
        }
        ```

### 5. Download Model
- **Endpoint**: `/download/model/<version>`
- **Method**: `GET`
- **Description**: Downloads the specified model version file.
- **Parameters**:
    - **Path Parameter**: 
        - `version` (string): The version of the model to download (format: `x.x.x`).
- **Response**:
    - **Status Code**: 
        - `200 OK`: File is successfully downloaded.
        - `404 Not Found`: The specified model version file does not exist.
    - **Example**:
        - Successful Response:
            - Content-Type: application/octet-stream
        - Error Response:
            ```json
            {
                "error": "File not found"
            }
            ```

### 6. Download Trajectory
- **Endpoint**: `/download/traj/<version>`
- **Method**: `GET`
- **Description**: Downloads the specified trajectory file.
- **Parameters**:
    - **Path Parameter**: 
        - `version` (string): The version of the trajectory to download (format: `x.x.x`).
- **Response**:
    - **Status Code**: 
        - `200 OK`: File is successfully downloaded.
        - `404 Not Found`: The specified trajectory version file does not exist.
    - **Example**:
        - Successful Response:
            - Content-Type: application/octet-stream
        - Error Response:
            ```json
            {
                "error": "File not found"
            }
            ```

### 7. Upload File
- **Endpoint**: `/upload/<type>`
- **Method**: `POST` or `PUT`
- **Description**: Uploads a model or trajectory file.
- **Parameters**:
    - **Path Parameter**:
        - `type` (string): The type of file to upload (`model` or `traj`).
    - **Form Data**:
        - `file` (file): The file to upload.
- **Response**:
    - **Status Code**: 
        - `200 OK`: File uploaded successfully.
        - `400 Bad Request`: No file part in the request or no selected file.
        - `500 Internal Server Error`: Internal error occurred during processing.
    - **Example**:
        - Successful Response:
            ```json
            {
                "success": "File model_v1.0.0.onnx uploaded successfully"
            }
            ```
        - Error Response (e.g., missing file):
            ```json
            {
                "error": "No file part in the request"
            }
            ```

## Error Handling
The API returns standard HTTP status codes for error handling. The following codes may be encountered:
- **400 Bad Request**: Invalid request parameters or missing file.
- **404 Not Found**: Requested resource could not be found.
- **500 Internal Server Error**: An unexpected error occurred on the server.

## Notes
- Ensure that file names adhere to the specified naming conventions:
    - Model files: `x.x.x.onnx`
    - Trajectory files: `x.x.x.idx.csv`
- All responses are returned in JSON format, with error messages clearly stated.

## Conclusion
This API provides a structured approach to manage model and trajectory files, allowing for easy uploads, downloads, and retrieval of version information. For further assistance, contact the API provider or refer to the codebase for detailed implementation insights.