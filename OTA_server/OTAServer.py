from flask import Flask, request, send_from_directory, jsonify
import os
import json
import shutil
from werkzeug.utils import secure_filename
import argparse


"""
Flask Web Application for Model and Trajectory Management

This application provides an interface for uploading, downloading, and managing model and trajectory files. 
It defines endpoints for retrieving version information, listing available files, and handling file uploads 
and downloads.

Key Features:
1. **File Structure**:
   - Model files follow the naming convention: `x.x.x.onnx`
   - Trajectory files follow the naming convention: `x.x.x.idx.csv`

2. **Directory Setup**:
   - The application organizes files into designated directories for models, trajectories, uploads, downloads, and version info.
   - Required directories are created if they do not already exist.

3. **Endpoints**:
   - **Info Endpoints**:
     - `/info/mdlversion`: Retrieve the current model version.
     - `/info/trajversion`: Retrieve the current trajectory version.
     - `/info/mdllist`: List all available model files.
     - `/info/trajlist`: List all available trajectory files.

   - **Download Endpoints**:
     - `/download/model/<version>`: Download a specified model version file.
     - `/download/traj/<version>`: Download a specified trajectory file.

   - **Upload Endpoint**:
     - `/upload/<type>`: Upload either a model or a trajectory file. Validates the filename format and saves the file to the appropriate directory.

4. **File Management**:
   - Functions for writing and reading version information to and from files.
   - Validations for file naming formats to ensure that uploads meet expected conventions.

5. **Error Handling**:
   - Returns appropriate JSON responses for errors such as missing files, invalid formats, or server issues.
"""

# Prepare
DIR_CURRENT = os.path.dirname(os.path.abspath(__file__))
DIR_CURRENT = os.path.join(DIR_CURRENT, "data")

DIR_MODELS = os.path.join(DIR_CURRENT, "models")
DIR_TRAJS = os.path.join(DIR_CURRENT, "trajs")

DIR_UPLOAD = os.path.join(DIR_CURRENT, "upload")
DIR_DOWNLOAD = os.path.join(DIR_CURRENT, "download")
DIR_INFO = os.path.join(DIR_CURRENT, "info")

DIRS_DATA = [DIR_MODELS, DIR_TRAJS, DIR_UPLOAD, DIR_DOWNLOAD, DIR_INFO]

FILE_MDLVERSION = "mdl_version.txt"
FILE_TRAJVERSION = "traj_version.txt"
FILES_INFO = [FILE_MDLVERSION, FILE_TRAJVERSION]

def write_info(file, info):
    """Write information to a file."""
    fp = os.path.join(DIR_INFO, file)
    with open(fp, "wt") as f:
        f.write(info)

def read_info(file) -> str:
    """Read information from a file."""
    fp = os.path.join(DIR_INFO, file)
    if os.path.exists(fp):
        with open(fp, "r") as f:
            return f.read()
    else:
        raise FileNotFoundError(f"No version info for {file}.")

# Create directories if they do not exist
for dir in DIRS_DATA:
    os.makedirs(dir, exist_ok=True)

# Uncomment and implement version management as needed
# for file in FILES_INFO:
#     last_version = read_info(file)
#     cur_version = f"{int(last_version.split('.')[0]) + 1}.0.0"
#     write_info(file, cur_version)

app = Flask(__name__)

"""
# Info endpoints
"""

@app.route('/info/mdlversion', methods=['GET'])
def get_model_version():
    """Get the model version file."""
    return get_info_file(FILE_MDLVERSION)

@app.route('/info/trajversion', methods=['GET'])
def get_traj_version():
    """Get the trajectory version file."""
    return get_info_file(FILE_TRAJVERSION)

def get_info_file(file_name):
    """Helper function to send info files."""
    try:
        return send_from_directory(DIR_INFO, file_name, as_attachment=False)
    except FileNotFoundError:
        return jsonify({'error': f'{file_name} not found'}), 404

def list_csv_files(directory):
    """List all CSV files in the specified directory."""
    return [file for file in os.listdir(directory) if file.endswith('.csv')]

@app.route('/info/mdllist', methods=['GET'])
def get_model_list():
    """Get the list of model version files (CSV) available."""
    model_files = list_csv_files(DIR_MODELS)
    return jsonify({'model_files': model_files}), 200

@app.route('/info/trajlist', methods=['GET'])
def get_traj_list():
    """Get the list of trajectory version files (CSV) available."""
    traj_files = list_csv_files(DIR_TRAJS)
    return jsonify({'traj_files': traj_files}), 200

"""
# Download endpoints
"""

@app.route('/download/model/<version>', methods=['GET'])
def download_model(version):
    """Download the specified model version."""
    return download_file(DIR_MODELS, f"{version}.onnx")

@app.route('/download/traj/<version>', methods=['GET'])
def download_traj(version):
    """Download the specified trajectory by version."""
    return download_file(DIR_TRAJS, f"{version}.csv")

def download_file(dir, file):
    """Helper function to handle file downloads."""
    try:
        return send_from_directory(dir, file, as_attachment=True)
    except FileNotFoundError:
        return jsonify({'error': f'File not found'}), 404

"""
# Upload endpoints
"""

import re

def is_valid_model_filename(filename):
    """Check if the model filename matches the format x.x.x.onnx."""
    model_pattern = r'^\d+\.\d+\.\d+\.onnx$'  # Format: x.x.x.onnx
    return re.match(model_pattern, filename) is not None

def is_valid_traj_filename(filename):
    """Check if the trajectory filename matches the format x.x.x.x.csv."""
    traj_pattern = r'^\d+\.\d+\.\d+\.\d\.csv$'  # Format: x.x.x.x.csv
    return re.match(traj_pattern, filename) is not None

@app.route('/upload/<type>', methods=['POST', 'PUT'])
def upload_file(type):
    """Upload a model or trajectory file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        print(f"Received file: {filename}")

        # Determine directory based on type
        try:
            if type == 'model':
                if not is_valid_model_filename(filename):
                    raise ValueError(f"Invalid format for {filename}")
                upload_dir = DIR_MODELS
                version = filename[0:-5]
                write_info(FILE_MDLVERSION, version)
            elif type == "traj":
                if not is_valid_traj_filename(filename):
                    raise ValueError(f"Invalid format for {filename}")
                upload_dir = DIR_TRAJS
                version = filename[0:-4]
                write_info(FILE_MDLVERSION, version)
            else:
                raise ValueError(f"Invalid type for {type}")
        except Exception as e:
            return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500
        
        filepath = os.path.join(upload_dir, filename)

        try:
            file.save(filepath)
            return jsonify({'success': f'File {filename} uploaded successfully'}), 200
        except Exception as e:
            return jsonify({'error': f'Internal Server Error: {str(e)}'}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=5000, type=int, help="Server listen port")
    args = parser.parse_args()
    app.run(debug=False, port=2790)
