import os
import requests

# Flask 服务器的 URL
BASE_URL = 'http://localhost:2790'

# 测试文件路径
TEST_MODEL_FILE = '1.0.0.onnx'
TEST_TRAJ_FILE = '1.0.0.0.csv'

# 创建测试文件
def create_test_files():
    with open(TEST_MODEL_FILE, 'wb') as f:
        f.write(b'This is a test model file for upload and download testing.')

    with open(TEST_TRAJ_FILE, 'w') as f:
        f.write('column1,column2\nvalue1,value2\n')

# 测试上传模型文件
def test_upload_model():
    url = f"{BASE_URL}/upload/model"
    with open(TEST_MODEL_FILE, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    print(f"Upload model response: {response.status_code} - {response.json()}")

# 测试上传轨迹文件
def test_upload_traj():
    url = f"{BASE_URL}/upload/traj"
    with open(TEST_TRAJ_FILE, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    print(f"Upload trajectory response: {response.status_code} - {response.json()}")

# 测试下载模型文件
def test_download_model(version):
    url = f"{BASE_URL}/download/model/{version}"
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(f'downloaded_{TEST_MODEL_FILE}', 'wb') as f:
            f.write(response.content)
        print(f"Model file downloaded successfully: downloaded_{TEST_MODEL_FILE}")
    else:
        # 先打印响应内容以调试
        print(f"Download model failed: {response.status_code} - {response.text}")

# 测试下载轨迹文件
def test_download_traj(version):
    url = f"{BASE_URL}/download/traj/{version}"
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(f'downloaded_{TEST_TRAJ_FILE}', 'wb') as f:
            f.write(response.content)
        print(f"Trajectory file downloaded successfully: downloaded_{TEST_TRAJ_FILE}")
    else:
        # 先打印响应内容以调试
        print(f"Download trajectory failed: {response.status_code} - {response.text}")


# 清理测试文件
def clean_up():
    if os.path.exists(TEST_MODEL_FILE):
        os.remove(TEST_MODEL_FILE)
    if os.path.exists(TEST_TRAJ_FILE):
        os.remove(TEST_TRAJ_FILE)
    
    downloaded_model_file = f"downloaded_{TEST_MODEL_FILE}"
    downloaded_traj_file = f"downloaded_{TEST_TRAJ_FILE}"
    
    if os.path.exists(downloaded_model_file):
        os.remove(downloaded_model_file)
    if os.path.exists(downloaded_traj_file):
        os.remove(downloaded_traj_file)

if __name__ == '__main__':
    # # 创建测试文件
    # create_test_files()

    # # 测试上传模型文件
    # test_upload_model()

    # # 测试上传轨迹文件
    # test_upload_traj()

    # 测试下载模型文件
    test_download_model('1.0.0')

    # 测试下载轨迹文件
    test_download_traj('1.0.0.0')

    # 清理测试文件
    # clean_up()
