from typing import List
import fitz  # PyMuPDF
from PIL import Image
import os, cv2
import numpy as np
import tqdm

def pdf_to_image(pdf_path, zoom=0.3):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(0)  # 只处理第一页
    
    # 设置缩放比例
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def images_to_gif(images, gif_path, duration=500):
    if images:
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0  # 0 表示无限循环
        )

fps = 60

def images_to_mp4(images: List[Image.Image], gif_path, duration=0.3):
    assert images, "Empty."
    frame_width = int(images[0].width)
    frame_height = int(images[0].height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(gif_path, fourcc, fps, (frame_width, frame_height))
    duration = int(duration * fps)
    for frame in images:
        for i in range(duration):
            video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video.release()

def process_batch(img_folder, name="result", data_attr=".png", ret_type="mp4"):
    # 获取所有文件名，并按数字排序
    img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(data_attr)]
    
    # 将文件名转换为整数进行排序
    img_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    images= []
    pbar = tqdm.tqdm(img_files)
    for file_name in pbar:
        if file_name.endswith(".pdf"):
            img_path = os.path.join(img_folder, file_name)
            try:
                img = pdf_to_image(img_path, zoom=1)  # 更小的缩放比例
                images.append(img)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        elif file_name.endswith(".png"):
            img_path = os.path.join(img_folder, file_name)
            try:
                img = Image.open(img_path)
                images.append(img)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
            
    
    if images:
        print("Writing...")
        gif_path = os.path.join(img_folder, name + '.gif')
        mp4_path = os.path.join(img_folder, name + '.mp4')
        images_to_mp4(images, mp4_path)
        # images_to_gif(images, gif_path)
        print(f"Created GIF: {gif_path}")
        print(f"Created MP4: {mp4_path}")

# 示例使用
if __name__ == "__main__":
    qx_folder = '/root/qianxing/gops-grpc/draw_qianxing'  # PDF文件夹路径
    tar_dirs = [
        # "09-05-11:32:41",
        # "09-01-20:51:48",
        # "09-01-20:53:32",
        # "09-05-20:02:11",
        # "09-05-19:24:57",
        # "09-06-19:32:24",
        # "09-06-19:29:54",
        "09-07-14:57:21"
    ]
    for dir in tar_dirs:
        print(f"process {dir}")
        process_batch(os.path.join(qx_folder, dir))
