import cv2
import os
import re
import tqdm
from matplotlib import pyplot as plt
import numpy as np

def create_video_from_images(folder_path, output_file, frame_rate=30):
    """
    Reads images from a folder, sorts them by their IDs in the filename, and creates a video.
    
    Parameters:
    - folder_path (str): Path to the folder containing images.
    - output_file (str): Path to the output video file (e.g., 'output.mp4').
    - frame_rate (int): Frames per second for the video.
    """
    # Ensure the folder path exists
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    # List and sort images by numeric ID
    images = [f for f in os.listdir(folder_path) if re.match(r'^\d+\.png$', f)]
    images.sort(key=lambda x: int(re.match(r'^(\d+)\.png$', x).group(1)))

    if not images:
        print("No images found in the specified folder.")
        return

    # Read the first image to get frame dimensions
    first_image_path = os.path.join(folder_path, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error reading image: {first_image_path}")
        return
    height, width, _ = frame.shape

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

    pbar = tqdm.tqdm(images)
    # Add frames to video
    for image_name in pbar:
        image_path = os.path.join(folder_path, image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Skipping invalid image: {image_name}")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_file}")

def create_video_from_matplotlib(output_file, frame_rate=30, num_frames=100):
    """
    Creates a video by rendering frames using matplotlib.
    
    Parameters:
    - output_file (str): Path to the output video file (e.g., 'output.mp4').
    - frame_rate (int): Frames per second for the video.
    - num_frames (int): Total number of frames to render.
    """
    # Define video writer
    width, height = 640, 480  # Dimensions of the frames
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .mp4
    video_writer = cv2.VideoWriter(output_file, fourcc, frame_rate, (width, height))

    for i in range(num_frames):
        # Clear and create a new figure for each frame
        plt.figure(figsize=(width / 100, height / 100), dpi=100)
        
        # Example plot: sine wave with changing phase
        x = np.linspace(0, 2 * np.pi, 500)
        y = np.sin(x + (2 * np.pi * i / num_frames))
        plt.plot(x, y, label=f"Frame {i + 1}")
        plt.title("Sine Wave Animation")
        plt.legend()
        plt.grid()

        # Render the figure to an image
        plt.tight_layout()
        plt.draw()
        frame = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))

        # Resize to ensure compatibility with video writer
        frame = cv2.resize(frame, (width, height))

        # Convert RGB to BGR for OpenCV
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Close the figure to free memory
        plt.close()

    video_writer.release()
    print(f"Video saved to {output_file}")

if __name__ == "__main__":
    create_video_from_images("/home/idlab/code/qx-oracle/data_qx/draw/DSACTPI_241124-021048/11-25-09:58:51", "temp.avi")
    pass
