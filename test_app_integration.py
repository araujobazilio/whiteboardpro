import os
import cv2
import numpy as np
import sys

# Add current directory to path so we can import app_licensed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app_licensed import generate_sketch_video

def create_test_image(filename="test_app_input.png"):
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 255
    cv2.circle(img, (640, 360), 100, (0, 0, 0), 5)
    cv2.rectangle(img, (200, 200), (400, 400), (0,0,0), 5)
    cv2.imwrite(filename, img)
    return filename

def dummy_progress(progress, desc=""):
    print(f"[Progress] {progress*100:.1f}% - {desc}")

def main():
    print("Testing app integration...")
    img_path = create_test_image()
    
    # Call generate_sketch_video
    # Signature: image_path, split_len, frame_rate, skip_rate, end_duration, draw_mode, progress, sketch_duration, fill_duration
    print("Calling generate_sketch_video...")
    video_path, msg = generate_sketch_video(
        image_path=img_path,
        split_len=10, # Ignored
        frame_rate=30,
        skip_rate=1, # Ignored
        end_duration=2,
        draw_mode="Contornos + Colorização", # Validating the color/fade logic too
        progress=dummy_progress,
        sketch_duration_sec=5.0,
        fill_duration_sec=2.0
    )
    
    print(f"Result Message: {msg}")
    
    if video_path and os.path.exists(video_path):
        print(f"SUCCESS: Video generated at: {video_path}")
        # Verify file size
        size = os.path.getsize(video_path)
        print(f"Video size: {size/1024:.2f} KB")
    else:
        print("FAILURE: Failed to generate video.")

if __name__ == "__main__":
    main()
