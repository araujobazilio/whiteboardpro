
import cv2
import numpy as np
import os
from engine.vectorize import TraceEngine
from engine.render import SketchRenderer
from engine.hand import HandAsset, HandConfig

def create_test_image(filename="test_input.png"):
    """Creates a simple test image (circle and rectangle)"""
    img = np.ones((1080, 1920, 3), dtype=np.uint8) * 255
    
    # Draw a black circle
    cv2.circle(img, (960, 540), 200, (0, 0, 0), 10)
    
    # Draw a rectangle
    cv2.rectangle(img, (200, 200), (600, 500), (0,0,0), 10)
    
    # Draw some text-like lines
    cv2.line(img, (1300, 300), (1700, 300), (0,0,0), 5)
    cv2.line(img, (1300, 350), (1600, 350), (0,0,0), 5)
    
    cv2.imwrite(filename, img)
    print(f"Created test image: {filename}")
    return filename

def main():
    # 1. Setup
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    HAND_PATH = os.path.join(BASE_PATH, 'kivy', 'data', 'images', 'drawing-hand.png')
    MASK_PATH = os.path.join(BASE_PATH, 'kivy', 'data', 'images', 'hand-mask.png')
    
    input_file = create_test_image()
    
    # 2. Vectorize
    print("Vectorizing...")
    tracer = TraceEngine()
    paths = tracer.process_image(input_file)
    print(f"Found {len(paths)} paths.")
    
    # 3. Load Hand
    print("Loading Hand...")
    hand_config = HandConfig(
        name="Default",
        image_path=HAND_PATH,
        mask_path=MASK_PATH,
        tip_offset=(0, 0) # Adjust this later if needed
    )
    hand = HandAsset(hand_config)
    
    # 4. Render
    print("Rendering...")
    renderer = SketchRenderer(width=1920, height=1080, fps=30)
    
    # Duration of 5 seconds
    frames = renderer.render(paths, hand, duration_sec=5.0)
    
    # Save video
    out_path = "test_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (1920, 1080))
    
    count = 0
    for frame in frames:
        out.write(frame)
        count += 1
        if count % 30 == 0:
            print(f"Rendered {count} frames...")
            
    out.release()
    print(f"Done! Saved to {out_path}")

if __name__ == "__main__":
    main()
