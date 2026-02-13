
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class Path:
    points: List[Tuple[int, int]] # List of (x, y) tuples
    is_closed: bool = False
    area: float = 0.0

class TraceEngine:
    """
    Handles conversion of raster images to vector-like paths using OpenCV.
    """
    
    def __init__(self, high_contrast_threshold: int = 200, min_path_len: int = 10):
        self.threshold = high_contrast_threshold
        self.min_path_len = min_path_len

    def process_image(self, image_path: str, max_dimension: int = 1920) -> List[Path]:
        """
        Loads an image, processes it, and returns a list of paths.
        """
        # Load image safely (assuming imread_safe is available or handled externally, 
        # but here we use standard cv2 for the engine module, assuming valid input)
        # Note: In a real scenario, we'd pass the numpy array directly to decoupling IO.
        # For now, we load it.
        img = cv2.imread(image_path)
        if img is None:
            # Try handling non-ascii paths if needed, or raise error
            # For the engine, we stick to standard logic. 
            # Ideally receive the image array, not path.
            # But adhering to the interface:
            stream = open(image_path, "rb")
            bytes_data = bytearray(stream.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
            stream.close()
            
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Resize if necessary
        h, w = img.shape[:2]
        if h > max_dimension or w > max_dimension:
            scale = max_dimension / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Denoise (optional but recommended for "sketch" look)
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Thresholding (Adaptive is usually better for drawings)
        # Using adaptive boolean inverted so lines are white (255) and background black (0)
        # This is standard for findContours
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological operations to close small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find Contours
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        
        paths = []
        
        if contours:
            # Determine hierarchy (optional, for now we treat all as strokes)
            # Maybe later filter by hierarchy to avoid filling holes if not desired
            
            for i, cnt in enumerate(contours):
                # Approximate contour to reduce points (simplify)
                epsilon = 0.001 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                
                # Check path length
                if cv2.arcLength(approx, True) < self.min_path_len:
                    continue
                
                # Convert to friendly format
                # approx has shape (N, 1, 2)
                points = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
                
                path = Path(
                    points=points,
                    is_closed=True, # contours are always closed in this mode
                    area=cv2.contourArea(cnt)
                )
                paths.append(path)
                
        # Sort paths? 
        # Basic heuristic: Top to bottom, left to right 
        # This will be handled by the specialized Sequence module later.
        # For now, just return valid paths.
        
        return paths

