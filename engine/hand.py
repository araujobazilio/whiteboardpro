
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import os

@dataclass
class HandConfig:
    name: str
    image_path: str
    mask_path: str
    tip_offset: Tuple[int, int] = (0, 0) # (x, y) relative to top-left of the cropped hand

class HandAsset:
    """
    Manages loading, preprocessing, and overlaying of hand assets.
    """
    def __init__(self, config: HandConfig):
        self.config = config
        self.image = None
        self.mask = None
        self.mask_inv = None
        self.height = 0
        self.width = 0
        self._loaded = False
        
        self.load()

    def load(self):
        """Loads and preprocesses the hand image and mask."""
        if not os.path.exists(self.config.image_path):
            # Fallback or error? For now, print warning and create dummy
            print(f"Warning: Hand image not found at {self.config.image_path}")
            return

        # Load safely (using standard cv2 for now, assuming path is valid or handled)
        # TODO: Use imread_safe logic if needed for unicode paths
        stream = open(self.config.image_path, "rb")
        bytes_data = bytearray(stream.read())
        numpy_array = np.asarray(bytes_data, dtype=np.uint8)
        self.image = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
        stream.close()
        
        if self.image is None:
            return

        # Handle RGBA
        if self.image.shape[2] == 4:
            # We could use alpha as mask, but for now let's just drop it 
            # as we expect a separate mask file in the config normally.
            self.image = self.image[:, :, :3]

        # Load mask
        if os.path.exists(self.config.mask_path):
            stream = open(self.config.mask_path, "rb")
            bytes_data = bytearray(stream.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            mask_orig = cv2.imdecode(numpy_array, cv2.IMREAD_GRAYSCALE)
            stream.close()
        else:
            # Create dummy mask if missing
            mask_orig = np.ones((self.image.shape[0], self.image.shape[1]), dtype=np.uint8) * 255

        # Preprocess (crop to content) using the mask
        # Find bounding box of white pixels in mask
        coords = cv2.findNonZero(mask_orig)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Crop
            self.image = self.image[y:y+h, x:x+w]
            mask_orig = mask_orig[y:y+h, x:x+w]
        
        # Prepare masks for alpha blending
        # Normalize to 0-1 float for multiplication
        self.mask = mask_orig.astype(float) / 255.0
        self.mask_inv = 1.0 - self.mask
        
        # Clean background of hand image (make sure masked area is black)
        for c in range(3):
            self.image[:, :, c] = self.image[:, :, c] * (self.mask > 0)
            
        self.height, self.width = self.image.shape[:2]
        self._loaded = True

    def overlay(self, canvas: np.ndarray, target_x: int, target_y: int) -> np.ndarray:
        """
        Overlays the hand onto the canvas such that the pen tip 
        is at (target_x, target_y).
        """
        if not self._loaded:
            return canvas
            
        # Calculate top-left position based on tip offset
        # The hand image's (0,0) is placed at (x - offset_x, y - offset_y)
        # Wait, if tip_offset is (50, 50), it means the tip is at 50,50 in the hand image.
        # So to place tip at target_x, we compute top_left_x = target_x - 50.
        
        tl_x = target_x - self.config.tip_offset[0]
        tl_y = target_y - self.config.tip_offset[1]
        
        # Canvas dimensions
        h_canvas, w_canvas = canvas.shape[:2]
        
        # Calculate overlap region
        # Intersection of [tl_x, tl_x + self.width] and [0, w_canvas]
        x_start = max(0, tl_x)
        y_start = max(0, tl_y)
        x_end = min(w_canvas, tl_x + self.width)
        y_end = min(h_canvas, tl_y + self.height)
        
        if x_start >= x_end or y_start >= y_end:
            return canvas
            
        # Calculate random hand slice indices
        # If tl_x < 0, we start slicing hand from local_x = -tl_x
        # If tl_x >= 0, local_x = 0
        local_x_start = max(0, -tl_x)
        local_y_start = max(0, -tl_y)
        local_x_end = local_x_start + (x_end - x_start)
        local_y_end = local_y_start + (y_end - y_start)
        
        # Get slices
        hand_slice = self.image[local_y_start:local_y_end, local_x_start:local_x_end]
        mask_inv_slice = self.mask_inv[local_y_start:local_y_end, local_x_start:local_x_end]
        
        # Apply overlay manually (faster than addWeighted for simple masking)
        # canvas_slice = canvas[y_start:y_end, x_start:x_end]
        # canvas_slice = canvas_slice * mask_inv_slice + hand_slice
        
        # We need to be careful with types. Canvas is uint8.
        roi = canvas[y_start:y_end, x_start:x_end].astype(float)
        
        # Broadcast mask if necessary (it is single channel)
        if len(mask_inv_slice.shape) == 2 and len(roi.shape) == 3:
             mask_inv_slice = np.stack([mask_inv_slice]*3, axis=-1)
             
        # Blend
        # ROI * MaskInv (clears area for hand) + Hand (adds hand)
        blended = roi * mask_inv_slice + hand_slice
        
        canvas[y_start:y_end, x_start:x_end] = blended.astype(np.uint8)
        
        return canvas

