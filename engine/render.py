
import cv2
import numpy as np
from typing import List, Tuple, Generator, Optional
from .vectorize import Path
from .path_sampler import PathSampler
from .hand import HandAsset

class SketchRenderer:
    """
    Renders sketch animations using vector paths and hand overlay.
    """
    def __init__(self, width: int, height: int, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
        
    def render(self, 
               paths: List[Path], 
               hand_asset: Optional[HandAsset], 
               duration_sec: float,
               stroke_width: int = 2,
               stroke_color: Tuple[int, int, int] = (0, 0, 0)) -> Generator[np.ndarray, None, None]:
        """
        Yields frames for the animation.
        """
        if not paths:
            return

        # 1. Flatten all paths into a single sequence of points for linear time progression
        # We need to calculate total length to distribute time correctly
        
        # Structure: List of (x, y, is_drawing)
        all_movements = []
        
        last_pos = None
        
        for path in paths:
            # Sample path points
            sampled = PathSampler.sample_path(path.points, step_size=2.0)
            if not sampled:
                continue
                
            start_pos = sampled[0]
            
            # If we have a previous position, add travel move
            if last_pos is not None:
                # Travel from last_pos to start_pos
                # We want travel to be fast but visible.
                # Travel speed could be faster than drawing speed.
                # Let's say travel step size is 10.0 (5x faster)
                travel_steps = int(np.linalg.norm(np.array(start_pos) - np.array(last_pos)) / 10.0)
                if travel_steps > 0:
                    travel_pts = np.linspace(last_pos, start_pos, travel_steps, endpoint=False).astype(int)
                    for tp in travel_pts:
                        all_movements.append((tuple(tp), False))
            
            # Add drawing points
            for p in sampled:
                all_movements.append((p, True))
                
            last_pos = sampled[-1]

        # Calculate total steps
        total_steps = len(all_movements)
        if total_steps == 0:
            return

        # Calculate steps per frame
        total_frames = int(duration_sec * self.fps)
        if total_frames < 1:
            total_frames = 1
            
        steps_per_frame = total_steps / total_frames
        
        current_step_idx = 0
        steps_accumulator = 0.0
        
        # Current drawing state
        self.canvas.fill(255)
        
        # Track last drawing position for line continuity
        last_draw_pos = None
        
        # Current hand position
        current_hand_pos = all_movements[0][0] if all_movements else (0,0)
        
        for frame_idx in range(total_frames):
            # Calculate how many steps to advance
            steps_accumulator += steps_per_frame
            steps_to_do = int(steps_accumulator)
            steps_accumulator -= steps_to_do
            
            for _ in range(steps_to_do):
                if current_step_idx >= total_steps:
                    break
                    
                pt, is_drawing = all_movements[current_step_idx]
                current_hand_pos = pt
                
                if is_drawing:
                    if last_draw_pos is not None:
                        # Draw line from previous drawn point
                        # Check distance to avoid connecting separate paths that happen to be adjacent in list 
                        # (though logic above handles travel, we need to reset last_draw_pos on travel)
                        
                        # Actually, we need to know if we just came from a travel.
                        # Check previous step?
                        
                        if current_step_idx > 0 and all_movements[current_step_idx-1][1]:
                             cv2.line(self.canvas, last_draw_pos, pt, stroke_color, stroke_width, cv2.LINE_AA)
                        else:
                             # Start of new stroke
                             cv2.circle(self.canvas, pt, stroke_width // 2, stroke_color, -1)
                             
                    last_draw_pos = pt
                else:
                    # Not drawing (travel)
                    # Reset last_draw_pos so we don't connect across travel
                    pass
                    
                current_step_idx += 1
                
            # Create frame copy
            frame = self.canvas.copy()
            
            # Overlay hand
            if hand_asset:
                frame = hand_asset.overlay(frame, current_hand_pos[0], current_hand_pos[1])
                
            yield frame

