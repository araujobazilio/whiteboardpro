
import numpy as np
from typing import List, Tuple

class PathSampler:
    """
    Handles sampling of paths into equidistant points for smooth rendering.
    """
    
    @staticmethod
    def sample_path(points: List[Tuple[int, int]], step_size: float = 2.0) -> List[Tuple[int, int]]:
        """
        Resamples a list of points (polyline) so that the distance between
        consecutive points is approximately step_size.
        """
        if len(points) < 2:
            return points
            
        sampled_points = [points[0]]
        current_idx = 0
        
        # Convert to numpy for easier vector math
        pts = np.array(points, dtype=np.float32)
        
        n_points = len(pts)
        if n_points < 2:
            return points

        # Accumulate distance
        dists = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        # Cumulative distance
        cum_dists = np.insert(np.cumsum(dists), 0, 0)
        total_len = cum_dists[-1]
        
        # Determine number of steps
        # Use simple interpolation
        num_steps = int(total_len / step_size)
        if num_steps <= 1:
             return points # return original if too short for sampling

        # New distances we want
        new_dists = np.linspace(0, total_len, num_steps)
        
        # Interpolate X and Y separately
        new_xs = np.interp(new_dists, cum_dists, pts[:, 0])
        new_ys = np.interp(new_dists, cum_dists, pts[:, 1])
        
        # Combine back
        resampled = np.column_stack((new_xs, new_ys)).astype(int)
        
        return [tuple(p) for p in resampled]

