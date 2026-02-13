from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class Quality(Enum):
    SD = "SD"
    HD = "HD" # 1920x1080

class HandStyle(Enum):
    # Mapping to internal assets will be done in engine/hand.py
    DEFAULT = "default"
    MALE = "male"
    FEMALE = "female"
    ROBOT = "robot"
    # Add more as needed

class SketchColorMode(Enum):
    BW = "bw"
    COLOR = "color"
    GREY = "grey"
    NEON_BLUE = "neon_blue"

class SequenceMode(Enum):
    AUTO = "auto"
    VERTICAL = "vertical"
    TEXT_FIRST = "text_first"
    TEXT_LAST = "text_last"

@dataclass
class ProjectSettings:
    # Speedpaint Parity Options
    preset: str = "presentation-16-9"
    fps: int = 30 # 30, 60, 120
    sketch_duration_sec: float = 12.0
    fill_duration_sec: float = 6.0
    fade_out_sec: float = 0.8
    quality: Quality = Quality.HD
    background_color: str = "#FFFFFF"
    
    hand_style: HandStyle = HandStyle.DEFAULT
    sketch_color_mode: SketchColorMode = SketchColorMode.BW
    sketch_detail_level: float = 0.65 # 0.0 to 1.0 (potrace threshold/opt)
    
    sequence_mode: SequenceMode = SequenceMode.AUTO
    
    # Internal / Technical
    seed: int = 12345
    width: int = 1920
    height: int = 1080
    
    def get_total_duration(self) -> float:
        return self.sketch_duration_sec + self.fill_duration_sec + self.fade_out_sec

    def get_total_frames(self) -> int:
        return int(self.get_total_duration() * self.fps)
