#!/usr/bin/env python3
"""
Tesla Dashcam Telemetry Overlay

Dual-pipe pipeline: ffmpeg decodes -> Python composites overlay -> ffmpeg encodes.
Single pass, one frame in memory at a time.
"""

import argparse
import os
import subprocess
import sys
from collections import deque
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add the tesla-dashcam directory to path for imports
sys.path.insert(0, "/tmp/tesla-dashcam")
import sei_extractor
import dashcam_pb2

VIDEO_WIDTH = 2896
VIDEO_HEIGHT = 1876
OVERLAY_HEIGHT = 200
FPS = 36  # approximate

# Layout: [LEFT] [SPEED] [ACCEL_PEDAL] [AUTOPILOT] [CHART] [BRAKE] [RIGHT]
LEFT_ARROW_WIDTH = 80
SPEED_WIDTH = 180
ACCEL_PEDAL_WIDTH = 100
AUTOPILOT_WIDTH = 100
BRAKE_WIDTH = 80
RIGHT_ARROW_WIDTH = 80
CHART_WIDTH = VIDEO_WIDTH - LEFT_ARROW_WIDTH - SPEED_WIDTH - ACCEL_PEDAL_WIDTH - AUTOPILOT_WIDTH - BRAKE_WIDTH - RIGHT_ARROW_WIDTH

CHART_HISTORY_SECONDS = 5
CHART_HISTORY_FRAMES = int(CHART_HISTORY_SECONDS * FPS)

# Colors
BG_COLOR = (0, 0, 0, 180)  # semi-transparent black
ARROW_OFF_COLOR = (60, 60, 60, 255)
ARROW_ON_COLOR = (255, 180, 0, 255)  # orange/amber for blinkers
BRAKE_OFF_COLOR = (60, 60, 60, 255)
BRAKE_ON_COLOR = (255, 0, 0, 255)  # red for brake
AUTOPILOT_OFF_COLOR = (60, 60, 60, 255)
AUTOPILOT_ON_COLOR = (0, 120, 255, 255)  # blue for autopilot/self-driving
TEXT_COLOR = (255, 255, 255, 255)
TEXT_DIM_COLOR = (150, 150, 150, 255)
ACCEL_BAR_COLOR = (0, 200, 100, 255)  # green for accelerator

# Autopilot state names
AUTOPILOT_STATES = {
    0: "OFF",
    1: "FSD",  # SELF_DRIVING
    2: "AUTO",  # AUTOSTEER
    3: "TACC",  # Traffic-Aware Cruise Control
}

# Try to load font
try:
    FONT_LARGE = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
    FONT_MEDIUM = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
    FONT_SMALL = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
except:
    FONT_LARGE = ImageFont.load_default()
    FONT_MEDIUM = ImageFont.load_default()
    FONT_SMALL = ImageFont.load_default()


def extract_all_sei_metadata(video_path: str) -> list:
    """Extract all SEI metadata from the video file."""
    metadata = []
    with open(video_path, "rb") as fp:
        offset, size = sei_extractor.find_mdat(fp)
        for meta in sei_extractor.iter_sei_messages(fp, offset, size):
            metadata.append(meta)
    return metadata


def create_arrow_image(width: int, height: int, direction: str, on: bool) -> Image.Image:
    """Create an arrow indicator image."""
    img = Image.new("RGBA", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    color = ARROW_ON_COLOR if on else ARROW_OFF_COLOR

    # Draw arrow in center
    cx, cy = width // 2, height // 2
    arrow_size = min(width, height) * 0.6

    if direction == "left":
        points = [
            (cx - arrow_size * 0.4, cy),
            (cx + arrow_size * 0.3, cy - arrow_size * 0.35),
            (cx + arrow_size * 0.3, cy + arrow_size * 0.35),
        ]
    else:
        points = [
            (cx + arrow_size * 0.4, cy),
            (cx - arrow_size * 0.3, cy - arrow_size * 0.35),
            (cx - arrow_size * 0.3, cy + arrow_size * 0.35),
        ]

    draw.polygon(points, fill=color)
    return img


def create_brake_image(width: int, height: int, on: bool) -> Image.Image:
    """Create a brake indicator image (circle with B)."""
    img = Image.new("RGBA", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    color = BRAKE_ON_COLOR if on else BRAKE_OFF_COLOR
    cx, cy = width // 2, height // 2
    radius = min(width, height) * 0.3

    bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.ellipse(bbox, fill=color)

    text_color = (255, 255, 255, 255) if on else (120, 120, 120, 255)
    draw.text((cx, cy), "B", fill=text_color, font=FONT_MEDIUM, anchor="mm")

    return img


def create_autopilot_image(width: int, height: int, state: int) -> Image.Image:
    """Create an autopilot indicator image."""
    img = Image.new("RGBA", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    is_active = state > 0
    color = AUTOPILOT_ON_COLOR if is_active else AUTOPILOT_OFF_COLOR
    text_color = TEXT_COLOR if is_active else TEXT_DIM_COLOR

    cx, cy = width // 2, height // 2

    # Draw steering wheel icon (simplified)
    radius = min(width, height) * 0.28
    draw.ellipse((cx - radius, cy - radius - 10, cx + radius, cy + radius - 10),
                 outline=color, width=4)

    # Draw state text below
    state_text = AUTOPILOT_STATES.get(state, "OFF")
    draw.text((cx, cy + 45), state_text, fill=text_color, font=FONT_SMALL, anchor="mm")

    return img


def create_speed_image(width: int, height: int, speed_mps: float) -> Image.Image:
    """Create a speed display image."""
    img = Image.new("RGBA", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Convert m/s to mph
    speed_mph = speed_mps * 2.23694
    cx, cy = width // 2, height // 2

    # Draw speed value
    draw.text((cx, cy - 15), f"{speed_mph:.0f}", fill=TEXT_COLOR, font=FONT_LARGE, anchor="mm")
    draw.text((cx, cy + 40), "MPH", fill=TEXT_DIM_COLOR, font=FONT_SMALL, anchor="mm")

    return img


def create_accel_pedal_image(width: int, height: int, pedal_position: float) -> Image.Image:
    """Create an accelerator pedal position bar."""
    img = Image.new("RGBA", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    cx = width // 2
    bar_width = 30
    bar_height = 140
    bar_top = 25
    bar_left = cx - bar_width // 2

    # Draw bar outline
    draw.rectangle((bar_left, bar_top, bar_left + bar_width, bar_top + bar_height),
                   outline=TEXT_DIM_COLOR, width=2)

    # Fill based on pedal position (0.0 to 1.0)
    fill_height = int(bar_height * min(1.0, max(0.0, pedal_position)))
    if fill_height > 0:
        draw.rectangle(
            (bar_left + 2, bar_top + bar_height - fill_height,
             bar_left + bar_width - 2, bar_top + bar_height - 2),
            fill=ACCEL_BAR_COLOR
        )

    # Label
    draw.text((cx, bar_top + bar_height + 15), "ACC", fill=TEXT_DIM_COLOR, font=FONT_SMALL, anchor="mm")

    return img


def cache_indicator_images() -> Dict[Tuple[bool, bool, bool], Tuple[Image.Image, Image.Image, Image.Image]]:
    """Pre-cache all 8 possible blinker/brake indicator combinations."""
    cache = {}
    for left_on in [False, True]:
        for right_on in [False, True]:
            for brake_on in [False, True]:
                left_img = create_arrow_image(LEFT_ARROW_WIDTH, OVERLAY_HEIGHT, "left", left_on)
                right_img = create_arrow_image(RIGHT_ARROW_WIDTH, OVERLAY_HEIGHT, "right", right_on)
                brake_img = create_brake_image(BRAKE_WIDTH, OVERLAY_HEIGHT, brake_on)
                cache[(left_on, right_on, brake_on)] = (left_img, brake_img, right_img)
    return cache


def cache_autopilot_images() -> Dict[int, Image.Image]:
    """Pre-cache autopilot state images."""
    cache = {}
    for state in range(4):
        cache[state] = create_autopilot_image(AUTOPILOT_WIDTH, OVERLAY_HEIGHT, state)
    return cache


class AccelChart:
    """Rolling acceleration chart using matplotlib with blitting."""

    def __init__(self, width: int, height: int, history_frames: int):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.width = width
        self.height = height
        self.history_frames = history_frames

        self.accel_x = deque([0.0] * history_frames, maxlen=history_frames)
        self.accel_y = deque([0.0] * history_frames, maxlen=history_frames)
        self.accel_z = deque([0.0] * history_frames, maxlen=history_frames)

        dpi = 100
        fig_width = width / dpi
        fig_height = height / dpi

        self.fig, self.ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        self.fig.patch.set_alpha(0.0)

        self.ax.set_facecolor((0, 0, 0, 0.7))
        self.ax.set_xlim(0, history_frames)
        self.ax.set_ylim(-3, 3)

        self.ax.tick_params(colors='white', labelsize=8)
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.set_ylabel("Accel (m/s²)", color='white', fontsize=9)

        x_data = list(range(history_frames))
        # Brighter colors for visibility against dark background
        self.line_x, = self.ax.plot(x_data, list(self.accel_x), color='#FF6B6B', linewidth=1.5, label='Front/Back')
        self.line_y, = self.ax.plot(x_data, list(self.accel_y), color='#4ECDC4', linewidth=1.5, label='Left/Right')
        self.line_z, = self.ax.plot(x_data, list(self.accel_z), color='#FFE66D', linewidth=1.5, label='Up/Down')

        self.ax.legend(loc='upper left', fontsize=8, facecolor='black',
                      labelcolor='white', framealpha=0.5)

        self.fig.tight_layout(pad=0.5)

        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def update(self, accel_x: float, accel_y: float, accel_z: float) -> Image.Image:
        """Update with new acceleration values and return rendered image."""
        self.accel_x.append(accel_x)
        self.accel_y.append(accel_y)
        self.accel_z.append(accel_z)

        self.line_x.set_ydata(list(self.accel_x))
        self.line_y.set_ydata(list(self.accel_y))
        self.line_z.set_ydata(list(self.accel_z))

        self.fig.canvas.restore_region(self.background)
        self.ax.draw_artist(self.line_x)
        self.ax.draw_artist(self.line_y)
        self.ax.draw_artist(self.line_z)
        self.fig.canvas.blit(self.ax.bbox)

        buf = self.fig.canvas.buffer_rgba()
        img = Image.frombuffer("RGBA", self.fig.canvas.get_width_height(), buf, "raw", "RGBA", 0, 1)
        return img.copy()


def create_overlay_frame(
    chart_img: Image.Image,
    indicator_images: Tuple[Image.Image, Image.Image, Image.Image],
    speed_img: Image.Image,
    accel_pedal_img: Image.Image,
    autopilot_img: Image.Image
) -> np.ndarray:
    """Compose the full overlay bar."""
    left_img, brake_img, right_img = indicator_images

    overlay = Image.new("RGBA", (VIDEO_WIDTH, OVERLAY_HEIGHT), BG_COLOR)

    # Layout: [LEFT] [SPEED] [ACCEL_PEDAL] [AUTOPILOT] [CHART] [BRAKE] [RIGHT]
    x = 0
    overlay.paste(left_img, (x, 0))
    x += LEFT_ARROW_WIDTH

    overlay.paste(speed_img, (x, 0))
    x += SPEED_WIDTH

    overlay.paste(accel_pedal_img, (x, 0))
    x += ACCEL_PEDAL_WIDTH

    overlay.paste(autopilot_img, (x, 0))
    x += AUTOPILOT_WIDTH

    chart_resized = chart_img.resize((CHART_WIDTH, OVERLAY_HEIGHT), Image.LANCZOS)
    overlay.paste(chart_resized, (x, 0))
    x += CHART_WIDTH

    overlay.paste(brake_img, (x, 0))
    x += BRAKE_WIDTH

    overlay.paste(right_img, (x, 0))

    return np.array(overlay)


def alpha_composite_overlay(frame: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Alpha-blend overlay onto the top of the frame."""
    overlay_rgb = overlay[:, :, :3].astype(np.float32)
    overlay_alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0

    top_section = frame[:OVERLAY_HEIGHT].astype(np.float32)

    blended = overlay_rgb * overlay_alpha + top_section * (1.0 - overlay_alpha)

    result = frame.copy()
    result[:OVERLAY_HEIGHT] = blended.astype(np.uint8)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Tesla Dashcam Telemetry Overlay")
    parser.add_argument("input", help="Input dashcam video file")
    parser.add_argument(
        "-o", "--output",
        help="Output video file (default: <input>-overlay.<ext>)",
    )
    return parser.parse_args()


def default_output_path(input_path: str) -> str:
    root, ext = os.path.splitext(input_path)
    return f"{root}-overlay{ext}"


def main():
    args = parse_args()
    input_video = args.input
    output_video = args.output or default_output_path(input_video)

    print("Extracting SEI metadata...")
    metadata = extract_all_sei_metadata(input_video)
    print(f"Found {len(metadata)} SEI frames")

    if len(metadata) == 0:
        print("Error: No SEI metadata found in video")
        sys.exit(1)

    print("Caching indicator images...")
    indicator_cache = cache_indicator_images()
    autopilot_cache = cache_autopilot_images()

    print("Initializing acceleration chart...")
    chart = AccelChart(CHART_WIDTH, OVERLAY_HEIGHT, CHART_HISTORY_FRAMES)

    print("Starting ffmpeg decoder...")
    decoder = subprocess.Popen(
        [
            "ffmpeg", "-i", input_video,
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    print("Starting ffmpeg encoder...")
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
            "-r", str(FPS),
            "-i", "-",
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            output_video
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

    frame_size = VIDEO_WIDTH * VIDEO_HEIGHT * 3
    frame_idx = 0

    print("Processing frames...")
    try:
        while True:
            raw_frame = decoder.stdout.read(frame_size)
            if len(raw_frame) != frame_size:
                break

            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((VIDEO_HEIGHT, VIDEO_WIDTH, 3))

            meta = metadata[min(frame_idx, len(metadata) - 1)]

            # Extract values
            accel_x = meta.linear_acceleration_mps2_x
            accel_y = meta.linear_acceleration_mps2_y
            accel_z = meta.linear_acceleration_mps2_z
            left_on = meta.blinker_on_left
            right_on = meta.blinker_on_right
            brake_on = meta.brake_applied
            speed_mps = meta.vehicle_speed_mps
            accel_pedal = meta.accelerator_pedal_position
            autopilot_state = meta.autopilot_state

            # Update chart
            chart_img = chart.update(accel_x, accel_y, accel_z)

            # Get cached/generated images
            indicators = indicator_cache[(left_on, right_on, brake_on)]
            autopilot_img = autopilot_cache.get(autopilot_state, autopilot_cache[0])

            # Generate dynamic images
            speed_img = create_speed_image(SPEED_WIDTH, OVERLAY_HEIGHT, speed_mps)
            accel_pedal_img = create_accel_pedal_image(ACCEL_PEDAL_WIDTH, OVERLAY_HEIGHT, accel_pedal)

            # Create overlay
            overlay = create_overlay_frame(chart_img, indicators, speed_img, accel_pedal_img, autopilot_img)

            # Composite onto frame
            result = alpha_composite_overlay(frame, overlay)

            encoder.stdin.write(result.tobytes())

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx} frames...")

    except Exception as e:
        print(f"Error during processing: {e}")
        raise
    finally:
        decoder.stdout.close()
        decoder.wait()
        encoder.stdin.close()
        encoder.wait()

    print(f"Done! Processed {frame_idx} frames")
    print(f"Output: {output_video}")


if __name__ == "__main__":
    main()
