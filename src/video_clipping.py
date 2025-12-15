"""Script for pre-processing of video data by splitting into individual cycles 
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_dynamic_roi(frame, roi_width, roi_height):
    height, width = frame.shape[:2]

    # Center the ROI horizontally
    roi_x = max(0, (width - roi_width) // 2)
    roi_y = 0  # Align to the top

    # Ensure ROI fits within the frame
    roi_width = min(roi_width, width - roi_x)
    roi_height = min(roi_height, height - roi_y)

    return roi_x, roi_y, roi_width, roi_height

def find_smallest_roi(video_configs):
    """
    Find the smallest ROI (Region of Interest) across all videos.
    Assumes the ROI is centered horizontally and aligned to the top vertically.
    """
    min_width = float('inf')
    min_height = float('inf')
    original_widths = []

    for video_path, keep_ranges in video_configs.items():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            continue

        # Get FPS of the video
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Get video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] {os.path.basename(video_path)}: {width} x {height} | FPS: {fps:.2f}")

        # Store the original width for ROI calculation
        if width not in original_widths:
            original_widths.append(width)

        # Update smallest dimensions
        if width < min_width:
            min_width = width
        if height < min_height:
            min_height = height

        cap.release()

    # Calculate ROI coordinates
    roi_width = min_width
    roi_height = min_height

    print(f"[INFO] Smallest ROI: width={roi_width}, height={roi_height}")
    return roi_width, roi_height

def verify_roi(frame, roi):
    """
    Verify the ROI by plotting:
    1. Original full-frame
    2. Original full-frame with ROI overlaid as a red box
    3. Final clipped frame

    Args:
        frame (numpy.ndarray): The frame to visualize.
        roi (tuple): ROI coordinates (roi_x, roi_y, roi_width, roi_height).
    """
    roi_x, roi_y, roi_width, roi_height = roi

    # Convert BGR frame to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Clip the frame to the ROI
    clipped_frame_rgb = frame_rgb[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    # Create the figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Subplot 1: Original full-frame
    axes[0].imshow(frame_rgb)
    axes[0].set_title('Original Full-Frame')
    axes[0].axis('off')

    # Subplot 2: Original full-frame with ROI overlaid
    axes[1].imshow(frame_rgb)
    rect = patches.Rectangle(
        (roi_x, roi_y), roi_width, roi_height,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    axes[1].add_patch(rect)
    axes[1].set_title('Original with ROI Overlay')
    axes[1].axis('off')

    # Subplot 3: Final clipped frame
    axes[2].imshow(clipped_frame_rgb)
    axes[2].set_title('Clipped frame')
    axes[2].axis('off')

def process_video_n_seconds(
    input_path: str,
    seconds_interval: float = 8.0, # Default of 8.0 seconds; roughly 175 frames at 21FPS
    keep_ranges: list = None,
    roi: tuple = None,
    save_files: bool = False,
    output_dir: str = "data"
) -> None:
    """
    Process a video file:
    - Saves a frame every for every 'seconds_interval' input.
    - Removes specified frame ranges.
    - Splits the result into parts based on the keep ranges.
    - Optionally normalizes frames to match a master's color distribution.
    - Saves frames as .jpg in folders: data/Ulva_DENSITY_1_cycle1/, etc.

    Args:
        input_path (str): Path to the input .avi file.
        seconds_interval (float): Save a frame every `seconds_interval` seconds (default: 5.0).
        keep_ranges (list): List of frame ranges to keep (e.g., [(104, 2450), (2551, 4919), (5001, None)]).
        roi (tuple): Target ROI dimensions as (width, height).
        save_jpg (bool): If True, saves frames as .jpg (default: True).
        output_dir (str): Base directory to save frames (default: "data").
    """
    if keep_ranges is None:
        keep_ranges = []
    
    if roi is None:
        roi = (None, None)  # Default: no clipping
    roi_width, roi_height = roi

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")
    print(f"{'-' * 50}")
    print(f"[INFO] Started processing of {input_path} into frames per {seconds_interval} seconds")

    # Extract footage parameters
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Total of {total_frames:,} frames found in {input_path}")

    # Calculate how many frames to skip on the desired interval in seconds
    frame_skip = int(fps * seconds_interval)
    
    # Create output directories
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    cycle_dirs = [os.path.join(output_dir, "intermediate", f"{base_name}_cycle{i+1}") for i in range(len(keep_ranges))]
    for d in cycle_dirs:
        os.makedirs(d, exist_ok=True)

    # Tracking variables
    frame_count = 0
    current_cycle = 0
    frame_number = 0
    cycle_frames = {f"{base_name}_cycle{i+1}": [] for i in range(len(keep_ranges))} # Store arrays for downstream inference

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the current frame is in any of the keep ranges
        in_keep_range = False
        for cycle_idx, (start, end) in enumerate(keep_ranges):
            if start <= frame_count <= (end if end is not None else float('inf')):
                in_keep_range = True
                if cycle_idx != current_cycle:
                    current_cycle = cycle_idx
                    frame_number = 0
                break

        if not in_keep_range:
            frame_count += 1
            continue

        # Save a frame based on the time interval
        if frame_count % frame_skip == 0:
            # Calculate dynamic ROI for this frame, so it is horizontally centered
            roi_x, roi_y, roi_width, roi_height = calculate_dynamic_roi(frame, roi_width, roi_height)

            # Clip the frame to the ROI
            clipped_frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

            # Visualize results
            # verify_roi(frame, roi) # For debugging

            # Save the frame to folder
            if save_files:
                frame_path = os.path.join(cycle_dirs[current_cycle], f"{frame_number:06d}.jpg")
                cv2.imwrite(frame_path, clipped_frame)
            
            cycle_frames[f"{base_name}_cycle{current_cycle+1}"].append(clipped_frame.copy())
            frame_number += 1

        frame_count += 1

    cap.release()
    print(f"[INFO] Finished processing of {input_path}")
    if save_files:
        print(f"[INFO] Frames saved in: {', '.join(cycle_dirs)}")
    return cycle_frames