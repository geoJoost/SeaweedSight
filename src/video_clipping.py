"""Script for pre-processing of video data by splitting into individual trials 
"""

import cv2
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

    plt.tight_layout()
    plt.savefig('doc/clipped_frame.png')
    plt.close()

def process_video_to_frames(
    input_path: str,
    frames_per_second: int = 1,
    keep_ranges: list = None,
    output_dir: str = "data"
) -> None:
    """
    Process a video file:
    - Selects 1 frame per second (or as specified).
    - Removes specified frame ranges.
    - Splits the result into three parts based on the remove ranges.
    - Saves frames as .jpg in folders: data/Ulva_DENSITY_1_trial1/, etc.

    Args:
        input_path: Path to the input .avi file.
        frames_per_second: Number of frames to select per second (default: 1).
        keep_ranges: List of frame ranges to keep (e.g., [(104, 2450), (2551, 4919), (5001, None)])
        output_dir: Base directory to save frames (default: "data").
    """
    if keep_ranges is None:
        keep_ranges = []

    # Read the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Total of {total_frames:,} frames found in {input_path}")

    # Calculate frames to skip for desired frame rate
    skip_frames = int(fps // frames_per_second)

    # Generate output directory names
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    trial_dirs = [os.path.join(output_dir, f"{base_name}_trial{i+1}") for i in range(len(keep_ranges))]

    # Create output directories if they don't exist
    for d in trial_dirs:
        os.makedirs(d, exist_ok=True)

    frame_count = 0
    current_trial = 0
    frame_number = 0 # Frame number within the current trial

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Skip frames to achieve the desired frame rate
        if frame_count % skip_frames != 0:
            frame_count += 1
            continue

        # Check if the current frame is in any of the keep ranges
        in_keep_range = False
        for trial_idx, (start, end) in enumerate(keep_ranges):
            # Check if the frame is within the current keep range
            if start <= frame_count <= (end if end is not None else float('inf')):
                in_keep_range = True
                # Switch trials if necessary
                if trial_idx != current_trial:
                    current_trial = trial_idx
                    frame_number = 0  # Reset frame number for the new trial
                break

        if not in_keep_range:
            frame_count += 1
            continue  # Skip frames not in any keep range

        # Save the frame
        frame_path = os.path.join(trial_dirs[current_trial], f"{frame_number:06d}.jpg")
        cv2.imwrite(frame_path, frame)

        # Increment counters
        frame_number += 1
        frame_count += 1

    cap.release()
    print(f"[INFO] Frames saved in: {', '.join(trial_dirs)}")

def process_video_n_frames(
    input_path: str,
    n_frames: int = 175,
    keep_ranges: list = None,
    roi: tuple = None,
    output_dir: str = "data"
) -> None:
    """
    Process a video file:
    - Saves 1 frame every `n_frames` frames.
    - Removes specified frame ranges.
    - Splits the result into parts based on the keep ranges.
    - Saves frames as .jpg in folders: data/Ulva_DENSITY_1_trial1/, etc.

    Args:
        input_path: Path to the input .avi file.
        n_frames: Save a frame every `n_frames` frames (default: 175).
        keep_ranges: List of frame ranges to keep (e.g., [(104, 2450), (2551, 4919), (5001, None)])
        roi:
        output_dir: Base directory to save frames (default: "data").
    """
    if keep_ranges is None:
        keep_ranges = []
    
    if roi is None:
        roi = (None, None)  # Default: no clipping
    roi_width, roi_height = roi

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error: Could not open video.")

    # Extract footage parameters
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Total of {total_frames:,} frames found in {input_path}")

    # Create output directories
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    trial_dirs = [os.path.join(output_dir, f"{base_name}_trial{i+1}") for i in range(len(keep_ranges))]
    for d in trial_dirs:
        os.makedirs(d, exist_ok=True)

    # Tracking variables
    frame_count = 0
    current_trial = 0
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the current frame is in any of the keep ranges
        in_keep_range = False
        for trial_idx, (start, end) in enumerate(keep_ranges):
            if start <= frame_count <= (end if end is not None else float('inf')):
                in_keep_range = True
                if trial_idx != current_trial:
                    current_trial = trial_idx
                    frame_number = 0
                break

        if not in_keep_range:
            frame_count += 1
            continue

        # Save a frame every `n_frames` frames
        if frame_count % n_frames == 0:
            # Calculate dynamic ROI for this frame, so it is horizontally centered
            roi_x, roi_y, roi_width, roi_height = calculate_dynamic_roi(frame, roi_width, roi_height)

            # Clip the frame to the ROI
            clipped_frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

            # Visualize results
            # verify_roi(frame, roi) # For debugging

            # Save the clipped frame
            frame_path = os.path.join(trial_dirs[current_trial], f"{frame_number:06d}.jpg")
            cv2.imwrite(frame_path, clipped_frame)
            frame_number += 1

        frame_count += 1

    cap.release()
    print(f"[INFO] Frames saved in: {', '.join(trial_dirs)}")

## TODO's ##
# TODO: Find exact frames using DaVinci
# TODO: Implement proper ROI for all videos
# TODO: Take correct number of frames based on the FPS used (i.e., 15 vs 30 fps)
# TODO: At 0.5G/L, more than three trials are made

video_configs = {
    r"data/Ducks/Ulva_05_1.avi": [(204, 3980), (4090, 9236), (9489, 13285)],
    r"data/Ducks/Ulva_10_1.avi": [(339, 4176), (4313, 7691), (7865, 11453)],
    r"data/Ducks/Ulva_15_1.avi": [(119, 2670), (2850, 5143), (5480, 7741)],
    r"data/Ducks/Ulva_20_3.avi": [(115, 2850), (2906, 5981), (6023, 8672)],
    r"data/Ducks/Ulva_25_3.avi": [(205, 2312), (2342, 4682), (4724, 6936)],

    }

# Find the smallest ROI
roi_width, roi_height = find_smallest_roi(video_configs)

# Split video into individual trials
for input_video, keep_ranges in video_configs.items():
    print(f"[INFO] Processing {input_video} with keep ranges: {keep_ranges}")
    process_video_n_frames(input_video, n_frames=175, keep_ranges=keep_ranges, roi=(roi_width, roi_height))

    # process_video_to_frames(input_video, frames_per_second=2, keep_ranges=keep_ranges) # Old function