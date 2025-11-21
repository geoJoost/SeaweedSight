"""Script for pre-processing of video data by splitting into individual trials 
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

    plt.tight_layout()
    plt.savefig('doc/clipped_frame.png')
    plt.close()


def get_master_stats(
    master_path: str,
    roi_width: int,
    roi_height: int,
    normalization_area: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std of the normalization_area across all frames of the master video.
    Optionally plots a sample frame with the normalization_area overlaid.

    Args:
        master_path (str): Path to the master .avi file.
        roi_width (int): Width of the ROI.
        roi_height (int): Height of the ROI.
        normalization_area (tuple): (x, y, w, h) relative to the clipped ROI.

    Returns:
        tuple[np.ndarray, np.ndarray]: Global mean and std of the normalization_area (BGR order).
    """
    cap = cv2.VideoCapture(master_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open master video: {master_path}")

    # List to accumulate all pixel values in the normalization_area
    pixels_list = [[] for _ in range(3)]  # B, G, R

    frame_idx = 0
    sample_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Clip frame to ROI
        roi_x, roi_y, _, _ = calculate_dynamic_roi(frame, roi_width, roi_height)
        clipped_frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # Extract normalization_area
        x, y, w, h = normalization_area
        area = clipped_frame[y:y+h, x:x+w]

        # Accumulate pixel values for each channel
        for i in range(3):
            pixels_list[i].extend(area[:, :, i].flatten())

        frame_idx += 1
        if frame_idx > 10:
            sample_frame = clipped_frame.copy()
            break

    cap.release()

    # Convert lists to NumPy arrays
    pixels_array = [np.array(channel) for channel in pixels_list]

    # Compute global mean and std for each channel
    global_mean = np.array([np.mean(channel) for channel in pixels_array])
    global_std = np.array([np.std(channel) for channel in pixels_array])

    print(
        f"[INFO] Pixels in normalization_area for each channel: {pixels_array[0].shape[0]:,}\n"
        f"[INFO] B: μ={global_mean[0]:.2f}, σ={global_std[0]:.2f}\n"
        f"[INFO] G: μ={global_mean[1]:.2f}, σ={global_std[1]:.2f}\n"
        f"[INFO] R: μ={global_mean[2]:.2f}, σ={global_std[2]:.2f}"
    )

    # # Plot sample frame
    # sample_frame_rgb = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(6, 6))
    # plt.imshow(sample_frame_rgb)
    # rect = patches.Rectangle(
    #     (x, y), w, h,
    #     linewidth=2, edgecolor='r', facecolor='none'
    # )
    # plt.gca().add_patch(rect)
    # plt.title(f"Sample Frame with normalization area")
    # plt.axis('off')
    # plt.savefig('doc/video_normalization.png')
    # plt.close()

    return global_mean, global_std


def normalize_frame(
    frame: np.ndarray,
    master_mean: np.ndarray,
    master_std: np.ndarray,
    normalization_area: tuple,
) -> np.ndarray:
    """
    Normalize a frame (in BGR) so that the normalization_area matches the master's mean and std.
    """
    frame = frame.copy() # Work on a copy
    x, y, w, h = normalization_area
    current_area = frame[y:y+h, x:x+w, :]

    # Calculate current mean and std for each channel
    current_mean = np.mean(current_area, axis=(0, 1))
    current_std = np.std(current_area, axis=(0, 1))

    # Avoid division by zero
    current_std[current_std == 0] = 1

    # Normalize each channel
    for i in range(frame.shape[-1]):  # R, G, B
        frame[:, :, i] = (
            # Scale data based on normalization_area taken from master frame
            # And the current channel statistics
            ((frame[:, :, i] - current_mean[i]) * (master_std[i] / current_std[i]))
            + master_mean[i]
        )

    return np.clip(frame, 0, 255).astype(np.uint8)

def normalize_clipped_frame(clipped_frame, master_mean, master_std):
    """
    Normalize a clipped frame to match the master's mean and std.
    Assumes the clipped frame should match the master's overall distribution.
    """
    # Normalize each channel
    normalized = (clipped_frame - master_mean) / master_std

    # Scale back to [0, 255] and clip to ensure valid range
    normalized = np.clip(normalized * 255, 0, 255)

    # Convert to integer
    return normalized.astype(np.uint8)

    # for i in range(3):  # For each channel (B, G, R)
    #     current_mean = np.mean(clipped_frame[:, :, i])
    #     current_std = np.std(clipped_frame[:, :, i])

    #     clipped_frame[:, :, i] = (clipped_frame[:, :, i] - master_mean[i]) / master_std[i]

    #     # Avoid division by zero
    #     if current_std > 0:
    #         clipped_frame[:, :, i] = (
    #             ((clipped_frame[:, :, i] - current_mean) * (master_std[i] / current_std))
    #             + master_mean[i]
    #         )
    # return np.clip(clipped_frame, 0, 255).astype(np.uint8)

def verify_normalization(
    original_frame: np.ndarray,
    normalized_frame: np.ndarray,
    normalization_area: tuple,
) -> None:
    """
    Plot the original and normalized frames with the normalization_area overlaid for verification.

    Args:
        original_frame (np.ndarray): Original frame (RGB).
        normalized_frame (np.ndarray): Normalized frame (RGB).
        normalization_area (tuple): (x, y, w, h) coordinates of the normalization area.
        title (str): Title for the plot (default: "Normalization Verification").
    """
    x, y, w, h = normalization_area

    # Convert BGR to RGB for plotting
    original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    normalized_rgb = cv2.cvtColor(normalized_frame, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original frame with normalization_area
    axes[0].imshow(original_rgb)
    rect_orig = patches.Rectangle(
        (x, y), w, h,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    axes[0].add_patch(rect_orig)
    axes[0].set_title('Original Frame\n(Red: Normalization Area)')
    axes[0].axis('off')

    # Plot normalized frame with normalization_area
    axes[1].imshow(normalized_rgb)
    rect_norm = patches.Rectangle(
        (x, y), w, h,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    axes[1].add_patch(rect_norm)
    axes[1].set_title('Normalized Frame\n(Red: Normalization Area)')
    axes[1].axis('off')

    plt.suptitle("Normalization")
    plt.tight_layout()
    plt.savefig('doc/video_normalization_verification.png')
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
    seconds_interval: float = 8.0, # Default of 8.0 seconds; roughly 175 frames at 21FPS
    keep_ranges: list = None,
    roi: tuple = None,
    normalize: bool = False,
    normalization_area: tuple = None,
    master_mean: np.ndarray = None,
    master_std: np.ndarray = None,
    save_files: bool = False,
    output_dir: str = "data"
) -> None:
    """
    Process a video file:
    - Saves a frame every for every 'seconds_interval' input.
    - Removes specified frame ranges.
    - Splits the result into parts based on the keep ranges.
    - Optionally normalizes frames to match a master's color distribution.
    - Saves frames as .jpg in folders: data/Ulva_DENSITY_1_trial1/, etc.

    Args:
        input_path (str): Path to the input .avi file.
        seconds_interval (float): Save a frame every `seconds_interval` seconds (default: 5.0).
        keep_ranges (list): List of frame ranges to keep (e.g., [(104, 2450), (2551, 4919), (5001, None)]).
        roi (tuple): Target ROI dimensions as (width, height).
        normalize (bool): If True, normalize frames using master_mean and master_std (default: False).
        normalization_area (tuple): Coordinates (x, y, w, h) of the area to use for normalization, relative to the clipped ROI.
        master_mean (np.ndarray): Mean RGB values of the master's normalization_area.
        master_std (np.ndarray): Standard deviation of RGB values of the master's normalization_area.
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
    trial_dirs = [os.path.join(output_dir, "intermediate", f"{base_name}_trial{i+1}") for i in range(len(keep_ranges))]
    for d in trial_dirs:
        os.makedirs(d, exist_ok=True)

    # Tracking variables
    frame_count = 0
    current_trial = 0
    frame_number = 0
    trial_frames = {f"{base_name}_trial{i+1}": [] for i in range(len(keep_ranges))} # Store arrays for downstream inference

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

        # Save a frame based on the time interval
        if frame_count % frame_skip == 0:
            # Calculate dynamic ROI for this frame, so it is horizontally centered
            roi_x, roi_y, roi_width, roi_height = calculate_dynamic_roi(frame, roi_width, roi_height)

            # Clip the frame to the ROI
            clipped_frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

            # Visualize results
            # verify_roi(frame, roi) # For debugging

            # Normalize frames to account for different illumination situations (i.e., decrease in sunlight)
            if normalize:
                normalized_frame = normalize_clipped_frame(clipped_frame.copy(), master_mean, master_std)

                # verify_normalization(clipped_frame.copy(), normalized_frame, normalization_area)
                clipped_frame = normalized_frame.copy() # Overwrite to keep existing code functional without normalization
            
            # Save the frame to folder
            if save_files:
                frame_path = os.path.join(trial_dirs[current_trial], f"{frame_number:06d}.jpg")
                cv2.imwrite(frame_path, clipped_frame)
            
            trial_frames[f"{base_name}_trial{current_trial+1}"].append(clipped_frame.copy())
            frame_number += 1

        frame_count += 1

    cap.release()
    print(f"[INFO] Finished processing of {input_path}")
    if save_files:
        print(f"[INFO] Frames saved in: {', '.join(trial_dirs)}")
    return trial_frames