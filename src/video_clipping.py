"""Script for pre-processing of video data by splitting into individual trials 
"""

import cv2
import os

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
    trial_dirs = [os.path.join(output_dir, f"{base_name}_trial_png{i+1}") for i in range(len(keep_ranges))]

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

## TODO's ##
# TODO: Find exact frames using DaVinci
# TODO: Implement proper ROI for all videos
# TODO: Take correct number of frames based on the FPS used (i.e., 15 vs 30 fps)
# TODO: At 0.5G/L, more than three trials are made

# Split video into individual trials
if __name__ == "__main__":
    input_video = r"data/Ducks/Ulva_05_1.avi"
    keep_ranges = [(104, 2450), (2550, 4920), (5000, None)]  
    process_video_to_frames(input_video, frames_per_second=1, keep_ranges=keep_ranges)
