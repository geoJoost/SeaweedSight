"""Script for pre-processing of video data by splitting into individual trials 
"""

# TODO: Resize to 1024x1024 max
# TODO: Implement proper ROI for all videos
# TODO: Take correct number of frames based on the FPS used (i.e., 15 vs 30 fps)
# TODO: 

import cv2
import os

def process_video_to_frames(
    input_path: str,
    frames_per_second: int = 1,
    remove_ranges: list = None,
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
        remove_ranges: List of frame ranges to remove (e.g., [(4000, 4500), (7900, 8000)]).
        output_dir: Base directory to save frames (default: "data").
    """
    if remove_ranges is None:
        remove_ranges = []

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
    trial_dirs = [os.path.join(output_dir, f"{base_name}_trial{i+1}") for i in range(3)]

    # Create output directories if they don't exist
    for d in trial_dirs:
        os.makedirs(d, exist_ok=True)

    frame_count = 0
    current_trial = 0
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames if needed
        if frame_count % skip_frames != 0:
            frame_count += 1
            continue

        # Check if current frame is in any remove range
        in_remove_range = any(
            start <= frame_count <= end
            for start, end in remove_ranges
        )
        if in_remove_range:
            frame_count += 1
            continue

        # Determine which trial to save to based on remove ranges
        if remove_ranges and frame_count > remove_ranges[0][1]:
            current_trial = 1
        if len(remove_ranges) > 1 and frame_count > remove_ranges[1][1]:
            current_trial = 2

        # Save frame as .jpg
        frame_path = os.path.join(trial_dirs[current_trial], f"{frame_number:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_number += 1
        frame_count += 1

    cap.release()
    print(f"[INFO] Frames saved in: {', '.join(trial_dirs)}")

if __name__ == "__main__":
    input_video = r"data/Ducks/Ulva_05_1.avi"
    remove_ranges = [(0, 104), (2450, 2550), (4920, 5000)]  # TODO: Find exact frames using DaVinci
    process_video_to_frames(input_video, frames_per_second=1, remove_ranges=remove_ranges)
