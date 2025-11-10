import torch
import numpy as np
import pandas as pd

# Custom imports
from src.sam_prompter import prompt_sam2
from src.utils import get_frame_paths, visualize_sam2_outputs, visualize_surface_area

def calculate_surface_area(frame_probs, conf_threshold):
    """
    Calculate surface areas for objects in the current frame.

    Args:
        frame_probs (torch.Tensor): Confidence probabilities for the current frame, shape [num_objects, 1, h, w]
        conf_threshold (int): Threshold to use for  (list): List of object IDs for the current frame.

    Returns:
        dict: Surface area data for the current frame.
    """

    # Binarize
    binarized_mask = (frame_probs > conf_threshold).to(torch.uint8)

    # Calculate surface area
    surface_area = binarized_mask.sum().item()

    return surface_area


# Arguments
data_dir = r"data/Ulva_05_1_trial2"
model_name = "facebook/sam2-hiera-base-plus"
conf_threshold = 0.5

# Initialize DataFrame to store results
df = pd.DataFrame()
df['frame'] = get_frame_paths(data_dir)
df[['model_name', 'conf_threshold']] = model_name, conf_threshold

# Prompt SAM2
video_frames, probs_stack, all_logits = prompt_sam2(data_dir=data_dir, model_name=model_name)

# Propagate over the frames
surface_areas = []
for frame_idx, probs in enumerate(probs_stack):
    # Unpack prompts for this frame
    current_points = all_logits[frame_idx]['points']
    current_labels = all_logits[frame_idx]['labels']

    # Calculate surface area (in px)
    surface_area = calculate_surface_area(probs.squeeze(), conf_threshold)
    surface_areas.append(surface_area)

    print(f"[INFO] Frame {frame_idx} surface area: {surface_area} pixels")

    # Save visualization for each frame with image, confidence, masks, and masks+image
    visualize_sam2_outputs(
        data_dir,
        video_frames,
        current_points,
        probs[frame_idx:frame_idx+1],
        frame_idx=frame_idx,
        data_dir=data_dir,
        conf_threshold=conf_threshold
        )

df['surface_area'] = surface_areas
df['cumulative_surface_area'] = np.cumsum(surface_areas)

visualize_surface_area(surface_areas, conf_threshold, data_dir, output_dir='doc')
print('...')
