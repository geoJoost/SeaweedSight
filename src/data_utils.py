import os
import numpy as np
import torch
from PIL import Image
import cv2
import re
import os
import torch
import cv2
import numpy as np
from scipy.spatial.distance import cdist

# Custom imports
from src.visualization_utils import visualize_luminance_prompts

def get_frame_paths(frame_dir):
    """Get sorted frame paths from a directory."""
    frame_paths = sorted([
        os.path.join(frame_dir, fname)
        for fname in os.listdir(frame_dir)
        if fname.endswith(('.jpg', '.png'))
    ])
    return frame_paths

def load_video_frames(frame_dir):
    """Load video frames from a directory as a list of PIL Images."""
    frame_paths = get_frame_paths(frame_dir)
    return [Image.open(fp) for fp in frame_paths]

def create_luminance_prompts(frame, existing_masks=None, num_prompts=5, luminance_percentile=10):
    """
    Automatically select new positive prompts on darker regions.
    Args:
        frame: Current RGB frame.
        existing_masks: Optional, to avoid re-prompting on already segmented regions.
        num_prompts: Number of positive prompts to select.
        luminance_percentile: Percentile to use for thresholding to create dark regions.
    Returns:
        points: List of new prompt coordinates.
        labels: List of prompt labels (1 for positive).
    """
    frame_np = np.array(frame)
    r, g, b = frame_np.transpose(2, 0, 1)

    # # Convert to Lab color space for better green/white separation
    lab = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Flatten luminance and find the threshold for the 10% darkest pixels, corresponding to the seaweed
    dark_threshold = np.percentile(l.flatten(), luminance_percentile)  # 10th percentile = 10% darkest
    dark_regions = l < dark_threshold

    # # Flatten luminance and find the threshold for the 90% brightest pixels, corresponding to the background
    # dark_threshold = np.percentile(l.flatten(), 10)  # 90th percentile = 90% darkest
    # dark_regions = l > dark_threshold
    
    # Exclude already segmented regions
    if existing_masks is not None:
        # Combine all existing masks into a single binary mask
        combined_mask = torch.mean(
            torch.stack(list(existing_masks.values())),
            dim=0
        )[0].to(torch.float32).squeeze(0)
        
        # Convert logits to probabilities and binarize at 0.5
        combined_mask_probs = torch.sigmoid(combined_mask)
        combined_mask_binary = (combined_mask_probs > 0.5).cpu().numpy().astype(bool)

        dark_regions = dark_regions & (~combined_mask_binary)

    # Find connected components in green regions
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_regions.astype(np.uint8), connectivity=8)

    # Select the largest 'num_prompts' components
    if num_labels > 1:
        largest_indices = np.argsort([stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)])[-num_prompts:]
        points = []
        for idx in largest_indices:
            # Calculate centroid
            x = int(stats[idx + 1, cv2.CC_STAT_LEFT] + stats[idx + 1, cv2.CC_STAT_WIDTH] / 2)
            y = int(stats[idx + 1, cv2.CC_STAT_TOP] + stats[idx + 1, cv2.CC_STAT_HEIGHT] / 2)

            # Get mask points for this component only
            component_mask = (labels == idx + 1)
            component_points = np.argwhere(component_mask)  # Shape: [N, 2] (y, x)

            # Check if centroid is within the region
            if dark_regions[y, x]:
                points.append([[x, y]])
                # print(f"[DEBUG] Point ({x, y} is on dark_regions)")
            else:
                # Find the closest mask point
                # print(f"[DEBUG] Point ({x, y} is NOT on dark_regions)")
                distances = cdist([(x, y)], component_points)
                closest_y, closest_x = component_points[np.argmin(distances)]
                points.append([[closest_x, closest_y]])

            # points.append([[x, y]])
        labels = [[1] for _ in range(len(points))]

        # Visualize steps used to create point prompts
        # visualize_luminance_prompts(frame, l, dark_regions, points, luminance_percentile)

        return points, labels
    else:
        print(f"[WARNING] No new point prompts found")
        # visualize_luminance_prompts(frame, l, dark_regions, [[[0, 0]]], luminance_threshold)
        return [[]], [[]]


def extract_density_from_dir(data_dir):
    """
    Extract density (g/L) from directory name (e.g., Ulva_05_1_trial1 -> 0.5).
    """
    match = re.search(r'Ulva_(\d+)', data_dir)
    if match:
        density_str = match.group(1)
        density = float(density_str) / 10  # Convert to float (e.g., 05 -> 0.5)
        return density
    else:
        raise ValueError(f"[ERROR] Could not extract density from directory: {data_dir}")

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

    return surface_area, binarized_mask

def extract_color_features(frame, binarized_mask):
    """
    Extract mean RGB and Lab color values from the region defined by binarized_mask.

    Args:
        frame (np.ndarray): Original frame (BGR format, as loaded by OpenCV).
        binarized_mask (np.ndarray): Binary mask (0, 1) where 1 indicates the region of interest.

    Returns:
        dict: Dictionary containing mean R, G, B, L, a, b values for the masked region.
    """
    # Correct formats to Numpy
    mask = binarized_mask.to(torch.bool).cpu().numpy()
    frame_np = np.array(frame)

    # Extract RGB values
    r, g, b = frame_np[mask, 0], frame_np[mask, 1], frame_np[mask, 2]

    # Convert to Lab color space
    lab_frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2LAB)
    l, a, lab_b = lab_frame[mask].T

    # Compute mean values
    features = {
        'mean_R': np.mean(r),
        'mean_G': np.mean(g),
        'mean_B': np.mean(b),
        'mean_L': np.mean(l),
        'mean_a': np.mean(a),
        'mean_b': np.mean(lab_b),
    }
    return features
