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
from typing import List, Dict, Tuple, Optional, Union

# Custom imports
from src.visualization_utils import visualize_luminance_prompts

def create_luminance_prompts(
        frame: np.ndarray, 
        num_prompts: int = 5, 
        luminance_percentile: int = 10
    ) -> Tuple[List[List[int]], List[List[int]]]:
    """ Automatically select new positive prompts on darker regions of an RGB frame, matching seaweed regions. """

    # Convert frame to CIELAB color space for luminance analysis
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    luminance_channel, a, b = cv2.split(lab)

    # Threshold luminance to find dark regions, corresponding to the seaweed
    dark_threshold = np.percentile(luminance_channel.flatten(), luminance_percentile) # 10th percentile = 10% darkest
    dark_regions = luminance_channel < dark_threshold

    # Find connected components in the dark regions
    num_components, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
        dark_regions.astype(np.uint8), connectivity=8
    )

    # Select the largest 'num_prompts' components
    if num_components > 1:
        # Sort components by area and select the largest
        largest_indices = np.argsort(
            [component_stats[i, cv2.CC_STAT_AREA] for i in range(1, num_components)]
            )[-num_prompts:]
        
        prompt_coordinates = []
        for idx in largest_indices:
            # Calculate centroid
            centroid_x = int(
                component_stats[idx + 1, cv2.CC_STAT_LEFT] + 
                component_stats[idx + 1, cv2.CC_STAT_WIDTH] / 2
            )
            centroid_y = int(
                component_stats[idx + 1, cv2.CC_STAT_TOP] + 
                component_stats[idx + 1, cv2.CC_STAT_HEIGHT] / 2
            )

            # Get all points in this component
            component_mask = (component_labels == idx + 1)
            component_points = np.argwhere(component_mask)  # Shape: [N, 2] (y, x)

            # Use centroid if it lies within the dark region, otherwise find the closest pixel
            if dark_regions[centroid_y, centroid_x]:
                prompt_coordinates.append([[centroid_x, centroid_y]])
            
            else:
                # Find the closest pixel in the component to the centroid
                distances = cdist([(centroid_x, centroid_y)], component_points)
                closest_y, closest_x = component_points[np.argmin(distances)]
                prompt_coordinates.append([[closest_x, closest_y]])

        prompt_labels = [[1] for _ in range(len(prompt_coordinates))]

        # Visualize steps used to create point prompts
        # visualize_luminance_prompts(frame, luminance_channel, dark_regions, prompt_coordinates, luminance_percentile)

        return prompt_coordinates, prompt_labels
    else:
        print(f"[WARNING] No new point prompts found")
        return [[]], [[]]

def extract_density_from_path(cycle_name: str) -> Union[float, None]:
    """ Extract density (g/L) from cycle_name name (e.g., Ulva_05_1_cyclel1 -> 0.5) """
    match = re.search(r'Ulva_(\d+)', cycle_name)
    if match:
        density_str = match.group(1)
        density = float(density_str) / 10  # Convert to float (e.g., 05 -> 0.5)
        return density
    else:
        raise ValueError(f"[ERROR] Could not extract density from path: {cycle_name}")

def calculate_surface_area(
        frame_probs: torch.Tensor,
        conf_threshold: float
        ) -> Tuple[int, torch.Tensor]:
    """ Calculate the surface area of segmented objects in a frame based on confidence probabilities """

    # Binarize the probability map using the confidence threshold
    binarized_mask = (frame_probs > conf_threshold).to(torch.uint8)

    # Calculate the surface area (number of pixels above the threshold)
    surface_area = binarized_mask.sum().item()

    return surface_area, binarized_mask

def extract_color_features(
        bgr_frame: np.ndarray, 
        binarized_mask: torch.Tensor
    ) -> Dict[str, float]:
    """
    Extract mean RGB and CIELAB color values from the region defined by binarized_mask """
    # Convert mask to boolean numpy array
    mask = binarized_mask.to(torch.bool).cpu().numpy()
    # frame_np = np.array(frame)

    # Extract RGB values (OpenCV uses BGR format)
    b, g, r = bgr_frame[mask, 0], bgr_frame[mask, 1], bgr_frame[mask, 2]

    # Convert to Lab color space
    lab_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
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