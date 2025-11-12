import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
from PIL import Image
import cv2

def get_frame_paths(frame_dir):
    """Get sorted frame paths from a directory."""
    frame_paths = sorted([
        os.path.join(frame_dir, fname)
        for fname in os.listdir(frame_dir)
        if fname.endswith('.jpg')
    ])
    return frame_paths

def load_video_frames(frame_dir):
    """Load video frames from a directory as a list of PIL Images."""
    frame_paths = get_frame_paths(frame_dir)
    return [Image.open(fp) for fp in frame_paths]

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

def visualize_luminance_prompts(frame, l, dark_regions, points, luminance_threshold, output_path="doc/prompt_experiment_luminance.png"):
    """
    Visualize the original frame, luminance channel, dark regions, and connected components.

    Args:
        frame (np.ndarray): Original RGB frame.
        l (np.ndarray): Luminance channel from Lab color space.
        dark_regions (np.ndarray): Boolean mask of dark regions.
        points (list): List of selected points (e.g., [[x1, y1], [x2, y2], ...]).
        luminance_threshold (float): Threshold used for dark regions.
        output_path (str): Path to save the debug figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    # 1. Original frame + prompts
    axes[0].imshow(frame)
    for point in points:
        axes[0].add_patch(Circle((point[0][0], point[0][1]), 10, color='green', fill=True))
        axes[0].add_patch(Circle((point[0][0], point[0][1]), 10, color='white', fill=False, lw=2))
    axes[0].set_title("Original frame with prompts")

    # 2. Luminance channel
    axes[1].imshow(l, cmap='gray')
    axes[1].set_title("Luminance (L)")

    # 3. Dark regions mask
    axes[2].imshow(dark_regions, cmap='gray')
    axes[2].set_title(f"Dark regions (L < {luminance_threshold})")

    # 4. Connected components
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(dark_regions.astype(np.uint8), connectivity=8)
    axes[3].imshow(labels, cmap='nipy_spectral')
    axes[3].set_title("Connected components")

    # Annotate
    fig.text(0.5, 0.01,
             f"Threshold: L < {luminance_threshold}\n"
             f"Selected {len(points)} prompts: {points}",
             ha='center', fontsize=10)

    # plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[INFO] Saved debug figure to: {output_path}")
    print(f"[INFO] Points: {points}")
    print(f"[INFO] Dark region pixels: {np.sum(dark_regions)}")

def visualize_sam2_outputs(input_path, video_frames, points, video_res_masks, frame_idx, data_dir, output_folder, conf_threshold=0.5):
    # Create output directory
    output_dir = os.path.join(output_folder, f"{os.path.basename(data_dir)}_processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the frame
    frame = video_frames[frame_idx]

    # Combine masks for all objects into a single mask
    binarized_mask = (video_res_masks > conf_threshold).to(torch.uint8) * 255
    binarized_mask = binarized_mask.cpu().squeeze().numpy()

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(12, 6))

    # Plot original image with point annotation
    axes[0].imshow(frame)
    for obj_points in points:
        for point in obj_points:
            plt.gca().add_patch(Circle((point[0], point[1]), 20, color='green', fill=True))
            plt.gca().add_patch(Circle((point[0], point[1]), 20, color='white', fill=False, lw=2))
    axes[0].set_title('Image with prompt')
    axes[0].axis('off')

    # Plot mask probabilities
    #axes[1].imshow(frame)
    axes[1].imshow(video_res_masks.cpu().squeeze().to(torch.float32), cmap='viridis')
    axes[1].set_title('Confidence')
    axes[1].axis('off')

    # Plot binarized mask
    axes[2].imshow(binarized_mask, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    # Plot binarized mask overlaid on the original image
    axes[3].imshow(frame)
    axes[3].imshow(binarized_mask, alpha=0.3, cmap='gray')
    axes[3].set_title('Output')
    axes[3].axis('off')

    plt.tight_layout()

    # Save the figure
    basename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"{basename}_{frame_idx:06d}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()

    print(f"[INFO] Saved image to: {output_path}")

    return output_path

def visualize_features(df, conf_threshold, data_dir, output_dir):
    """
    Visualize trends of all features (except cumulative_surface_area) over time.

    Args:
        df (pd.DataFrame): DataFrame containing all features.
        conf_threshold (float): Confidence threshold used for segmentation.
        data_dir (str): Directory containing the data.
        output_dir (str): Directory to save the plots.
    """
    # Define feature keys (excluding cumulative_surface_area)
    feature_keys = ['surface_area', 'mean_R', 'mean_G', 'mean_B', 'mean_L', 'mean_a', 'mean_b']

    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=len(feature_keys), ncols=1, figsize=(8, 8), sharex=True)

    # Plot each feature in its own subplot
    for i, feature in enumerate(feature_keys):
        ax = axes[i]
        ax.plot(df['frame_id'], df[feature], label=feature)
        ax.set_ylabel(feature)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper left')

    # Set common x-label and title
    axes[-1].set_xlabel('Frame [-]')
    plt.suptitle(f'Feature Trends [SAM2; conf: {conf_threshold}]', y=1.02)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    basename = os.path.basename(data_dir)
    plot_path = os.path.join(output_dir, f'{basename}_features.png')
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"[INFO] Saved feature trends plot to: {plot_path}")
    return plot_path
