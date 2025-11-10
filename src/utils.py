import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
from PIL import Image
import cv2

def load_video_frames(frame_dir):
    """Load video frames from a directory as a list of PIL Images."""
    frame_paths = sorted([
        os.path.join(frame_dir, fname)
        for fname in os.listdir(frame_dir)
        if fname.endswith('.jpg')
    ])
    return [Image.open(fp) for fp in frame_paths]

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

def visualize_and_save(input_path, video_frames, points, video_res_masks, frame_idx, output_dir):
    # Create 'doc' folder if it doesn't exist
    os.makedirs('doc', exist_ok=True)

    # Load the first frame
    frame = video_frames[frame_idx]

    # Combine masks for all objects into a single mask
    # combined_mask = torch.max(video_res_masks, dim=0, keepdim=True)[0]
    # combined_mask = torch.sigmoid(combined_mask) # Apply sigmoid to convert logits to probabilities
    binarized_mask = (video_res_masks > 0.5).to(torch.uint8) * 255
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
