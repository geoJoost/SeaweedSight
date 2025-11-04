import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
from PIL import Image

def load_video_frames(frame_dir):
    """Load video frames from a directory as a list of PIL Images."""
    frame_paths = sorted([
        os.path.join(frame_dir, fname)
        for fname in os.listdir(frame_dir)
        if fname.endswith('.jpg')
    ])
    return [Image.open(fp) for fp in frame_paths]

def visualize_and_save(input_path, video_frames, points, video_res_masks, frame_idx, output_dir):
    # Create 'doc' folder if it doesn't exist
    os.makedirs('doc', exist_ok=True)

    # Load the first frame
    frame = video_frames[frame_idx]

    # Combine masks for all objects into a single mask
    # combined_mask = torch.max(video_res_masks, dim=0, keepdim=True)[0]
    # combined_mask = torch.sigmoid(combined_mask) # Apply sigmoid to convert logits to probabilities
    binarized_mask = (video_res_masks > 0.2).to(torch.uint8) * 255
    binarized_mask = binarized_mask.cpu().squeeze().numpy()

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(12, 6))

    # Plot original image with point annotation
    axes[0].imshow(frame)
    for obj_points in points:
        for point_list in obj_points:
            for point in point_list:
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
