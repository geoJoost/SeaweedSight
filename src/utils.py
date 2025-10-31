import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch

def visualize_and_save(input_path, video_frames, points, video_res_masks, ann_frame_idx=0):
    # Create 'doc' folder if it doesn't exist
    os.makedirs('doc', exist_ok=True)

    # Load the first frame
    frame = video_frames[ann_frame_idx]

    # Binarize the mask
    binarized_mask = (video_res_masks[ann_frame_idx] > 0.5).to(torch.uint8) * 255
    binarized_mask = binarized_mask.squeeze(0) # Remove first channel

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(12, 6))

    # Plot original image with point annotation
    axes[0].imshow(frame)
    for point_list in points:
        for point in point_list[0]:
            axes[0].add_patch(Circle((point[0], point[1]), 5, color='green', fill=True))
            axes[0].add_patch(Circle((point[0], point[1]), 5, color='white', fill=False, lw=2))
    axes[0].set_title('Image with prompt')
    axes[0].axis('off')

    # Plot mask probabilities
    #axes[1].imshow(frame)
    axes[1].imshow(video_res_masks[ann_frame_idx].squeeze(0).to(torch.float32), cmap='viridis')
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
    output_path = os.path.join('doc', f"{os.path.basename(input_path)}_prompt_example.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()

    return output_path
