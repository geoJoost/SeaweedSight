"""Script for prompting SAM2 for video segmentation """

import os
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor
import cv2
import numpy as np

# Custom imports
from utils import load_video_frames, visualize_and_save, visualize_luminance_prompts

def select_prompts(frame, existing_masks=None, num_prompts=5, luminance_percentile=10):
    """
    Automatically select new positive prompts on green/edge regions.
    Args:
        frame: Current RGB frame.
        existing_masks: Optional, to avoid re-prompting on already segmented regions.
        num_prompts: Number of positive prompts to select.
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
            x = int(stats[idx + 1, cv2.CC_STAT_LEFT] + stats[idx + 1, cv2.CC_STAT_WIDTH] / 2)
            y = int(stats[idx + 1, cv2.CC_STAT_TOP] + stats[idx + 1, cv2.CC_STAT_HEIGHT] / 2)
            points.append([[x, y]])
        labels = [[1] for _ in range(len(points))]

        # Visualize steps used to create point prompts
        # visualize_luminance_prompts(frame, l, dark_regions, points, luminance_percentile)

        return points, labels
    else:
        print(f"[WARNING] No new point prompts found")
        # visualize_luminance_prompts(frame, l, dark_regions, [[[0, 0]]], luminance_threshold)
        return [[]], [[]]

def main(data_dir, model):
    # Set device: use CUDA if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Initialize predictor
    model = Sam2VideoModel.from_pretrained("facebook/sam2-hiera-base-plus").to(device, dtype=torch.bfloat16)
    processor = Sam2VideoProcessor.from_pretrained("facebook/sam2-hiera-base-plus")

    # Load your video frames
    video_frames = load_video_frames(data_dir)

    # Initialize video inference session
    inference_session = processor.init_video_session(
        video=video_frames[:150],
        inference_device=device,
        inference_state_device='cpu', # Store cache on CPU
        video_storage_device='cpu', # Store frames on CPU
        max_vision_features_cache_size=1,  # Test
        dtype=torch.bfloat16,
    )

    # Automatically prompt the first frame
    first_frame = video_frames[0]
    points, labels = select_prompts(first_frame, num_prompts=5, luminance_percentile=10)
    obj_ids = [1, 2, 3, 4, 5]  # Start with object IDs 1 and 2

    # Inputs into processor
    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=0,
        obj_ids=obj_ids,
        input_points=[points],
        input_labels=[labels],
    )

    # Run inference for the first frame
    outputs = model(inference_session=inference_session, frame_idx=0)
    video_res_masks = processor.post_process_masks(
        [outputs.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
    )[0]
    print(f"[INFO] Segmentation shape: {video_res_masks.shape}")

    # Visualize first frame
    # visualize_and_save(data_dir, video_frames, points, video_res_masks, frame_idx=0, output_dir="doc")

    # Create output directory for this video
    output_dir = os.path.join("data", f"{os.path.basename(data_dir)}_processed")
    os.makedirs(output_dir, exist_ok=True)

    # Collect logits for all frames
    all_logits = {}

    # Initialize with first frame
    all_logits[0] = {
        'logits': torch.max(video_res_masks, dim=0, keepdim=True)[0],
        'points': points,
        'labels': labels
    }

    # Propagate through the entire video
    for sam2_video_output in model.propagate_in_video_iterator(inference_session):
        frame_idx = sam2_video_output.frame_idx
        video_res_masks = processor.post_process_masks(
            [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
        )[0]

        # Combine logits for all objects into a single mask per frame
        combined_logits = torch.max(video_res_masks, dim=0, keepdim=True)[0]
        all_logits[frame_idx] = {
            'logits': combined_logits,
            'points': [],  # Default: no prompts
            'labels': []
        }

        # Reprompt every 5 frames
        if frame_idx % 5 == 0 and frame_idx != 0:
            current_masks = {obj_id: mask for obj_id, mask in zip(inference_session.obj_ids, video_res_masks)}

            # Select new prompts
            new_points, new_labels = select_prompts(video_frames[frame_idx], 
                                                    existing_masks=current_masks, 
                                                    num_prompts=5, 
                                                    luminance_percentile=10
                                                    )

            # Update prompts for this frame
            all_logits[frame_idx]['points'] = new_points
            all_logits[frame_idx]['labels'] = new_labels

            # Assign new object IDS
            new_obj_ids = [max(inference_session.obj_ids) + i + 1 for i in range(len(new_points))]

            # Re-prompt SAM2
            processor.add_inputs_to_inference_session(
                inference_session=inference_session,
                frame_idx=frame_idx,
                obj_ids=new_obj_ids,
                input_points=[new_points],
                input_labels=[new_labels],
            )

            # Clear memory
            torch.cuda.empty_cache() # TODO: Check if this actually works

    # Stack all logits into a single tensor (shape: [N, H, W])
    logits_stack = torch.stack([all_logits[i]['logits'] for i in sorted(all_logits.keys())], dim=0)

    # Apply sigmoid to the entire stack at once
    probs_stack = torch.sigmoid(logits_stack)

    # Create output directory
    output_dir = os.path.join("data", f"{os.path.basename(data_dir)}_processed")
    os.makedirs(output_dir, exist_ok=True)

    # Visualize and save all frames
    for frame_idx, probs in enumerate(probs_stack):
        # Unpack prompts for this frame
        current_points = all_logits[frame_idx]['points']
        current_labels = all_logits[frame_idx]['labels']

        # Save visualization for each frame with image, confidence, masks, and masks+image
        visualize_and_save(
            data_dir,
            video_frames,
            current_points,
            probs_stack[frame_idx:frame_idx+1],
            frame_idx=frame_idx,
            output_dir=output_dir
        )

    # print(f"[INFO] Tracked {len(inference_session.obj_ids):,} objects through {len(video_segments):,} frames")
    print('...')

# Execute function
if __name__ == "__main__":
    main(data_dir=r"data/Ulva_05_1_trial3", model="facebook/sam2-hiera-large")