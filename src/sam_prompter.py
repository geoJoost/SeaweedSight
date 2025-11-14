"""Script for prompting SAM2 for video segmentation """

import os
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor
import cv2
import numpy as np

# Custom imports
from data_utils import load_video_frames, create_luminance_prompts
from visualization_utils import visualize_sam2_outputs, visualize_luminance_prompts


def prompt_sam2(data_dir, model_name, max_frames=None):
    # Set device: use CUDA if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Initialize predictor
    model = Sam2VideoModel.from_pretrained(model_name).to(device, dtype=torch.bfloat16)
    processor = Sam2VideoProcessor.from_pretrained(model_name)

    # Load your video frames
    video_frames = load_video_frames(data_dir)

    # Slice video_frames if max_frames is specified
    video_frames_inference = video_frames if max_frames is None else video_frames[:max_frames]

    # Initialize video inference session
    inference_session = processor.init_video_session(
        video=video_frames_inference,
        inference_device=device,
        inference_state_device='cpu', # Store cache on CPU
        video_storage_device='cpu', # Store frames on CPU
        max_vision_features_cache_size=1,  # Test
        dtype=torch.bfloat16,
    )

    # Automatically prompt the first frame
    first_frame = video_frames[0]
    points, labels = create_luminance_prompts(first_frame, num_prompts=5, luminance_percentile=10)
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

    # Collect logits for all frames
    all_logits = {}

    # Initialize with first frame
    all_logits[0] = {
        # 'logits': torch.max(video_res_masks, dim=0, keepdim=True)[0],
        'logits': video_res_masks, # Store full logits tensor for all objects
        'points': points,
        'labels': labels,
        'obj_ids': obj_ids
    }

    # Propagate through the entire video
    for sam2_video_output in model.propagate_in_video_iterator(inference_session):
        frame_idx = sam2_video_output.frame_idx
        video_res_masks = processor.post_process_masks(
            [sam2_video_output.pred_masks], 
            original_sizes=[[inference_session.video_height, inference_session.video_width]], 
            binarize=False,
            # binarize=True
        )[0]

        # Combine logits for all objects into a single mask per frame
        combined_logits = torch.max(video_res_masks, dim=0, keepdim=True)[0]
        all_logits[frame_idx] = {
            'logits': combined_logits,
            # 'logits': video_res_masks, # Full logits tensor
            'points': [],  # Default: no prompts
            'labels': [],
            'obj_ids': inference_session.obj_ids
        }

        # Reprompt every 5 frames
        if frame_idx % 5 == 0 and frame_idx != 0:
            current_masks = {obj_id: mask for obj_id, mask in zip(inference_session.obj_ids, video_res_masks)}

            # Select new prompts
            new_points, new_labels = create_luminance_prompts(video_frames[frame_idx], 
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
    
    # For visualiztaion, combine logits across objects as before
    # Stack all logits into a single tensor (shape: [N, H, W])
    logits_stack = torch.stack([all_logits[i]['logits'] for i in sorted(all_logits.keys())], dim=0)

    # Apply sigmoid to the entire stack at once
    probs_stack = torch.sigmoid(logits_stack)

    print(f"[INFO] Tracked {len(inference_session.obj_ids):,} objects through {len(all_logits):,} frames")

    return video_frames, probs_stack, all_logits