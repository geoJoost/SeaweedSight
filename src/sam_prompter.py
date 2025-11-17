"""Script for prompting SAM2 for video segmentation """

import os
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor
from transformers import Sam2Model, Sam2Processor, Sam2ImageProcessorFast
from transformers import SamModel, SamProcessor
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Custom imports
from src.data_utils import load_video_frames, create_luminance_prompts
from src.visualization_utils import visualize_sam2_outputs, visualize_luminance_prompts

""" Used for semantic/instance segmentation on video data """
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
        # max_vision_features_cache_size=1,  # Test
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
        print(f"[DEBUG] Working on frame: {frame_idx}")
        
        # Reprompt every 5 frames
        if frame_idx % 5 == 0 and frame_idx != 0:
            current_masks = {obj_id: mask for obj_id, mask in zip(inference_session.obj_ids, video_res_masks)}

            # Select new prompts
            new_points, new_labels = create_luminance_prompts(video_frames[frame_idx], 
                                                    existing_masks=current_masks, 
                                                    num_prompts=5, 
                                                    luminance_percentile=10
                                                    )

            # Assign new object IDS
            new_obj_ids = [max(inference_session.obj_ids) + i + 1 for i in range(len(new_points))]

            # Re-prompt SAM2
            processor.add_inputs_to_inference_session(
                inference_session=inference_session,
                frame_idx=frame_idx,
                obj_ids=obj_ids,#new_obj_ids,
                input_points=[new_points],
                input_labels=[new_labels],
            )

            # Clear memory
            torch.cuda.empty_cache() # TODO: Check if this actually works

        # Run inference on this frame
        video_res_masks = processor.post_process_masks(
            [sam2_video_output.pred_masks], 
            original_sizes=[[inference_session.video_height, inference_session.video_width]], 
            binarize=False,
            # binarize=True
        )[0]

        # Combine logits for all objects into a single mask per frame
        combined_logits = torch.max(video_res_masks, dim=0, keepdim=True)[0]

        # Update all_logits for this frame
        all_logits[frame_idx] = {
            'logits': combined_logits,
            'points': new_points if frame_idx % 5 == 0 and frame_idx != 0 else [],
            'labels': new_labels if frame_idx % 5 == 0 and frame_idx != 0 else [],
            'obj_ids': inference_session.obj_ids,
        }


    
    # For visualiztaion, combine logits across objects as before
    # Stack all logits into a single tensor (shape: [N, H, W])
    logits_stack = torch.stack([all_logits[i]['logits'] for i in sorted(all_logits.keys())], dim=0)

    # Apply sigmoid to the entire stack at once
    probs_stack = torch.sigmoid(logits_stack)

    print(f"[INFO] Tracked {len(inference_session.obj_ids):,} objects through {len(all_logits):,} frames")

    return video_frames, probs_stack, all_logits

def segment_frames_sam1(data_dir, model_name, max_frames=None):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Initialize model
    # TODO: Remove hard-coding
    model_name = 'facebook/sam-vit-huge'
    model = SamModel.from_pretrained(model_name).to(device, dtype=torch.bfloat16)
    processor = SamProcessor.from_pretrained(model_name)

    # Load your video frames
    video_frames = load_video_frames(data_dir)

    # Slice video_frames if max_frames is specified
    video_frames_inference = video_frames if max_frames is None else video_frames[:max_frames]

    # Collect logits for all frames
    all_outputs = {}

    # Segment each image individually (i.e., not treated as video for SAM2)
    for frame_idx, pil_image in enumerate(video_frames_inference):
        # Create prompts
        points, labels = create_luminance_prompts(pil_image, num_prompts=5, luminance_percentile=10)

        # # Convert to tensor and explicitly cast to bfloat16
        # import torchvision.transforms.functional as TF
        # image_tensor = TF.to_tensor(pil_image).to(torch.bfloat16)

        # Convert outputs into SAM1 format
        points_tensor = torch.tensor(points).view(1, 1, -1, 2)
        labels_tensor = torch.tensor(labels).view(1, 1, -1)

        inputs = processor(
            images=pil_image,
            input_points=points_tensor,
            input_labels=labels_tensor,
            return_tensors="pt",
            # dtype=torch.bfloat16,
        ).to(device, dtype=torch.bfloat16)

        # Get predictions
        # Options for outputs are found here: https://huggingface.co/docs/transformers/en/model_doc/sam2#multiple-points-for-refinement
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        # Get logits
        logits = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"], # SAM1 only
            binarize=False
        )[0]

        # Get masks
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
            binarize=True
        )[0]

        def plot_results(pil_image, points_tensor, masks, logits):
            # Convert PIL image to numpy array
            image_np = np.array(pil_image)


            # Squeeze and detach tensors for plotting
            logits_np = logits.to(torch.float32).squeeze().cpu().numpy()
            masks_np = masks.squeeze().cpu().numpy()

            # Create figure with 4 subplots
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            # 1. Original image with point prompts
            axes[0].imshow(image_np)
            points = points_tensor.squeeze().cpu().numpy()  # Shape: [10, 2]
            axes[0].scatter(points[:, 0], points[:, 1], c='red', s=50)
            axes[0].set_title('Original Image with Prompts')
            axes[0].axis('off')

            # 2. Logits
            axes[1].imshow(logits_np, cmap='viridis')
            axes[1].set_title('Logits')
            axes[1].axis('off')

            # 3. Mask
            axes[2].imshow(masks_np, cmap='gray')
            axes[2].set_title('Mask')
            axes[2].axis('off')

            # 4. Masks overlaid on the original image
            overlay = image_np.copy()
            overlay[masks_np > 0.5] = [255, 0, 0]  # Overlay mask in red
            axes[3].imshow(overlay)
            axes[3].set_title('Mask Overlay')
            axes[3].axis('off')

            plt.tight_layout()
            plt.savefig('doc/sam_results.png', dpi=200, bbox_inches='tight')
            plt.close(fig)

        # plot_results(pil_image, points_tensor, masks, logits)


        all_outputs[frame_idx] = {
            'logits': logits, # Store full logits tensor for all objects
            'masks': masks,
            'points': points,
            'labels': labels,
        }

    # For visualiztaion, combine logits across objects as before
    # Stack all logits into a single tensor (shape: [N, H, W])
    logits_stack = torch.stack([all_outputs[i]['logits'] for i in sorted(all_outputs.keys())], dim=0)

    # Apply sigmoid to the entire stack at once
    probs_stack = torch.sigmoid(logits_stack)

    print(f"[INFO] Tracked seaweed objects through {len(all_outputs):,} frames")

    return video_frames, probs_stack, all_outputs

def segment_frames_sam2(data_dir, model_name, max_frames=None):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Initialize predictor
    model = Sam2Model.from_pretrained(model_name).to(device, dtype=torch.bfloat16)
    processor = Sam2Processor.from_pretrained(model_name)

    # Load your video frames
    video_frames = load_video_frames(data_dir)

    # Slice video_frames if max_frames is specified
    video_frames_inference = video_frames if max_frames is None else video_frames[:max_frames]

    # Collect logits for all frames
    all_outputs = {}

    # Segment each image individually (i.e., not treated as video for SAM2)
    for frame_idx, pil_image in enumerate(video_frames_inference):
        # Create prompts
        points, labels = create_luminance_prompts(pil_image, num_prompts=5, luminance_percentile=10)

        # # Convert to tensor and explicitly cast to bfloat16
        # import torchvision.transforms.functional as TF
        # image_tensor = TF.to_tensor(pil_image).to(torch.bfloat16)

        # Convert outputs into SAM1 format
        points_tensor = torch.tensor(points).view(1, 1, -1, 2)
        labels_tensor = torch.tensor(labels).view(1, 1, -1)

        inputs = processor(
            images=pil_image,
            input_points=points_tensor,
            input_labels=labels_tensor,
            return_tensors="pt",
            # dtype=torch.bfloat16,
        ).to(device, dtype=torch.bfloat16)

        # Get predictions
        # Options for outputs are found here: https://huggingface.co/docs/transformers/en/model_doc/sam2#multiple-points-for-refinement
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        # Get logits
        logits = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"], # SAM1 only
            binarize=False
        )[0]

        # Get masks
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
            # inputs["reshaped_input_sizes"],
            binarize=True
        )[0]

        def plot_results(pil_image, points_tensor, masks, logits):
            # Convert PIL image to numpy array
            image_np = np.array(pil_image)


            # Squeeze and detach tensors for plotting
            logits_np = logits.to(torch.float32).squeeze().cpu().numpy()
            masks_np = masks.squeeze().cpu().numpy()

            # Create figure with 4 subplots
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            # 1. Original image with point prompts
            axes[0].imshow(image_np)
            points = points_tensor.squeeze().cpu().numpy()  # Shape: [10, 2]
            axes[0].scatter(points[:, 0], points[:, 1], c='red', s=50)
            axes[0].set_title('Original Image with Prompts')
            axes[0].axis('off')

            # 2. Logits
            axes[1].imshow(logits_np, cmap='viridis')
            axes[1].set_title('Logits')
            axes[1].axis('off')

            # 3. Mask
            axes[2].imshow(masks_np, cmap='gray')
            axes[2].set_title('Mask')
            axes[2].axis('off')

            # 4. Masks overlaid on the original image
            overlay = image_np.copy()
            overlay[masks_np > 0.5] = [255, 0, 0]  # Overlay mask in red
            axes[3].imshow(overlay)
            axes[3].set_title('Mask Overlay')
            axes[3].axis('off')

            plt.tight_layout()
            plt.savefig('doc/sam_results.png', dpi=200, bbox_inches='tight')
            plt.close(fig)

        # plot_results(pil_image, points_tensor, masks, logits)


        all_outputs[frame_idx] = {
            'logits': logits, # Store full logits tensor for all objects
            'masks': masks,
            'points': points,
            'labels': labels,
        }

    # For visualiztaion, combine logits across objects as before
    # Stack all logits into a single tensor (shape: [N, H, W])
    logits_stack = torch.stack([all_outputs[i]['logits'] for i in sorted(all_outputs.keys())], dim=0)

    # Apply sigmoid to the entire stack at once
    probs_stack = torch.sigmoid(logits_stack)

    print(f"[INFO] Tracked seaweed objects through {len(all_outputs):,} frames")

    return video_frames, probs_stack, all_outputs