"""Script for prompting SAM1/SAM2 for image segmentation """

import torch
from transformers import SamModel, SamProcessor
from typing import List, Dict, Tuple, Any
import numpy as np

# Custom imports
from src.data_utils import create_luminance_prompts

""" Used for semantic/instance segmentation on frame-level data """
def segment_frames_sam1(
        frames: List[np.ndarray], 
        model_name: str,
        num_prompts: int = 5, 
        luminance_percentile: int = 10
    ) -> Tuple[
    List[np.ndarray],         # List of input frames
    torch.Tensor,             # Stacked probability maps (shape: [N, 1, H, W], dtype=torch.bfloat16)
    Dict[int, Dict[str, Any]] # Dictionary of per-frame outputs
    ]:
    """Segment frames using SAM1 (Segment Anything Model) with luminance-based prompts."""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Initialize model
    model = SamModel.from_pretrained(model_name).to(device, dtype=torch.bfloat16)
    processor = SamProcessor.from_pretrained(model_name)

    # Collect logits for all frames
    frame_outputs = {}

    # Segment each image individually (i.e., not treated as video for SAM2)
    for frame_idx, frame in enumerate(frames):
        # Create prompts
        prompt_coordinates, prompt_labels = create_luminance_prompts(
            frame,
            num_prompts=num_prompts, 
            luminance_percentile=luminance_percentile
        )

        # Convert outputs into SAM1 format
        points_tensor = torch.tensor(prompt_coordinates).view(1, 1, -1, 2)
        labels_tensor = torch.tensor(prompt_labels).view(1, 1, -1)

        inputs = processor(
            images=frame,
            input_points=points_tensor,
            input_labels=labels_tensor,
            return_tensors="pt",
        ).to(device, dtype=torch.bfloat16)

        # Get predictions
        # Options for outputs are found here: https://huggingface.co/docs/transformers/en/model_doc/sam2#multiple-points-for-refinement
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        # Post-process logits and masks
        logits = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"], # SAM1 only
            binarize=False
        )[0]

        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
            binarize=True
        )[0]

        # Store outputs for the current frame
        frame_outputs[frame_idx] = {
            'logits': logits, # Full logits tensor for all objects
            'masks': masks, # Binarized masks
            'points': prompt_coordinates,
            'labels': prompt_labels,
        }

    # Stack logits across all frames into a single tensor (shape: [N, H, W])
    logits_stack = torch.stack(
        [frame_outputs[i]['logits'] for i in sorted(frame_outputs.keys())], 
        dim=0
    )

    # Convert logits to probabilities
    probs_stack = torch.sigmoid(logits_stack)

    print(f"[INFO] Segmented seaweed in {len(frame_outputs):,} frames")

    return frames, probs_stack, frame_outputs