"""Script for prompting SAM2 for video segmentation """

import os
import torch
from PIL import Image
from transformers import Sam2VideoModel, Sam2VideoProcessor

# Custom imports
from utils import visualize_and_save

def load_video_frames(frame_dir):
    """Load video frames from a directory as a list of PIL Images."""
    frame_paths = sorted([
        os.path.join(frame_dir, fname)
        for fname in os.listdir(frame_dir)
        if fname.endswith('.jpg')
    ])
    return [Image.open(fp) for fp in frame_paths]

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
        video=video_frames,
        inference_device=device,
        dtype=torch.bfloat16,
    )

    # Add click on first frame to select object
    ann_frame_idx = 0
    ann_obj_id = [2, 3]
    points = [[[[180, 350]], [[880, 250]]]]
    labels = [[[1], [1]]]

    # Experiment with negative prompts for background
    # ann_obj_id = [2, 3]
    # points = [[[[210, 900]], [[250, 600]]]]  # Points for two objects
    # labels = [[[0], [0]]]

    ## TODO: REMOVE THIS CODE ##
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.imshow(video_frames[0])

    # Loop through the points with the correct indexing
    for obj_points in points:
        for point_list in obj_points:
            for point in point_list:
                plt.gca().add_patch(Circle((point[0], point[1]), 20, color='green', fill=True))
                plt.gca().add_patch(Circle((point[0], point[1]), 20, color='white', fill=False, lw=2))

    plt.title('Image with Point Annotations')
    plt.axis('off')
    plt.savefig("doc/prompt_test.png")
    plt.show()

    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=ann_frame_idx,
        obj_ids=ann_obj_id,
        input_points=points,
        input_labels=labels,
    )

    # Segment the object on the first frame
    outputs = model(
        inference_session=inference_session,
        frame_idx=ann_frame_idx
    )

    video_res_masks = processor.post_process_masks(
        [outputs.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
        )[0]
    
    print(f"[INFO] Segmentation shape: {video_res_masks.shape}")

    # Visualize first frame
    visualize_and_save(data_dir, video_frames, points, video_res_masks, ann_frame_idx)

    # Propogate through the entire video
    video_segments = {}

    for sam2_video_output in model.propagate_in_video_iterator(inference_session):
        video_res_masks = processor.post_process_masks(
            [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
            )[0]
        video_segments[sam2_video_output.frame_idx] = {
            obj_id: video_res_masks[i]
            for i, obj_id in enumerate(inference_session.obj_ids)
        }
        # video_segments[sam2_video_output.frame_idx] = video_res_masks # <= Single object prompt
    
    print(f"[INFO] Tracked {len(inference_session.obj_ids):,} objects through {len(video_segments):,} frames")


if __name__ == "__main__":
    main(data_dir=r"data/Ulva_05_1_trial2", model="facebook/sam2-hiera-large")