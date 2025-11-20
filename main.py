import numpy as np
import pandas as pd
import os
import glob

# Custom imports
from src.sam_prompter import prompt_sam2, segment_frames_sam2, segment_frames_sam1
from src.data_utils import get_frame_paths, extract_density_from_dir, calculate_surface_area, extract_color_features
from src.visualization_utils import visualize_sam2_outputs, visualize_features
from src.data_exploration import plot_features_vs_density
from src.video_clipping import *

def process_video_directory(
    # data_dirs,
    trial_frames_dict,
    model_name="facebook/sam2-hiera-base-plus",
    conf_threshold=0.5,
    output_folder="data/processed",
    max_frames=None,
):
    """
    Process video frames from multiple directories, extract features, and visualize results.

    Args:
        data_dirs (list): List of directories containing video frames.
        model_name (str): Name of the SAM2 model to use.
        conf_threshold (float): Confidence threshold for segmentation.
        output_folder (str): Directory to save outputs (CSV, plots, etc.).
        max_frames (int): Maximum number of frames to process per directory (optional).
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process each directory
    # for data_dir in data_dirs:
    for trial_name, frames in trial_frames_dict.items():
        print(f"[INFO] Processing trial:: {trial_name}")

        # Initialize DataFrame for this directory
        df = pd.DataFrame()
        # df['frame_path'] = get_frame_paths(data_dir)[:max_frames] if max_frames else get_frame_paths(data_dir)
        # df['frame_id'] = df['frame_path'].apply(lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        df['frame_id'] = list(range(len(frames)))
        df[['model_name', 'conf_threshold']] = model_name, conf_threshold
        df['density'] = extract_density_from_dir(trial_name)

        # Prompt SAM2 for video processing
        # video_frames, probs_stack, all_outputs = prompt_sam2(data_dir, model_name, max_frames, num_prompts=5, luminance_percentile=10)

        # Prompt SAM2 for semantic segmentation per-frame
        # video_frames, probs_stack, all_outputs = segment_frames_sam2(frames, model_name, max_frames, num_prompts=5, luminance_percentile=10)

        # Prompt SAM1 for semantic segmentation per-frame
        video_frames, probs_stack, all_outputs = segment_frames_sam1(frames, model_name, max_frames, num_prompts=5, luminance_percentile=15)

        # Pre-allocate lists for all features
        feature_keys = ['surface_area', 'mean_R', 'mean_G', 'mean_B', 'mean_L', 'mean_a', 'mean_b']
        feature_data = {key: [] for key in feature_keys}

        # Propagate over the frames
        for frame_idx, probs in enumerate(probs_stack):
            # Unpack prompts for this frame
            current_points = all_outputs[frame_idx]['points']
            current_labels = all_outputs[frame_idx]['labels']
            # current_logits = all_outputs[frame_idx]['logits']
            # current_masks = all_outputs[frame_idx]['masks'][frame_idx] # Second idx corresponds to the objectID. No tracking in current implementation

            # Calculate surface area (in px)
            surface_area, binarized_mask = calculate_surface_area(probs.squeeze(), conf_threshold)

            # Extract color features for the current frame
            color_features = extract_color_features(video_frames[frame_idx], binarized_mask)

            # Append all features to lists
            feature_data['surface_area'].append(surface_area)
            for key in ['mean_R', 'mean_G', 'mean_B', 'mean_L', 'mean_a', 'mean_b']:
                feature_data[key].append(color_features[key])

            # Save visualization for each frame
            # Save visualization for each frame
            visualize_sam2_outputs(
                trial_name,
                video_frames,
                current_points,
                probs,
                frame_idx=frame_idx,
                data_dir=trial_name,
                output_folder=output_folder,
                conf_threshold=conf_threshold
            )

        # Assign all data to the DataFrame at once
        for key in feature_keys:
            df[key] = feature_data[key]

        # Calculate cumulative surface area
        df['cumulative_surface_area'] = np.cumsum(df['surface_area'])

        # Create visualization of features over entire timeseries
        visualize_features(df, conf_threshold, trial_name, output_dir=output_folder)

        # Save .csv for downstream processes
        output_path = f"{os.path.basename(trial_name)}_processed.csv"
        df.to_csv(os.path.join(output_folder, output_path), index=False)

        print(f"[INFO] Saved processed data for {trial_name} to: {output_path}")
    
    # Combine all CSV's
    csv_files = glob.glob(os.path.join(output_folder,'*.csv'))
    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

    # Compute correlation
    plot_features_vs_density(
        df=df,
        features=['surface_area', 'mean_R', 'mean_G', 'mean_B', 'mean_L', 'mean_a', 'mean_b'],
        output_folder=output_folder
    )

    print('[INFO] Code finished...')

## Video to frames preperation ##
video_configs = {
    r"data/Ducks/Ulva_05_1.avi": [(204, 3980), (4090, 9236), (9489, 13285)],
    r"data/Ducks/Ulva_10_1.avi": [(339, 4176), (4313, 7691), (7865, 11453)],
    r"data/Ducks/Ulva_15_1.avi": [(119, 2670), (2850, 5143), (5480, 7741)],
    r"data/Ducks/Ulva_20_3.avi": [(115, 2850), (2906, 5981), (6023, 8672)],
    r"data/Ducks/Ulva_25_3.avi": [(205, 2312), (2342, 4682), (4724, 6936)],
    r"data/Ducks/Ulva_30_1.avi": [(120, 2777), (2816, 4931), (4967, 7585)],
    r"data/Ducks/Ulva_35_1.avi": [(108, 2546), (2587, 4952), (4994, 7295)],
    r"data/Ducks/Ulva_40_1.avi": [(357, 2769), (2826, 5357), (5390, 7672)],
    r"data/Ducks/Ulva_45_1.avi": [(114, 2508), (2542, 5027), (5056, 7710)],
    r"data/Ducks/Ulva_50_1.avi": [(358, 2929), (3016, 5350), (5400, 7976)],

    }

# Find the smallest ROI
roi_width, roi_height = find_smallest_roi(video_configs)

# Get normalization statistics for colour correction
# normalization_area = (150, 35, 100, 50) # Small square in center-top of the image
normalization_area = (0, 100, 400, 800) # Main ROI covering 95% of entire frame (except sides)
master_mean, master_std = get_master_stats(r"data/Ducks/Ulva_05_1.avi", roi_width, roi_height, normalization_area)

# Split video into individual trials
all_trial_frames = {}
for input_video, keep_ranges in video_configs.items():
    print(f"[INFO] Processing {input_video} with keep ranges: {keep_ranges}")

    # Process .avi
    trial_frames = process_video_n_frames(input_video, # .avi file
                           # Frame specificatrions
                           seconds_interval=8.0,
                           keep_ranges=keep_ranges, 
                           roi=(roi_width, roi_height),

                           # Normalization parameters
                           normalize=False,
                           normalization_area=normalization_area,
                           master_mean=master_mean,
                           master_std=master_std,

                           # Save frames
                           save_files=True
                           )
    all_trial_frames.update(trial_frames)

## Semantic segmentation ##
process_video_directory(
    trial_frames_dict=all_trial_frames,
    model_name="facebook/sam2.1-hiera-large", # UNUSED FOR SAM1
    # model_name='facebook/sam-vit-huge',
    conf_threshold=0.5,
    output_folder="data/processed",
    max_frames=50,
)
