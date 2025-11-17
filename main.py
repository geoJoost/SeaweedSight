import numpy as np
import pandas as pd
import os
import glob

# Custom imports
from src.sam_prompter import prompt_sam2, segment_frames_sam2
from src.data_utils import get_frame_paths, extract_density_from_dir, calculate_surface_area, extract_color_features
from src.visualization_utils import visualize_sam2_outputs, visualize_features
from src.data_exploration import plot_features_vs_density

def process_video_directory(
    data_dirs,
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
    for data_dir in data_dirs:
        print(f"[INFO] Processing directory: {data_dir}")

        # Initialize DataFrame for this directory
        df = pd.DataFrame()
        df['frame_path'] = get_frame_paths(data_dir)[:max_frames] if max_frames else get_frame_paths(data_dir)
        df['frame_id'] = df['frame_path'].apply(lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        df[['model_name', 'conf_threshold']] = model_name, conf_threshold
        df['density'] = extract_density_from_dir(data_dir)

        # Prompt SAM2 for video processing
        # video_frames, probs_stack, all_outputs = prompt_sam2(data_dir, model_name, max_frames)

        # Prompt SAM2 for semantic segmentation per-frame
        video_frames, probs_stack, all_outputs = segment_frames_sam2(data_dir, model_name, max_frames)

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
                data_dir,
                video_frames,
                current_points,
                probs,
                frame_idx=frame_idx,
                data_dir=data_dir,
                output_folder=output_folder,
                conf_threshold=conf_threshold
            )

        # Assign all data to the DataFrame at once
        for key in feature_keys:
            df[key] = feature_data[key]

        # Calculate cumulative surface area
        df['cumulative_surface_area'] = np.cumsum(df['surface_area'])

        # Create visualization of features over entire timeseries
        visualize_features(df, conf_threshold, data_dir, output_dir=output_folder)

        # Save .csv for downstream processes
        output_path = f"{os.path.basename(data_dir)}_processed.csv"
        df.to_csv(os.path.join(output_folder, output_path), index=False)

        print(f"[INFO] Saved processed data for {data_dir} to: {output_path}")
    
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


# Define data folders
data_dirs = [
    "data/Ulva_05_1_trial1",
    # "data/Ulva_05_1_trial2",
    # "data/Ulva_05_1_trial3",
    "data/Ulva_10_1_trial1",
    # "data/Ulva_10_1_trial2",
    # "data/Ulva_10_1_trial3",
    "data/Ulva_15_1_trial1",
    "data/Ulva_20_3_trial1",
    "data/Ulva_25_3_trial1"
]

process_video_directory(
    data_dirs=data_dirs,
    model_name="facebook/sam2-hiera-base-plus",
    # model_name='facebook/sam-vit-huge',
    conf_threshold=0.5,
    output_folder="data/processed",
    max_frames=30,
)
