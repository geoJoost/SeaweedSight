import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple, Any

# Custom imports
from src.sam_prompter import segment_frames_sam1, segment_frames_sam2
from src.data_utils import extract_density_from_path, calculate_surface_area, extract_color_features
from src.visualization_utils import visualize_sam_segmentation, plot_density_examples, plot_all_predictors, plot_select_predictors
from src.statistics import analyze_feature_relationships
from src.video_clipping import *

def ulva_analysis_pipeline(
    # Pre-processing
    video_configs: Dict[str, List[Tuple[int, int]]],
    frame_interval_seconds: float = 5.0,
    
    # Segmentation
    model_name: str = "facebook/sam-vit-huge",
    conf_threshold: float = 0.5,

    # Prompt generation
    num_prompts: int = 5,
    luminance_percentile: int = 10,
    
    # Plotting
    output_folder: str = "data/processed",
    save_files: bool = False
) -> pd.DataFrame:
    """
    End-to-end pipeline for Ulva spp. video analysis:
    1. Finds the smallest ROI across al videos for clipping.
    2. Extracts relevant frames from videos.
    3. Semantic segmentation using SAM and extract features (RGB, CIELAB, surface area).
    5. Analyzes results by fitting linear and power regressions.
    6. Plots example frames at different biomass densities.

    Args:
        video_configs: Dictionary mapping video paths to frame ranges to keep.
        sam_model: SAM model name (e.g., 'facebook/sam-vit-huge').
        frame_interval_seconds: Interval for frame extraction (default: 5.0s -> 0.2fps).
        confidence_threshold: Confidence threshold for segmentation.
        num_prompts: Number of SAM point prompts per frame.
        luminance_percentile: Percentile for luminance thresholding.
        output_dir: Base directory for all outputs.
        save_files: If True, saves extracted frames to disk.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    output_csv = os.path.join(output_folder, "ulva_processed_data.csv") # Create output .csv

    # Find ROI
    print("[INFO] Step 1/4: Determining ROI and extracting frames...")
    roi_width, roi_height = find_smallest_roi(video_configs)

    # Split biomass video into multiple cycles/revolutions, corresponding to when duck passed underneath the camera
    # And take relevant frames to prevent segmenting same object multiple times
    all_extracted_frames = {}
    for video_path, frame_ranges in video_configs.items():
        print(f"[INFO] Processing {video_path} with frame keep ranges: {frame_ranges}")

        # Process .avi
        extracted_frames = extract_relevant_frames(
            # .avi or .mp4 file
            video_path,

            # Frame specificatrions
            frame_interval_seconds=frame_interval_seconds, # 0.2 fps
            frame_ranges=frame_ranges, 
            roi=(roi_width, roi_height),

            # Save frames
            save_frames=save_files
        )
        all_extracted_frames.update(extracted_frames)

    # Analyze footage
    print("\n[INFO] Step 2/4: Segmenting frames and extracting features...")
    if not os.path.exists(output_csv):
        # Initialize a list to hold all DataFrames
        cycle_dataframes = []

        # Process each directory
        for cycle_name, frames in all_extracted_frames.items():
            print(f"{'-' * 50}")
            print(f"[INFO] Processing cycle: {cycle_name} using SAM")

            # Get dimensions of first frame to calculate total pixels
            w, h, c = frames[0].shape
            total_pixels = w * h

            # Initialize DataFrame for this directory
            cycle_df = pd.DataFrame()
            cycle_df['frame_id'] = list(range(len(frames)))
            cycle_df['model_name'] = model_name
            cycle_df['conf_threshold'] = conf_threshold
            cycle_df['num_prompts'] = num_prompts
            cycle_df['luminance_percentile'] = luminance_percentile
            cycle_df['density'] = extract_density_from_path(cycle_name)
            cycle_df['cycle'] = cycle_name[-1] # Takes number from names like 'Ulva_05_1_cycle2'

            # Prompt SAM1 for semantic segmentation per-frame
            video_frames, probs_stack, sam_outputs = segment_frames_sam1(
                frames, 
                model_name, 
                num_prompts=num_prompts, 
                luminance_percentile=luminance_percentile
            )

            # UNUSED #
            # Prompt SAM2 for semantic segmentation per-frame
            # video_frames, probs_stack, sam_outputs = segment_frames_sam2(frames, model_name, num_prompts=5, luminance_percentile=10)

            # Pre-allocate lists for all features
            feature_keys = ['surface_area', 'mean_R', 'mean_G', 'mean_B', 'mean_L', 'mean_a', 'mean_b']
            feature_data = {key: [] for key in feature_keys}

            # Propagate over the frames
            for frame_idx, frame_probs in enumerate(probs_stack):
                # Calculate surface area (in px)
                surface_area, binarized_mask = calculate_surface_area(frame_probs.squeeze(), conf_threshold)

                # Extract color features for the current frame
                color_features = extract_color_features(video_frames[frame_idx], binarized_mask)

                # Append all features to lists
                feature_data['surface_area'].append(surface_area)
                for key in ['mean_R', 'mean_G', 'mean_B', 'mean_L', 'mean_a', 'mean_b']:
                    feature_data[key].append(color_features[key])

                # Save visualization for each frame
                if save_files:
                    # Unpack prompts for this frame
                    frame_points = sam_outputs[frame_idx]['points']
                    # current_labels = sam_outputs[frame_idx]['labels'] # Not used
                    # current_logits = sam_outputs[frame_idx]['logits'] # Not used
                    frame_masks = sam_outputs[frame_idx]['masks']

                    # Visualize frame
                    visualize_sam_segmentation(
                        cycle_name,
                        video_frames,
                        frame_points,
                        frame_probs,
                        frame_masks,
                        frame_idx=frame_idx,
                        data_dir=cycle_name,
                        output_folder=output_folder,
                        conf_threshold=conf_threshold
                    )

            # Assign all data to the DataFrame at once
            for key in feature_keys:
                cycle_df[key] = feature_data[key]

            # Calculate cumulative surface area
            cycle_df['tot_surface_area'] = cycle_df['surface_area'].sum()

            # Calculate surface area percentage
            cycle_df['surface_area_pct'] = (cycle_df['surface_area'] / total_pixels) * 100

            # Append to the list of DataFrames
            cycle_dataframes.append(cycle_df)
            print(f"[INFO] Finished processing {cycle_name}")

        # Concatenate all cycle DataFrames into one
        processed_data = pd.concat(cycle_dataframes, ignore_index=True)

        # Save a single combined CSV
        processed_data.to_csv(output_csv, index=False)
        print(f"[INFO] Saved processed data to: {output_csv}")

    # Compute regression and statistics
    print("\n[INFO] Step 3/4: Fitting and plotting regressions...")
    analysis_df = pd.read_csv(output_csv)
    features = ['surface_area_pct', 'tot_surface_area', 'mean_R', 'mean_G', 'mean_B', 'mean_L', 'mean_a', 'mean_b']
    feature_names = ['Surface area [%]', 'Tot. surface area [px]', 'Red', 'Green', 'Blue', 'Luminance', 'a*', 'b*']

    # Combined regressors plot
    plot_all_predictors(analysis_df, features, feature_names, output_folder='doc/output')

    # Regressions but limited to surface area and RGB
    plot_select_predictors(analysis_df, output_folder='doc')

    # Regressors into own plots
    analyze_feature_relationships(
        analysis_df=analysis_df,
        feature_columns=features,
        output_folder='doc/output'
    )

    print("\n[INFO] Step 4/4: Plotting frame examples...")
    #  Plot random frames at different biomass densities (0.5, 2.0, 4.0 and 5.0 g/L)
    plot_density_examples(
        all_extracted_frames,
        model_name='facebook/sam-vit-huge',
        conf_threshold=0.5,
        num_prompts=5,
        luminance_percentile=10,
        output_folder="doc/output"
    )
    print("\n[INFO] Pipeline completed successfully!")
    return analysis_df

# Input videos paths
# Second argument are relevant frame ranges to keep for each trial
# This splits the recording of one biomass density level, into triplicate measurements
video_configs = {
    r"data/footage/Ulva_05_1_C.mp4": [(208, 3978), (4089, 9233), (9490, 13285)], # 0.5 g/L
    r"data/footage/Ulva_10_1_C.mp4": [(379, 4652), (4804, 8566), (8760, 12755)], # 1.0 g/L
    r"data/footage/Ulva_15_1_C.mp4": [(119, 2670), (2850, 5143), (5480, 7741)],
    r"data/footage/Ulva_20_3.avi": [(115, 2850), (2906, 5981), (6023, 8672)],
    r"data/footage/Ulva_25_3.avi": [(205, 2312), (2342, 4682), (4724, 6936)],
    r"data/footage/Ulva_30_1.avi": [(120, 2777), (2816, 4931), (4967, 7585)],
    r"data/footage/Ulva_35_1.avi": [(108, 2546), (2587, 4952), (4994, 7295)],
    r"data/footage/Ulva_40_1.avi": [(357, 2769), (2826, 5357), (5390, 7672)],
    r"data/footage/Ulva_45_1.avi": [(114, 2508), (2542, 5027), (5056, 7710)],
    r"data/footage/Ulva_50_1.avi": [(358, 2929), (3016, 5350), (5400, 7976)], # 5.0 g/L
    }

ulva_analysis_pipeline(
    # Pre-processing
    video_configs = video_configs,
    frame_interval_seconds = 5, # Take a frame every 5 seconds
    
    # Segmentation
    # model_name="facebook/sam2.1-hiera-large", # SAM2
    model_name = "facebook/sam-vit-huge", # SAM1
    conf_threshold = 0.5,

    # Point prompt generation
    num_prompts = 5,
    luminance_percentile  = 10,
    
    # Plotting
    output_folder = "data/processed",
    save_files = True
)
