import pandas as pd
import os
from typing import Dict, List, Tuple
import json
import numpy

# Custom imports
from src.sam_prompter import segment_frames_sam1
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

    # Validation-only
    true_density: float = None,

    # Prompt generation
    num_prompts: int = 5,
    luminance_percentile: int = 10,
    
    # Plotting
    output_folder: str = "data/processed",
    save_files: bool = False
) -> pd.DataFrame:
    """

    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    output_csv = os.path.join(output_folder, "predictions.csv") # Create output .csv

    # Hard-code ROI based on the original footage
    roi_width, roi_height = 408, 1012

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
            # cycle_df['density'] = extract_density_from_path(cycle_name) # Not known for predictions
            cycle_df['cycle'] = cycle_name[-1] # cycle = revolution

            # Prompt SAM1 for semantic segmentation per-frame
            video_frames, probs_stack, sam_outputs = segment_frames_sam1(
                frames, 
                model_name, 
                num_prompts=num_prompts, 
                luminance_percentile=luminance_percentile
            )

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
        # processed_data.to_csv(output_csv, index=False)
        # print(f"[INFO] Saved processed data to: {output_csv}")

    # Perform predictions
    model_path = "models/regression_results.json"
    with open(model_path, "r") as f:
        models = json.load(f)
        FEATURE = "surface_area_pct"

    # Aggregate data into per-cycle
    df_cycle = processed_data.groupby(
        ["cycle"],
        as_index=False
    ).mean(numeric_only=True)

    # Store predictions
    results = []

    for _, row in df_cycle.iterrows():

        cycle_id = row["cycle"]
        x = row[FEATURE]
        print(f"[INFO] Value for {FEATURE} = {x:.2f}")

        predictions = {"cycle": cycle_id}

        # LINEAR MODEL
        # density = a + b * x
        if (
            FEATURE in models and
            "per_cycle" in models[FEATURE] and
            "linear" in models[FEATURE]["per_cycle"]
        ):

            model = models[FEATURE]["per_cycle"]["linear"]

            b0 = model["params"]["const"]
            b1 = model["params"][FEATURE]

            predictions["linear_preds"] = float(b0 + b1 * x)

        # LOG-LINEAR MODEL
        # density = exp(a + b * x)
        if (
            FEATURE in models and
            "per_cycle" in models[FEATURE] and
            "loglinear" in models[FEATURE]["per_cycle"]
        ):
            # Extract log-linear model for the given feature
            model = models[FEATURE]["per_cycle"]["loglinear"]

            # Extract model parameters
            b0 = model["params"]["const"]
            b1 = model["params"][FEATURE]
            smearing_factor = model["smearing_factor"] # Duan's smearing estimator

            # Get predictions
            y_pred_log = b0 + b1 * x
            predictions["loglinear_preds"] = np.exp(y_pred_log) * smearing_factor

        results.append(predictions)

        print(f"[INFO] Cycle {cycle_id}")
        print(f"[INFO] Linear: {predictions.get('linear_preds', None):.2f} g/L")
        print(f"[INFO] Log-linear: {predictions.get('loglinear_preds', None):.2f} g/L")

    # Summary statistics
    print(f"\nResults for {video_path} with known density of {true_density}")
    print("-" * 60)
    print(
        f"{'Cycle':<8}"
        f"{'Linear ŷ':>12}"
        f"{'Error (ŷ - y)':>12}"
        f"{'Log-linear ŷ':>15}"
        f"{'Error (ŷ - y)':>12}"
    )
    print("-" * 60)

    preds_linear = np.array([r["linear_preds"] for r in results])
    preds_loglinear = np.array([r["loglinear_preds"] for r in results])

    for r in results:
        print(
            f"{r['cycle']:<8}"
            f"{r['linear_preds']:>12.2f}"
            f"{r['linear_preds'] - true_density:>+12.2f}"
            f"{r['loglinear_preds']:>15.2f}"
            f"{r['loglinear_preds'] - true_density:>+12.2f}"
        )

    print("-" * 60)
    print(
        f"{'Average':<8}"
        f"{np.mean(preds_linear):>12.2f}"
        f"{np.mean(preds_linear - true_density):>+12.2f}"
        f"{np.mean(preds_loglinear):>15.2f}"
        f"{np.mean(preds_loglinear - true_density):>+12.2f}"
    )

    print(
        f"{'Std dev':<8}"
        f"{np.std(preds_linear):>12.2f}"
        f"{np.std(preds_linear - true_density):>12.2f}"
        f"{np.std(preds_loglinear):>15.2f}"
        f"{np.std(preds_loglinear - true_density):>12.2f}"
    )

    print("\n[INFO] Pipeline completed successfully!")
    return processed_data

# Input videos paths
# Second argument are relevant frame ranges to keep for each trial
# This splits the recording of one biomass density level, into triplicate measurements
video_configs = {
    r"data/validation_footage/Google_Pixel9a_4_04gl.mp4": [(4943, 9232), (9351, 14473), (14562, 18941)], # 4.04 g/L
    r"data/validation_footage/Samsung_S23Ultra_4_04gl.mp4": [(4140, 9300), (9390, 12990), (14010, 18240)], # 4.04 g/L

    }

ulva_analysis_pipeline(
    # Pre-processing
    video_configs = video_configs,
    frame_interval_seconds = 5, # Take a frame every 5 seconds
    
    # Segmentation
    model_name = "facebook/sam-vit-huge", # SAM1
    conf_threshold = 0.5,

    # Point prompt generation
    num_prompts = 5,
    luminance_percentile  = 10,

    # Validation-only
    true_density = 4.04,
    
    # Plotting
    output_folder = "data/processed",
    save_files = False
)
