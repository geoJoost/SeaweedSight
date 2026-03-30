import cv2
import numpy as np
import os
import random
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from typing import List, Tuple, Dict

# Custom imports
from src.statistics import fit_power_law_regression

def visualize_luminance_prompts(
        frame, 
        luminance_channel, 
        dark_regions, 
        points, 
        luminance_threshold, 
        output_path="doc/prompt_experiment_luminance.png"
    ) -> str:
    """ Visualize luminance-based prompt selection process """
    
    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    # Original frame with point prompts
    axes[0].imshow(frame)
    for point in points:
        axes[0].add_patch(Circle((point[0][0], point[0][1]), 10, color='green', fill=True))
        axes[0].add_patch(Circle((point[0][0], point[0][1]), 10, color='white', fill=False, lw=2))
    axes[0].set_title("Original frame with prompts")

    # Luminance channel
    axes[1].imshow(luminance_channel, cmap='gray')
    axes[1].set_title("Luminance (L)")

    # Dark regions with point prompts
    axes[2].imshow(dark_regions, cmap='gray')
    for point in points:
        axes[2].add_patch(Circle((point[0][0], point[0][1]), 10, color='green', fill=True))
    axes[2].set_title(f"Dark regions (L < {luminance_threshold})")

    # Connected components
    num_components, component_labels, _, _ = cv2.connectedComponentsWithStats(dark_regions.astype(np.uint8), connectivity=8)
    axes[3].imshow(component_labels, cmap='nipy_spectral')
    axes[3].set_title(f"Connected components (n={num_components})")

    # Add annotation
    fig.text(
        0.5, 0.01,
        f"Threshold: L < {luminance_threshold}\n",
        ha='center', fontsize=10
    )

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[INFO] Saved figure to: {output_path}")
    return output_path

def visualize_sam_segmentation(
        video_path: str, 
        frames: List[np.ndarray], 
        points: List[List[Tuple[int, int]]], 
        probs: torch.Tensor, 
        current_masks: torch.Tensor, 
        frame_idx: int, 
        data_dir, 
        output_folder, 
        conf_threshold=0.5
    ) -> str:
    """ Visualize SAM segmentation results for a single frame """
    
    # Create output directory
    output_dir = os.path.join(output_folder, f"{os.path.basename(data_dir)}_processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the frame
    frame = frames[frame_idx]

    # Combine masks for all objects into a single mask
    binarized_mask = (probs > conf_threshold).to(torch.uint8) * 255
    binarized_mask = binarized_mask.cpu().squeeze().numpy()

    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(12, 6))

    # Original image with point prompts
    axes[0].imshow(frame)
    for obj_points in points:
        for point in obj_points:
            plt.gca().add_patch(Circle((point[0], point[1]), 20, color='green', fill=True))
            plt.gca().add_patch(Circle((point[0], point[1]), 20, color='white', fill=False, lw=2))
    axes[0].set_title('Image with prompt')
    axes[0].axis('off')

    # Probability map
    axes[1].imshow(probs.cpu().squeeze().to(torch.float32), cmap='viridis', vmin=0, vmax=1.0)
    axes[1].set_title('Confidence')
    axes[1].axis('off')

    # Binarized mask
    axes[2].imshow(current_masks.squeeze(), cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    # Binarized mask overlaid on the original frame
    axes[3].imshow(frame)
    axes[3].imshow(binarized_mask, alpha=0.3, cmap='gray')
    axes[3].set_title('Output')
    axes[3].axis('off')

    plt.tight_layout()

    # Save the figure
    basename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, f"{basename}_{frame_idx:06d}.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()

    print(f"[INFO] Saved image to: {output_path}")
    return output_path

def plot_density_examples(
        all_extracted_frames: Dict[str, List[np.ndarray]],
        model_name: str, 
        conf_threshold: float = 0.5,
        num_prompts: int = 5, 
        luminance_percentile: int = 10, 
        output_folder: str = "doc"
    ) -> None:
    """
    Plot random frames, probability maps, and segmentation masks for selected biomass densities.
    Each row represents a biomass density level, with columns showing:
    1. Original frame with SAM point prompts
    2. Probability map
    3. Binary segmentation mask.
    """
    # Custom imports
    from src.data_utils import extract_density_from_path
    from src.sam_prompter import segment_frames_sam1

    # Create output directory if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Define target densities and colormap
    # target_densities = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    target_densities = [0.5, 2.0, 4.0, 5.0]
    binary_cmap = ListedColormap(['#FCF3EE', '#68000D'])

    # Filter and sort cycles by target densities
    filtered_cycles = [
        cycle_name for cycle_name in all_extracted_frames.keys()
        if extract_density_from_path(cycle_name) in target_densities
    ]
    filtered_cycles.sort(key=lambda x: extract_density_from_path(x))

    # Select one random cycle per density
    selected_cycles = {}
    for cycle_name in filtered_cycles:
        density = extract_density_from_path(cycle_name)
        if density not in selected_cycles:
            selected_cycles[density] = []
        selected_cycles[density].append(cycle_name)

    # Keep only one cycle per density
    for density in selected_cycles:
        selected_cycles[density] = selected_cycles[density][:1]

    # Create figure with subplots
    nrows = len(target_densities)
    fig, axes = plt.subplots(nrows, 3, figsize=(3.5, 2.85))  # Portrait A4 = (8.27, 11.69)

    # Set column titles
    for col, title in enumerate(['Image', 'Confidence', 'Mask'] * 1):
        axes[0, col].set_title(title, fontsize=8)

    # Process each biomass density
    for row_idx, density in enumerate(target_densities):
        cycle_names = selected_cycles.get(density, [])

        for cycle_idx, cycle_name in enumerate(cycle_names[:1]):
            frames = all_extracted_frames[cycle_name]
            if not frames:
                continue

            # Select a random frame
            random.seed(42)
            frame_idx = random.randint(0, len(frames) - 1)
            selected_frame = frames[frame_idx]

            # Prompt SAM for this frame
            _, probs, outputs = segment_frames_sam1(
                [selected_frame],
                model_name,
                num_prompts=num_prompts, 
                luminance_percentile=luminance_percentile
            )
            probs = probs[0].to(torch.float32)
            prompt_points = outputs[0]['points']
            masks = outputs[0]['masks']

            # To better use the space, we rotate everything by 90 degrees
            rotated_frame = np.rot90(selected_frame, k=1)
            rotated_probs = np.rot90(probs.squeeze().cpu().numpy())
            rotated_mask = np.rot90(masks.squeeze().cpu().numpy(), k=1)

            # Rotate prompt points to match image rotation
            rotated_points = [
                [(point[1], selected_frame.shape[1] - point[0]) for point in obj_points]
                for obj_points in prompt_points
            ]

            # Plot original frame with point prompts
            axes[row_idx, 0].imshow(rotated_frame)
            for obj_points in rotated_points:
                for point in obj_points:
                    axes[row_idx, 0].add_patch(
                        Circle((point[0], point[1]), 20, color='green', fill=True)
                    )
                    axes[row_idx, 0].add_patch(
                        Circle((point[0], point[1]), 20, color='white', fill=False, lw=1)
                    )
            
            # Add biomass density label to first column
            if cycle_idx == 0:
                axes[row_idx, 0].set_ylabel(f'{density} g L$^{{-1}}$', fontsize=8)

            # Remove ticks from all images
            for col_idx in range(3):
                axes[row_idx, col_idx].set_xticks([])
                axes[row_idx, col_idx].set_yticks([])

            # Plot probabilities
            axes[row_idx,1].imshow(rotated_probs, cmap='viridis', vmin=0, vmax=1.0)

            # Plot segmentation mask
            axes[row_idx, 2].imshow(rotated_mask, cmap=binary_cmap)

    # Adjust layout 
    plt.tight_layout()

    output_path = os.path.join(output_folder, "model_output.eps")
    plt.savefig(output_path, format='eps', bbox_inches='tight')
    plt.savefig(os.path.join(output_folder, 'model_output.png'), dpi=300, bbox_inches='tight')

    plt.close()

    print(f"[INFO] Saved biomass density examples to: {output_path}")

def plot_all_predictors(
        analysis_df: pd.DataFrame, 
        feature_columns: List[str], 
        feature_labels: List[str], 
        output_folder='doc'
    ) -> None:
    """
    Plot all regressions (per-frame/per-cycle, linear/log-linear) in a single figure.
    Uses the existing plot_regression logic but combines everything into one grid.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Prepare data for per-frame and per-cycle analyses
    df_per_frame = analysis_df.copy()
    df_per_revolution = analysis_df.groupby(['density', 'cycle'], as_index=False)[feature_columns].mean()

    fig, axes = plt.subplots(len(feature_columns), 4, figsize=(6, 9.3), sharex='row', sharey='col', constrained_layout=True)

    # Column configuration
    analysis_configs = [
        ('Frame', df_per_frame, 'Linear'),
        ('Revolution', df_per_revolution, 'Linear'),
        ('Frame', df_per_frame, 'Log-linear'),
        ('Revolution', df_per_revolution, 'Log-linear')
    ]

    col_titles = [
        'Frame | Linear',
        'Revolution | Linear',
        'Frame | Log-linear',
        'Revolution | Log-linear'
    ]

    colors = {
        'Frame | Linear': '#219ebc',
        'Revolution | Linear': '#606c38',
        'Frame | Log-linear': '#fb8500',
        'Revolution | Log-linear': '#8b5cf6'
    }

    # Column titles
    for ax, col_title in zip(axes[0], col_titles):
        ax.set_title(col_title, fontsize=8)#, pad=5)

    # Plot loop
    for i, (feature, feature_name) in enumerate(zip(feature_columns, feature_labels)):
        for j, (analysis_type, df_analysis, reg_type) in enumerate(analysis_configs):

            ax = axes[i, j]

            y = df_analysis['density']
            x = df_analysis[feature]

            ax.tick_params(axis='both', labelsize=8)

            # Disable invalid plots
            if "tot. surface area" in feature_name.lower() and j in (0, 2):
                ax.cla()
                ax.set_frame_on(True)
                ax.text(
                    0.5, 0.5,
                    "Revolution\nonly.",
                    ha="center", va="center",
                    fontsize=8,
                    transform=ax.transAxes
                )
                if j == 0:
                    ax.set_ylabel("Density [g L$^{-1}$]", fontsize=8)
                continue

            # Format large x-axis values
            if "tot. surface area" in feature_name.lower():
                ax.xaxis.set_major_formatter(mtick.EngFormatter())

            scatter_color = colors[col_titles[j]]

            # Linear plots
            if j < 2:
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()

                sns.regplot(
                    x=x, y=y, ax=ax, ci=None,
                    scatter_kws={'s': 5, 'alpha': 1.0, 'color': scatter_color},
                    line_kws={'color': '#000000', 'linewidth': 1}
                )

                ax.set_yticks([1, 2, 3, 4, 5])
                ax.set_ylim(0, 5.5)
                ax.set_xlabel(feature_name, fontsize=8)

            # Log-linear plots
            else:
                log_y = np.log(y)
                X = sm.add_constant(x)
                model = sm.OLS(log_y, X).fit()
                
                # Scatter (raw data)
                ax.scatter(x, y, s=5, alpha=1.0, color=scatter_color)

                # Create smooth x range
                x_fit = np.linspace(x.min(), x.max(), 100)

                # Predict in log-space, then transform back
                X_fit = sm.add_constant(x_fit)
                log_y_fit = model.predict(X_fit)
                y_fit = np.exp(log_y_fit)

                # Plot fitted curve
                ax.plot(x_fit, y_fit, color='#000000', linewidth=1)

                # Log scale axis
                ax.set_yscale('log')
                ax.set_ylim()
                                
                #ax.yaxis.set_major_formatter(mtick.LogFormatter())
                ax.set_xlabel(feature_name, fontsize=8)

            # Y-label
            if j == 0:
                ax.set_ylabel("Density [g L$^{-1}$]", fontsize=8)
            else:
                ax.set_ylabel(None)

    # Align layout
    fig.align_ylabels()

    # Save figure
    plt.savefig(os.path.join(output_folder, 'all_regressions.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_folder, 'all_regressions.eps'), bbox_inches='tight')
    plt.close()

    print(f"[INFO] Finished printing all regressions regression figure to {output_folder}")

def plot_select_predictors(
        analysis_df, 
        output_folder='doc'):
    """ Plot regressions for surface area and RGB. """
    # Select only the features of interest
    feature_columns = ['surface_area_pct', 'mean_R', 'mean_G', 'mean_B']
    feature_labels = ['Surface Area [%]', 'Red [-]', 'Green [-]', 'Blue [-]']

    # Prepare data for per-frame and per-cycle analyses
    df_per_frame = analysis_df.copy()
    df_per_revolution = analysis_df.groupby(['density', 'cycle'], as_index=False)[feature_columns].mean()

    # Create figure with shared y-axes
    fig, axes = plt.subplots(len(feature_columns), 4, figsize=(6, 6), sharex='row', sharey='col', constrained_layout=True)

    # Column configuration
    analysis_configs = [
        ('Frame', df_per_frame, 'Linear'),
        ('Revolution', df_per_revolution, 'Linear'),
        ('Frame', df_per_frame, 'Log-linear'),
        ('Revolution', df_per_revolution, 'Log-linear')
    ]


    # Define row-specific colors
    colors = {
        'Surface Area [%]': '#bcbcbc',
        'Red [-]': '#a72d2b',
        'Green [-]': '#3b892d',
        'Blue [-]': '#5381b1'
    }

    # Column titles
    col_titles = [
        'Frame | Linear',
        'Revolution | Linear',
        'Frame | Log-linear',
        'Revolution | Log-linear'
    ]

    # Set column titles
    for ax, col_title in zip(axes[0], col_titles):
        ax.set_title(col_title, fontsize=8, ha='center')

    # Plot each feature
    for i, (feature, feature_name) in enumerate(zip(feature_columns, feature_labels)):
        for j, (analysis_type, df_analysis, reg_type) in enumerate(analysis_configs):

            ax = axes[i, j]

            y = df_analysis['density']
            x = df_analysis[feature]

            ax.tick_params(axis='both', labelsize=8)

            # Disable invalid plots
            if "tot. surface area" in feature_name.lower() and j in (0, 2):
                ax.cla()
                ax.set_frame_on(True)
                ax.text(
                    0.5, 0.5,
                    "Revolution\nonly.",
                    ha="center", va="center",
                    fontsize=8,
                    transform=ax.transAxes
                )
                if j == 0:
                    ax.set_ylabel("Density [g L$^{-1}$]", fontsize=8)
                continue

            # Format large x-axis values
            if "tot. surface area" in feature_name.lower():
                ax.xaxis.set_major_formatter(mtick.EngFormatter())

            scatter_color = colors[feature_name]

            # Linear plots
            if j < 2:
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()

                sns.regplot(
                    x=x, y=y, ax=ax, ci=None,
                    scatter_kws={'s': 5, 'alpha': 1.0, 'color': scatter_color},
                    line_kws={'color': '#000000', 'linewidth': 1}
                )

                ax.set_yticks([1, 2, 3, 4, 5])
                ax.set_ylim(0, 5.5)
                ax.set_xlabel(feature_name, fontsize=8)

            # Log-linear plots
            else:
                log_y = np.log(y)
                X = sm.add_constant(x)
                model = sm.OLS(log_y, X).fit()

                # Scatter (raw data)
                ax.scatter(x, y, s=5, alpha=1.0, color=scatter_color)

                # Create smooth x range
                x_fit = np.linspace(x.min(), x.max(), 100)

                # Predict in log-space, then transform back
                X_fit = sm.add_constant(x_fit)
                log_y_fit = model.predict(X_fit)
                y_fit = np.exp(log_y_fit)

                # Plot fitted curve
                ax.plot(x_fit, y_fit, color='#000000', linewidth=1)

                # Log scale axis
                ax.set_yscale('log')
                ax.set_ylim()
                                
                #ax.yaxis.set_major_formatter(mtick.LogFormatter())
                ax.set_xlabel(feature_name, fontsize=8)

            # Y-label
            if j == 0:
                ax.set_ylabel("Density [g L$^{-1}$]", fontsize=8)
            else:
                ax.set_ylabel(None)

    # Adjust layout
    fig.align_ylabels()
    # plt.tight_layout()

    # Save
    plt.savefig(os.path.join(output_folder, 'selected_regressions.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_folder, 'selected_regressions.eps'), bbox_inches='tight')
    plt.close()

    print(f"[INFO] Finished printing selected regression plot to {output_folder}")