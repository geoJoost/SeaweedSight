""" Script for graphs, etc"""
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os


def plot_features_vs_density(df, features, output_folder='.'):
    """
    Plot scatter plots with regression lines and correlation coefficients for each feature vs. density.

    Args:
        df (pd.DataFrame): DataFrame containing features and density.
        features (list): List of feature columns to plot.
        density_col (str): Name of the density column.
        output_folder (str): Directory to save the plot.
    """

    # Make a copy of the DataFrame for plotting
    df_plot = df.copy()

    # For 'cumulative_surface_area', take the max per trial per density
    if 'cumulative_surface_area' in features:
        max_surface = df.groupby(['density', 'trial'], as_index=False)['cumulative_surface_area'].max()
        df_plot = df_plot.drop(columns=['cumulative_surface_area']).merge(max_surface, on=['density', 'trial'], how='left')

    # Compute correlations
    correlations = df_plot[['density'] + features].corr()['density'][1:]

    # Create subplots
    n_rows = (len(features) + 1) // 2  # Arrange in 2 columns
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    axes = axes.flatten()  # Flatten for easy iteration

    # Plot each feature
    for i, feature in enumerate(features):
        ax = axes[i]
        # Scatter plot with regression line
        sns.regplot(
            x='density',
            y=feature,
            data=df_plot,
            ax=ax,
            scatter_kws={'alpha':0.5},
            line_kws={'color':'red'}
        )
        # Add correlation as text
        corr = correlations[feature]
        ax.set_title(f'{feature} (r = {corr:.2f})')
        ax.set_xlabel('Density [g/L]')
        ax.set_ylabel(feature)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'correlation.png'), dpi=200)
    plt.close()


    # Print correlation table
    print(f"[INFO] Correlation with density:\n{correlations}")