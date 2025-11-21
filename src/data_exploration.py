""" Script for graphs, etc"""
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import statsmodels.api as sm
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

def plot_correlation(df, correlations, features, output_name, output_folder='doc', scatter_color='#219ebc'):
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
            data=df,
            ax=ax,
            scatter_kws={'alpha':0.5, 'color':scatter_color},
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
    plt.savefig(os.path.join(output_folder, f'model_correlation_{output_name}.png'), dpi=200)
    plt.close()

    # Print correlation table
    print(f"[INFO] Correlation with density ({output_name}):\n{correlations.round(2)}\n{'-' * 50}")

def compute_colinearity(df, features, output_name):
    """
    Compute colinearity for each feature in the DataFrame.
    """
    # Compute VIF
    X = df[features]
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    print(f"[INFO] Variance Inflation Factor (VIF) output ({output_name}):\n {vif_data.round(1)}")

    # Compute correlation heatmap
    corr_matrix = df[features + ['density']].corr().round(2)

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Plot correlation heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, fmt=".2f", ax=ax1)
    ax1.set_title('Correlation heatmap')

    # Plot VIF
    ax2.barh(vif_data['feature'], vif_data['VIF'], color='skyblue')
    ax2.axvline(x=5, color='red', linestyle='--', label='VIF = 5')
    ax2.axvline(x=10, color='orange', linestyle='--', label='VIF = 10')
    ax2.set_xlabel('VIF [-]')
    ax2.set_title('Variance Inflation Factor')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'doc/model_colinearity_{output_name}', dpi=200, bbox_inches='tight')
    plt.close()
    
    return vif_data, corr_matrix

# Linear regression for the DataFrame
def compute_regression(df, features, output_name):
    """ Calculate linear regression using OLS """
    print(f"{'-' * 50}")
    print(f'[INFO] {'#' * 20} Regression results {output_name} {'#' * 20}')
    for feature in features:
        # print(f"{'-' * 50}")

        X = sm.add_constant(df['density'])
        y = df[feature]
        model = sm.OLS(y, X).fit()
        print(f"\n\n[INFO] Regression for {feature}: \n{model.summary(alpha=0.05)}")

def plot_features_vs_density(df, features, output_folder='.'):
    """
    Plot scatter plots with regression lines and correlation coefficients for each feature vs. density.

    Args:
        df (pd.DataFrame): DataFrame containing features and density.
        features (list): List of feature columns to plot.
        density_col (str): Name of the density column.
        output_folder (str): Directory to save the plot.
    """
    print(f"{'-' * 50}")

    # Make a copy of the DataFrame for plotting
    df_plot = df.copy()

    # For 'cumulative_surface_area', take the max per trial per density
    max_surface = df.groupby(['density', 'trial'], as_index=False)['cumulative_surface_area'].max()
    df_plot = df_plot.drop(columns=['cumulative_surface_area']).merge(max_surface, on=['density', 'trial'], how='left')

    # Compute correlations for each row in the DataFrame
    correlations = df_plot[['density'] + features].corr()['density'][1:]
    plot_correlation(df_plot, correlations, features, output_name="per_frame", scatter_color='#219ebc')

    # Group by density and compute mean for each feature
    df_grouped = df_plot.groupby(['density', 'trial'], as_index=False)[features].mean()
    group_correlations = df_grouped.corr()['density'][1:]
    plot_correlation(df_grouped, group_correlations, features, output_name="per_trial", scatter_color='#606c38')

    # Assumptions check: VIF
    vif_data, corr_matrix = compute_colinearity(df_plot, features, output_name='per_frame')
    vif_data, corr_matrix = compute_colinearity(df_plot, features, output_name='per_trial')

    # Compute linear regression per feature
    # Due to extreme multi-colinearity (see VIF), we do it per feature
    compute_regression(df_plot, features=['surface_area', 'cumulative_surface_area', 'mean_G'], output_name="per frame")
    compute_regression(df_grouped, features=['surface_area', 'cumulative_surface_area', 'mean_G'], output_name="per trial")

    print('...')


