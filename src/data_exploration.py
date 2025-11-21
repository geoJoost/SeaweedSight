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

import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t

def logarithmic_curve(x, a, b):
    """Logarithmic curve: y = a * ln(x) + b"""
    return a * np.log(x) + b

def plot_regression(df, features, output_name, output_folder='doc', scatter_color='#219ebc', log_curve=False):
    """
    Compute and plot linear or logarithmic curve regression for each feature vs. density.
    Prints regression summaries and saves plots.
    """
    print(f"{'-' * 50}")
    print(f'[INFO] {'#' * 20} Regression results {output_name} {'#' * 20}')

    # Create subplots
    n_rows = (len(features) + 1) // 2  # Arrange in 2 columns
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    axes = axes.flatten()  # Flatten for easy iteration

    # Plot each feature and compute regression
    for i, feature in enumerate(features):
        ax = axes[i]
        x = df['density']
        y = df[feature]

        if log_curve:
            # Fit logarithmic curve: y = a * ln(x) + b
            try:
                # Ensure x > 0 for log(x)
                # x = np.clip(x, a_min=1e-10, a_max=None)
                popt, pcov = curve_fit(logarithmic_curve, x, y, maxfev=10000)
                a, b = popt

                # Plot scatter points
                sns.scatterplot(x=x, y=y, ax=ax, alpha=0.5, color=scatter_color)

                # Plot fitted logarithmic curve
                x_fit = np.linspace(x.min(), x.max(), 100)
                y_fit = logarithmic_curve(x_fit, *popt)
                ax.plot(x_fit, y_fit, color='red', label=f'Log curve: y = {a:.2f} ln(x) + {b:.2f}')

                # Calculate R²
                residuals = y - logarithmic_curve(x, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y - np.mean(y))**2)
                r_squared = 1 - (ss_res / ss_tot)
                ax.set_title(f'{feature} (log-curve, R² = {r_squared:.2f})')
                print(f"[INFO] Logarithmic curve for {feature}: a={a:.2f}, b={b:.2f}, R²={r_squared:.2f}")

                # Calculate degrees of freedom
                n = len(x)
                p = len(popt)
                dof = max(0, n-p)

                # Calculate residual standard deviation
                residual_std = np.sqrt(ss_res / dof)
                cov_matrix = pcov * residual_std**2
                std_errors = np.sqrt(np.diag(cov_matrix))

                # t-values and p-values
                t_values = popt / std_errors
                p_values = 2 * (1 - t.cdf(np.abs(t_values), df=dof))

                print(f"\n\n[INFO] Logarithmic curve for {feature}:")
                print(f"[INFO] a = {a:.4f} (p = {p_values[0]:.4f}), b = {b:.4f} (p = {p_values[1]:.4f}), R² = {r_squared:.4f}")

            except RuntimeError:
                print(f"\n\n[INFO] Logarithmic curve fit failed for {feature}")
                sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha':0.5, 'color': scatter_color}, line_kws={'color':'red'})
                ax.set_title(f'{feature} (fit failed)')

        else:
            # Fit linear regression
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()

            # Print regression summary
            print(f"\n\n[INFO] Linear regression for {feature}: \n{model.summary(alpha=0.05)}")

            # Scatter plot with regression line
            sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha':0.5, 'color': scatter_color}, line_kws={'color':'red'})

            # Add R² as text
            ax.set_title(f'{feature} (R² = {model.rsquared:.2f})')

        ax.set_xlabel('Density [g/L]')
        ax.set_ylabel(feature)
        ax.legend()

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'model_regression_{output_name}.png'), dpi=200)
    plt.close()

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

# # Linear regression for the DataFrame
# def compute_regression(df, features, output_name):
#     """ Calculate linear regression using OLS """
#     print(f"{'-' * 50}")
#     print(f'[INFO] {'#' * 20} Regression results {output_name} {'#' * 20}')
#     for feature in features:

#         X = sm.add_constant(df['density'])
#         y = df[feature]
#         model = sm.OLS(y, X).fit()
#         print(f"\n\n[INFO] Regression for {feature}: \n{model.summary(alpha=0.05)}")

def plot_features_vs_density(df, features, output_name, scatter_color, log_transform=False):
    """
    Helper function to compute correlations, VIF, and regression for a given DataFrame.
    Can handle both linear and logarithmic transformations.
    """
    print(f"{'-' * 50}")
    # Make a copy of the DataFrame
    df_analysis = df.copy()

    # Apply log transform if requested
    if log_transform:
        for feature in features:# + ['density']:
            df_analysis[feature] = np.log1p(df_analysis[feature])
        output_name = f"log_{output_name}"
        plot_features = [f"log_{feature}" for feature in features] # Prepend 'log_' to feature names for plotting

    # Compute correlations
    correlations = df_analysis[['density'] + features].corr()['density'][1:]

    # Plot correlations
    plot_correlation(df_analysis, correlations, features, output_name=output_name, scatter_color=scatter_color)

    # Compute VIF and collinearity
    vif_data, corr_matrix = compute_colinearity(df_analysis, features, output_name=output_name)

    # Compute regression for selected features
    plot_regression(df_analysis, features=features, output_name=output_name, scatter_color=scatter_color, log_curve=log_transform)

    print(f"[INFO] Computed statistics for {output_name}")

def analyze_data(df, features, output_folder='.'):
    """
    Plot scatter plots with regression lines and correlation coefficients for each feature vs. density.

    Args:
        df (pd.DataFrame): DataFrame containing features and density.
        features (list): List of feature columns to plot.
        density_col (str): Name of the density column.
        output_folder (str): Directory to save the plot.
    """
    print(f"{'-' * 50}")

    # Make a copy of the DataFrame
    df_plot = df.copy()

    # Pre-process for cumulative surface area
    max_surface = df.groupby(['density', 'trial'], as_index=False)['cumulative_surface_area'].max()
    df_plot = df_plot.drop(columns=['cumulative_surface_area']).merge(max_surface, on=['density', 'trial'], how='left')

    # 1. Per-frame (linear)
    plot_features_vs_density(df_plot, features, output_name="per_frame", scatter_color='#219ebc', log_transform=False)

    # 2. Per-trial (linear)
    df_grouped = df_plot.groupby(['density', 'trial'], as_index=False)[features].mean()
    plot_features_vs_density(df_grouped, features, output_name="per_trial", scatter_color='#606c38', log_transform=False)

    # 3. Per-frame (log)
    plot_features_vs_density(df_plot, features, output_name="per_frame", scatter_color='#fb8500', log_transform=True)

    # 4. Per-trial (log)
    df_log_grouped = df_plot.groupby(['density', 'trial'], as_index=False)[features].mean()
    plot_features_vs_density(df_log_grouped, features, output_name="per_trial", scatter_color='#8b5cf6', log_transform=True)

    # # Compute correlations for each row in the DataFrame
    # correlations = df_plot[['density'] + features].corr()['density'][1:]
    # plot_correlation(df_plot, correlations, features, output_name="per_frame", scatter_color='#219ebc')

    # # Group by density and compute mean for each feature
    # df_grouped = df_plot.groupby(['density', 'trial'], as_index=False)[features].mean()
    # group_correlations = df_grouped.corr()['density'][1:]
    # plot_correlation(df_grouped, group_correlations, features, output_name="per_trial", scatter_color='#606c38')

    # # Assumptions check: VIF
    # vif_data, corr_matrix = compute_colinearity(df_plot, features, output_name='per_frame')
    # vif_data, corr_matrix = compute_colinearity(df_grouped, features, output_name='per_trial')

    # # Compute linear regression per feature
    # # Due to extreme multi-colinearity (see VIF), we do it per feature
    # compute_regression(df_plot, features=['surface_area', 'cumulative_surface_area', 'mean_G'], output_name="per frame")
    # compute_regression(df_grouped, features=['surface_area', 'cumulative_surface_area', 'mean_G'], output_name="per trial")

    print('...')


