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
import matplotlib.ticker as mtick

import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t
from matplotlib.lines import Line2D  

def logarithmic_curve(x, a, b):
    """Logarithmic curve: y = a * ln(x) + b"""
    return a * np.log(x) + b # Linear-log model

def fit_logarithmic_regression(x, y, ax, scatter_color):
    """Fit and plot a logarithmic regression, returning R² and p-values."""
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

        return r_squared, p_values[1]
    except RuntimeError:
        print(f"\n\n[ERROR] Logarithmic curve fit failed")
        sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha':0.5, 'color':scatter_color}, line_kws={'color':'red'})
        return None, None

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
            r_squared, p_value = fit_logarithmic_regression(x, y, ax, scatter_color)
            print(f"\n[INFO] Logarithmic regression for {feature}:")
            print(f"R² = {r_squared * 100:.1f}%, p-value = {p_value:.3f}")
            ax.set_title(f'{feature} (R² = {r_squared * 100:.1f}, p = {p_value:.3f})')

        else:
            # Fit linear regression
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()

            # Extract R² and p-value
            r_squared = model.rsquared
            p_value = model.f_pvalue

            # Format p-value for reporting
            if p_value < 0.001:
                p_value_str = "<.001"
            else:
                p_value_str = f"{p_value:.3f}"
            
            # Print results in your desired format
            print(f"\n[INFO] Linear regression for {feature}:")
            print(f"R² = {r_squared * 100:.1f}%, p-value = {p_value_str}")

            # Print regression summary
            # print(f"\n\n[INFO] \n{model.summary(alpha=0.05)}")

            # Scatter plot with regression line
            sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha':0.5, 'color': scatter_color}, line_kws={'color':'red'})

            # Add R² as text
            ax.set_title(f'{feature} (R² = {model.rsquared * 100:.1f}, p-value = {p_value:.3f})')

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
    # Even for log-models, we keep the original values
    # Instead, we compute a logarithmic curve on the original values
    plot_regression(df, features=features, output_name=output_name, scatter_color=scatter_color, log_curve=log_transform)

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

def plot_combined_regressions(df, features, output_folder='doc'):
    """
    Plot all regressions (per-frame/per-cycle, linear/log-linear) in a single figure.
    """
    # Prepare data for per-frame and per-cycle analyses
    df_per_frame = df.copy()
    df_per_cycle = df.groupby(['density', 'trial'], as_index=False)[features].mean()

    # Create a 4x4 grid of subplots (8 features × 2 analysis types × 2 regression types)
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()

    # Define colors for scatter points
    colors = {
        'per_frame_linear': '#219ebc',
        'per_frame_log': '#fb8500',
        'per_cycle_linear': '#606c38',
        'per_cycle_log': '#8b5cf6'
    }

    for idx, feature in enumerate(features):
        for analysis_type, df_analysis in [('per_frame', df_per_frame), ('per_cycle', df_per_cycle)]:
            for regression_type in ['linear', 'log']:
                ax = axes[idx * 4 + (0 if analysis_type == 'per_frame' else 2) + (0 if regression_type == 'linear' else 1)]
                x = df_analysis['density']
                y = df_analysis[feature]

                # Fit regression
                if regression_type == 'log':
                    try:
                        popt, pcov = curve_fit(logarithmic_curve, x, y, maxfev=10000)
                        a, b = popt
                        x_fit = np.linspace(x.min(), x.max(), 100)
                        y_fit = logarithmic_curve(x_fit, *popt)
                        sns.scatterplot(x=x, y=y, ax=ax, alpha=0.5, color=colors[f'{analysis_type}_{regression_type}'])
                        ax.plot(x_fit, y_fit, color='red', label=f'Log curve: y = {a:.2f} ln(x) + {b:.2f}')
                        residuals = y - logarithmic_curve(x, *popt)
                        ss_res = np.sum(residuals**2)
                        ss_tot = np.sum((y - np.mean(y))**2)
                        r_squared = 1 - (ss_res / ss_tot)
                        n = len(x)
                        p = len(popt)
                        dof = max(0, n - p)
                        residual_std = np.sqrt(ss_res / dof)
                        cov_matrix = pcov * residual_std**2
                        std_errors = np.sqrt(np.diag(cov_matrix))
                        t_values = popt / std_errors
                        p_values = 2 * (1 - t.cdf(np.abs(t_values), df=dof))
                        ax.set_title(f'{feature} | {analysis_type} | Log (R² = {r_squared*100:.1f}, p = {p_values[1]:.3f})')
                    except RuntimeError:
                        sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha':0.5, 'color':colors[f'{analysis_type}_{regression_type}']}, line_kws={'color':'red'})
                        ax.set_title(f'{feature} | {analysis_type} | Log (fit failed)')
                else:
                    X = sm.add_constant(x)
                    model = sm.OLS(y, X).fit()
                    r_squared = model.rsquared
                    p_value = model.f_pvalue
                    p_value_str = f"{p_value:.3f}" if p_value >= 0.001 else "<.001"
                    sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha':0.5, 'color':colors[f'{analysis_type}_{regression_type}']}, line_kws={'color':'red'})
                    ax.set_title(f'{feature} | {analysis_type} | Linear (R² = {r_squared*100:.1f}, p = {p_value_str})')

                ax.set_xlabel('Density [g/L]')
                ax.set_ylabel(feature)
                ax.legend()

    # Hide unused subplots
    for j in range(len(features) * 4, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'combined_regressions.png'), dpi=200)
    plt.close()

def plot_all_regressions(df, features, feature_names, output_folder='doc'):
    """
    Plot all regressions (per-frame/per-cycle, linear/log-linear) in a single figure.
    Uses the existing plot_regression logic but combines everything into one grid.
    """
    # Prepare data for per-frame and per-cycle analyses
    df_per_frame = df.copy()
    df_per_cycle = df.groupby(['density', 'trial'], as_index=False)[features].mean()

    # Create figure with shared y-axes
    fig, axes = plt.subplots(len(features), 4, figsize=(7.5, 1.8*len(features)),
                            sharey='row', sharex='col')

    if len(features) == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array even with 1 feature

    # Define colors and analysis configurations
    colors = {
        'Per-frame | Linear': '#219ebc',
        'Per-frame | Log-linear': '#fb8500',
        'Per-cycle | Linear': '#606c38',
        'Per-cycle | Log-linear': '#8b5cf6'
    }

    # Column titles
    col_titles = [
        'Per-frame | Linear',
        'Per-frame | Log-linear',
        'Per-cycle | Linear',
        'Per-cycle | Log-linear'
    ]

    # Set column titles
    for ax, col_title in zip(axes[0], col_titles):
        ax.set_title(col_title, fontsize=10, pad=5, ha='center')

    # Plot each feature
    for i, (feature, feature_name) in enumerate(zip(features, feature_names)):
        for j, (analysis_type, df_analysis, reg_type) in enumerate([
            ('Per-frame', df_per_frame, 'Linear'),
            ('Per-frame', df_per_frame, 'Log-linear'),
            ('Per-cycle', df_per_cycle, 'Linear'),
            ('Per-cycle', df_per_cycle, 'Log-linear')
        ]):
            ax = axes[i, j]
            x = df_analysis['density']
            y = df_analysis[feature]

            # For total surface area, use million formatter
            ax.yaxis.set_major_formatter(mtick.EngFormatter(unit=''))

            # Configure x-ticks: only bottom row gets ticks and labels
            if i == len(features) - 1:
                ax.set_xticks([1, 2, 3, 4, 5])
            else:
                ax.set_xticks([])

            # Completely disable per-frame plots for cumulative surface area
            if feature_name == "Cumulative surface area" and j in (0, 1):

                # Break sharey/sharex links so the axes stops being re-populated automatically
                ax._shared_x_refs = []
                ax._shared_y_refs = []

                # Fully clear the axes including shared elements
                ax.cla()

                # Turn off ticks and frame
                #ax.set_xticks([])
                #ax.set_yticks([])
                #ax.set_xlabel('')
                #ax.set_ylabel('')
                ax.set_frame_on(True)

                ax.text(0.5, 0.5, "Per-cycle only.", ha="center", va="center", fontsize=10, transform=ax.transAxes)

                
                if j == 0:
                    ax.set_ylabel("Tot. surface area [px]", fontsize=10, labelpad=5)
                continue

            # Plot data
            scatter_color = colors[col_titles[j]]
            log_curve = 'Log' in reg_type

            if log_curve:
                fit_logarithmic_regression(x, y, ax, scatter_color)
            else:
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()
                sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha':0.5, 'color':scatter_color}, line_kws={'color':'red'})

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)

            # Only first column gets a ylabel
            if j == 0:
                ax.set_ylabel(feature_name, fontsize=10, labelpad=5)
            else:
                ax.set_ylabel('')

            # Only bottom row gets the x-label
            if i < len(features) - 1:
                ax.set_xlabel('')
            else:
                ax.set_xlabel('Density [g/L]', fontsize=10)


    # Adjust layout
    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'all_regressions.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_folder, 'all_regressions.pdf'), dpi=300, bbox_inches='tight')
    plt.close()