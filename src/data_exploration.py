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
from statsmodels.tools.eval_measures import rmse

def logarithmic_curve(x, a, b):
    """Logarithmic curve: y = a * ln(x) + b"""
    return a * np.log(x) + b # Linear-log model

def power_law_curve(x, a, b):
    """Power-law curve: y = a * x^b"""
    return a * np.power(x, b)

def fit_logarithmic_regression(x, y, ax, scatter_color):
    """Fit and plot a logarithmic regression, returning R² and p-values."""
    try:
        # Ensure x > 0 for log(x)
        x = np.clip(x, a_min=1e-10, a_max=None)
        popt, pcov = curve_fit(logarithmic_curve, x, y, maxfev=10000)
        a, b = popt

        # Plot scatter points
        sns.scatterplot(x=x, y=y, ax=ax, alpha=0.5, color=scatter_color)

        # Plot fitted logarithmic curve
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = logarithmic_curve(x_fit, *popt)
        ax.plot(x_fit, y_fit, color='red', label=f'Logarithmic curve: y = {a:.2f} ln(x) + {b:.2f}')

        # Calculate R²
        residuals = y - logarithmic_curve(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Calculate RMSE
        rmse_val = np.sqrt(np.mean(residuals**2))

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

        return r_squared, p_values[1], rmse_val
    except RuntimeError:
        print(f"\n\n[ERROR] Logarithmic curve fit failed")
        sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha':0.5, 'color':scatter_color}, line_kws={'color':'red'})
        return None, None, None

def fit_power_law_regression(x, y, ax, scatter_color):
    """Fit and plot a power-law regression, returning R² and p-values."""
    try:
        # Ensure x > 0 for log(x)
        x = np.clip(x, a_min=1e-10, a_max=None)
        popt, pcov = curve_fit(power_law_curve, x, y, maxfev=10000)
        a, b = popt

        # Plot scatter points
        sns.scatterplot(x=x, y=y, ax=ax, alpha=0.5, color=scatter_color)

        # Plot fitted logarithmic curve
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = power_law_curve(x_fit, *popt)
        ax.plot(x_fit, y_fit, color='red', label=f'Power curve: y = {a:.2f} ln(x) + {b:.2f}')

        # Calculate R²
        residuals = y - power_law_curve(x, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Calculate RMSE
        rmse_val = np.sqrt(np.mean(residuals**2))

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

        return r_squared, p_values[1], rmse_val
    except RuntimeError:
        print(f"\n\n[ERROR] Power curve fit failed")
        sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha':0.5, 'color':scatter_color}, line_kws={'color':'red'})
        return None, None, None

def plot_regression(df, features, output_name, output_folder='doc', scatter_color='#219ebc', curve=''):
    """
    Compute and plot linear or power curve regression for each feature vs. density.
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
        x = df[feature] #df['density']
        y = df['density'] #df[feature]

        if curve == 'log':
            # Fit logarithmic curve
            r_squared, p_value, rmse_val = fit_logarithmic_regression(x, y, ax, scatter_color)
            print(f"\n[INFO] Logarithmic regression for {feature}:")
            print(f"R² = {r_squared:.2f}%, p-value = {p_value:.3f}")
            ax.set_title(f'{feature} (R² = {r_squared:.2f}, p = {p_value:.3f}, RMSE = {rmse_val:.2f} g/L)', fontsize=10)
        
        elif curve == 'pow':
            # Fit power-law curve
            r_squared, p_value, rmse_val = fit_power_law_regression(x, y, ax, scatter_color)

            print(f"\n[INFO] Power regression for {feature}:")
            print(f"R² = {r_squared:.2f}%, p-value = {p_value:.3f}, RMSE = {rmse_val:.2f} g/L")
            ax.set_title(f'{feature} (R² = {r_squared:.2f}, p = {p_value:.3f}, RMSE = {rmse_val:.2f} g/L)', fontsize=10)

        else:
            # Fit linear regression
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()

            # Extract R² and p-value
            r_squared = model.rsquared
            p_value = model.f_pvalue

            # Calculate RMSE in g/L
            y_pred = model.predict(X)
            rmse_val = rmse(y, y_pred)

            # Format p-value for reporting
            if p_value < 0.001:
                p_value_str = "<.001"
            else:
                p_value_str = f"{p_value:.3f}"
            
            # Print results
            print(f"\n[INFO] Linear regression for {feature}:")
            print(f"R² = {r_squared:.2f} | p-value = {p_value_str} | RMSE {rmse_val:.2f} g/L")

            # Print regression summary
            # print(f"\n\n[INFO] \n{model.summary(alpha=0.05)}")

            # Scatter plot with regression line
            sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha':0.5, 'color': scatter_color}, line_kws={'color':'red'})

            # Add R² as text
            ax.set_title(f'{feature} (R² = {model.rsquared:.2f}, p-value = {p_value:.3f}, RMSE = {rmse_val:.2f})', fontsize=10)

        ax.set_ylabel('Density [g/L]')
        ax.set_xlabel(feature)
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

def plot_features_vs_density(df, features, output_name, scatter_color, curve=''):
    """
    Helper function to compute correlations, VIF, and regression for a given DataFrame.
    Can handle both linear and logarithmic transformations.
    """
    print(f"{'-' * 50}")
    # Make a copy of the DataFrame
    df_analysis = df.copy()

    # Apply new naming convention
    if curve != '':
        output_name = f"{curve}_{output_name}"

    # Compute correlations
    correlations = df_analysis[['density'] + features].corr()['density'][1:]

    # Plot correlations
    if curve == '': # Only compute once with linear
        plot_correlation(df_analysis, correlations, features, output_name=output_name, scatter_color=scatter_color)

    # Compute VIF and collinearity
    vif_data, corr_matrix = compute_colinearity(df_analysis, features, output_name=output_name)

    # Compute regression for selected features
    # Even for log-models, we keep the original values
    # Instead, we compute a logarithmic curve on the original values
    plot_regression(df, features=features, output_name=output_name, scatter_color=scatter_color, curve=curve)

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
    plot_features_vs_density(df_plot, features, output_name="per_frame", scatter_color='#219ebc', curve='')

    # 2. Per-trial (linear)
    df_grouped = df_plot.groupby(['density', 'trial'], as_index=False)[features].mean()
    plot_features_vs_density(df_grouped, features, output_name="per_cycle", scatter_color='#606c38', curve='')

    # 3. Per-frame (log)
    plot_features_vs_density(df_plot, features, output_name="per_frame", scatter_color='#fb8500', curve='log')

    # 4. Per-trial (log)
    plot_features_vs_density(df_grouped, features, output_name="per_cycle", scatter_color='#8b5cf6', curve='log')

    # 5. Per-frame (power-law)
    plot_features_vs_density(df_plot, features, output_name="per_frame", scatter_color='#fb8500', curve='pow')

    # 6. Per-trial (power-law)
    plot_features_vs_density(df_grouped, features, output_name="per_cycle", scatter_color='#8b5cf6', curve='pow')

def plot_all_regressions(df, features, feature_names, output_folder='doc'):
    """
    Plot all regressions (per-frame/per-cycle, linear/log-linear) in a single figure.
    Uses the existing plot_regression logic but combines everything into one grid.
    """
    # Prepare data for per-frame and per-cycle analyses
    df_per_frame = df.copy()
    df_per_cycle = df.groupby(['density', 'trial'], as_index=False)[features].mean()

    # Create figure with shared y-axes (now for density)
    fig, axes = plt.subplots(len(features), 4, figsize=(7.5, 1.8*len(features)), sharey=True, sharex='row')
    if len(features) == 1:
        axes = axes.reshape(1, -1)  # Ensure 2D array even with 1 feature

    # Define colors and analysis configurations
    colors = {
        'Per-frame | Linear': '#219ebc',
        'Per-frame | Power': '#fb8500',
        'Per-cycle | Linear': '#606c38',
        'Per-cycle | Power': '#8b5cf6'
    }

    # Column titles
    col_titles = [
        'Per-frame | Linear',
        'Per-frame | Power',
        'Per-cycle | Linear',
        'Per-cycle | Power'
    ]

    # Set column titles
    for ax, col_title in zip(axes[0], col_titles):
        ax.set_title(col_title, fontsize=10, pad=5, ha='center')

    # Plot each feature
    for i, (feature, feature_name) in enumerate(zip(features, feature_names)):
        for j, (analysis_type, df_analysis, reg_type) in enumerate([
            ('Per-frame', df_per_frame, 'Linear'),
            ('Per-frame', df_per_frame, 'Power'),
            ('Per-cycle', df_per_cycle, 'Linear'),
            ('Per-cycle', df_per_cycle, 'Power')
        ]):
            ax = axes[i, j]
            # Switch x and y: y is density, x is feature
            y = df_analysis['density']
            x = df_analysis[feature]

            # Set Y-axis ticks for density (shared across all subplots)
            ax.set_yticks([1, 2, 3, 4, 5])
            ax.set_ylim(0, 5.5)  # Ensure consistent Y-axis range

            # Completely disable per-frame plots for cumulative surface area
            if "tot. surface area" in feature_name.lower() and j in (0, 1):
                #ax._shared_x_refs = []
                #ax._shared_y_refs = []
                ax.cla()
                # ax.set_xticks([])  # Remove x-tick marks
                # ax.set_yticks([])  # Remove y-tick marks
                ax.set_xlabel('')  # Remove x-axis label
                ax.set_ylabel('')  # Remove y-axis label
                ax.set_frame_on(True)
                ax.text(0.5, 0.5, "Per-cycle only.", ha="center", va="center", fontsize=10, transform=ax.transAxes)
                if j == 0:
                    ax.set_ylabel("Density [g/L]", fontsize=10, labelpad=5)
                continue

            # For surface area features, use million formatter on X-axis
            if "tot. surface area" in feature_name.lower():
                ax.xaxis.set_major_formatter(mtick.EngFormatter(unit=''))

            # Plot data
            scatter_color = colors[col_titles[j]]
            power_curve = 'Pow' in reg_type

            if power_curve:
                fit_power_law_regression(x, y, ax, scatter_color)
            else:
                X = sm.add_constant(x)
                model = sm.OLS(y, X).fit()
                sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha':0.5, 'color':scatter_color}, line_kws={'color':'red'})

            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)

            # Only first column gets a ylabel (density)
            if j == 0:
                ax.set_ylabel("Density [g/L]", fontsize=10, labelpad=5)
            else:
                ax.set_ylabel('')

            # Only bottom row gets the x-label (feature name)
            ax.set_xlabel(feature_name, fontsize=10)

            # # Show X-axis tick marks for the second row, third and fourth columns
            # if i == 1 and j in (2, 3):
            #     ax.set_xticks(ax.get_xticks())
            #     ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
            #     ax.set_xlabel("Tot. surface area [px]")
            #if i == len(features) - 1:
            #    ax.set_xlabel(feature_name, fontsize=10, labelpad=5)
            #else:
            #    ax.set_xlabel('')

    # Adjust layout
    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'all_regressions.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(output_folder, 'all_regressions.pdf'), dpi=600, bbox_inches='tight')
    plt.close()

    print(f"[INFO] Finished printing shared regression plot to {output_folder}")