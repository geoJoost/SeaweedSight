""" Script for statistics (correlation, VIF, regression) """
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import t
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.eval_measures import rmse
from typing import List, Tuple

def power_law_curve(x, a, b):
    """Power-law curve: y = a * x^b"""
    return a * np.power(x, b)

def fit_power_law_regression(
        x_values: np.ndarray,
        y_values: np.ndarray, 
        ax: plt.Axes, 
        scatter_color: str
    ) -> Tuple[float, float, float]:
    """Fit and plot a power-law regression (y = a * x^b), returning R2 and p-values."""
    try:
        x_values = np.clip(x_values, a_min=1e-10, a_max=None)
        popt, pcov = curve_fit(power_law_curve, x_values, y_values, maxfev=10000)
        a, b = popt

        # Plot scatter points
        sns.scatterplot(x=x_values, y=y_values, ax=ax, alpha=1.0, color=scatter_color, linewidth=0, s=5)

        # Plot fitted power curve
        x_fit = np.linspace(x_values.min(), x_values.max(), 100)
        y_fit = power_law_curve(x_fit, *popt)
        ax.plot(x_fit, y_fit, color='#000000', lw=1, label=f'Power curve: y = {a:.2f} ln(x) + {b:.2f}')

        # Calculate R²
        residuals = y_values - power_law_curve(x_values, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_values - np.mean(y_values))**2)
        r_squared = 1 - (ss_res / ss_tot)

        # Calculate RMSE
        rmse_val = np.sqrt(np.mean(residuals**2))

        # Calculate degrees of freedom
        n = len(x_values)
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
        sns.regplot(x=x_values, y=y_values, ax=ax, scatter_kws={'alpha':0.5, 'color':scatter_color}, line_kws={'color':'red', 'linewidth':1})
        return None, None, None

def create_regression_plot(
        df: pd.DataFrame, 
        feature_columns: List[str], 
        output_name: str, 
        output_folder: str ='doc', 
        scatter_color: str ='#219ebc',
        regression_type=''
    ):
    """
    Compute and plot linear or power curve regression for each feature vs. density.
    Prints regression summaries and saves plots.
    """
    print(f"{'-' * 50}")
    print(f'[INFO] {'#' * 20} Regression results {output_name} {'#' * 20}')

    # Create subplots
    n_rows = (len(feature_columns) + 1) // 2  # Arrange in 2 columns
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    axes = axes.flatten()  # Flatten for easy iteration

    # Plot each feature and compute regression
    for i, feature in enumerate(feature_columns):
        ax = axes[i]
        x = df[feature]
        y = df['density']
        
        if regression_type == 'pow':
            # Fit power-law curve
            r_squared, p_value, rmse_val = fit_power_law_regression(x, y, ax, scatter_color)

            print(f"\n[INFO] Power regression for {feature}:")
            print(f"R² = {r_squared:.2f}%, p-value = {p_value:.3f}, RMSE = {rmse_val:.2f} g L$^{-1}$")
            ax.set_title(f'{feature} (R² = {r_squared:.2f}, p = {p_value:.3f}, RMSE = {rmse_val:.2f} g L$^{-1}$)', fontsize=10)

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
            print(f"R² = {r_squared:.2f} | p-value = {p_value_str} | RMSE {rmse_val:.2f} g L$^{-1}$")

            # Print regression summary
            print(f"\n\n[INFO] \n{model.summary(alpha=0.05)}")

            # Scatter plot with regression line
            sns.regplot(x=x, y=y, ax=ax, scatter_kws={'alpha':0.5, 'color': scatter_color}, line_kws={'color':'red'})

            # Add R² as text
            ax.set_title(f'{feature} (R² = {model.rsquared:.2f}, p-value = {p_value:.3f}, RMSE = {rmse_val:.2f} g L$^{-1}$)', fontsize=10)

        ax.set_ylabel('Density [g L$^{-1}$]')
        ax.set_xlabel(feature)
        ax.legend()

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'model_regression_{output_name}.png'), dpi=200)
    plt.close()

def create_correlation_plot(
        df: pd.DataFrame, 
        correlations: pd.Series,
        feature_columns: List[str], 
        output_name: str, 
        output_folder: str = 'doc', 
        scatter_color: str = '#219ebc'
    ) -> None:
    """ Create and save correlation plots for given features """

    # Create subplots
    n_rows = (len(feature_columns) + 1) // 2  # Arrange in 2 columns
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    axes = axes.flatten()  # Flatten for easy iteration

    # Plot each feature
    for i, feature in enumerate(feature_columns):
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
        ax.set_xlabel('Density [g L$^{-1}$]')
        ax.set_ylabel(feature)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'correlation_{output_name}.png'), dpi=200)
    plt.close()

    # Print correlation table
    print(f"[INFO] Correlation with density ({output_name}):\n{correlations.round(2)}\n{'-' * 50}")

def create_colinearity_plot(
        df: pd.DataFrame, 
        feature_columns: List[str],
        output_name: str,
        output_folder: str = 'doc'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Compute colinearity for each feature in the DataFrame """
    # Compute VIF
    X = df[feature_columns]
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    # Compute correlation heatmap
    corr_matrix = df[feature_columns + ['density']].corr().round(2)

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
    plt.savefig(os.path.join(output_folder, f'colinearity_{output_name}.png'), dpi=200)
    plt.close()

    print(f"[INFO] Variance Inflation Factor (VIF) output ({output_name}):\n {vif_data.round(1)}")
    return vif_data, corr_matrix

def analyze_feature_relationships(
    analysis_df: pd.DataFrame,
    feature_columns: List[str],
    output_folder: str = 'doc'
) -> None:
    """
    Main function to analyze relationships between features and density.
    Performs correlation, collinearity, and regression analysis.

    Args:
        data: DataFrame containing features and density.
        features: List of feature columns to analyze.
        output_directory: Base directory for outputs.
    """
    print(f"\n{'=' * 50}")
    print(f"Analyzing relationships between predictor and biomass density")
    print(f"{'=' * 50}")

    # Preprocess data for cumulative surface area
    max_surface = analysis_df.groupby(['density', 'cycle'], as_index=False)['tot_surface_area'].max()
    processed_data = analysis_df.drop(columns=['tot_surface_area']).merge(max_surface, on=['density', 'cycle'], how='left')

    # Compute correlations
    correlations = processed_data[['density'] + feature_columns].corr()['density'][1:]

    # Analysis configurations
    analyses = [
        # (data, name, color, regression_type)
        (processed_data, "per_frame_linear", '#219ebc', ''),
        (processed_data.groupby(['density', 'cycle'], as_index=False)[feature_columns].mean(), "per_cycle_linear", '#606c38', ''),
        (processed_data, "per_frame_power", '#fb8500', 'pow'),
        (processed_data.groupby(['density', 'cycle'], as_index=False)[feature_columns].mean(), "per_cycle_power", '#8b5cf6', 'pow')
    ]

    # Run all analyses
    for data_subset, name, color, regression in analyses:
        # Correlation plot (only for linear)
        if regression == '':
            create_correlation_plot(data_subset, correlations, feature_columns, name, output_folder, color)

        # Colinearity plot (only for linear)
        if regression == '':
            create_colinearity_plot(data_subset, feature_columns, name, output_folder)

        # Regression plot
        create_regression_plot(data_subset, feature_columns, name, output_folder, color, regression)

    print(f"\n[INFO] Analysis complete. Results saved to: {output_folder}")