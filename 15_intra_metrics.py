import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simps
from termcolor import colored
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_weighted_mean_intra_pss(ka_scores):
    iterations = list(ka_scores.keys())
    pss_values = np.array([ka_scores[iteration]['Average Alpha'] for iteration in iterations])
    weights = 1 / (np.array(iterations) + 1)  # More weight to lower iterations
    weighted_mean_pss = np.average(pss_values, weights=weights)
    return weighted_mean_pss


def find_iteration_threshold(ka_scores, threshold=0.8):
    if not all('ka_lower' in ka_scores[iteration] for iteration in ka_scores):
        raise ValueError("No bootstrap estimates found for iteration threshold calculation.")
    
    iterations = list(ka_scores.keys())
    ka_lower_values = [ka_scores[iteration]['ka_lower'] for iteration in iterations]

    for i, (iteration, ka_lower) in enumerate(zip(iterations, ka_lower_values)):
        if ka_lower > threshold:
            return iteration
    return None  # If no convergence, return None


def calculate_auc_intra_pss(ka_scores):
    iterations = list(ka_scores.keys())
    pss_values = [ka_scores[iteration]['Average Alpha'] for iteration in iterations]

    # Use numerical integration to calculate the area under the curve
    auc_pss = simps(pss_values, iterations)
    return auc_pss

def calculate_intra_pss_variance(ka_scores):
    pss_values = np.array([ka_scores[iteration]['Average Alpha'] for iteration in ka_scores])
    pss_variance = np.nanvar(pss_values)
    return pss_variance

def load_annotated_data(file_path):
    annotated_data = pd.read_csv(file_path)
    return annotated_data


def calculate_ka_scores(annotated_data):
    required_columns = {'ka_mean', 'ka_lower', 'ka_upper', 'iteration'}
    if not required_columns.issubset(annotated_data.columns):
        raise ValueError(f"Required columns {required_columns} not found in the data.")
    
    ka_scores = {}
    grouped = annotated_data.groupby('iteration')
    for iteration, group in grouped:
        mean_alpha = group['ka_mean'].mean()
        ka_lower = group['ka_lower'].mean()
        ka_upper = group['ka_upper'].mean()
        ka_scores[iteration] = {'Average Alpha': mean_alpha, 'ka_lower': ka_lower, 'ka_upper': ka_upper}
    return ka_scores

def process_file(name, file_path):
    annotated_data = load_annotated_data(file_path)
    ka_scores = calculate_ka_scores(annotated_data)

    # Remove any NaN values before calculations
    ka_scores = {k: v for k, v in ka_scores.items() if not np.isnan(v['Average Alpha'])}

    # Calculate additional metrics
    weighted_mean_intra_pss = calculate_weighted_mean_intra_pss(ka_scores)
    auc_intra_pss = calculate_auc_intra_pss(ka_scores)
    iteration_threshold = find_iteration_threshold(ka_scores, threshold=0.8)
    intra_pss_variance = calculate_intra_pss_variance(ka_scores)

    additional_metrics = {
        'Weighted mean': weighted_mean_intra_pss,
        'Iteration threshold': iteration_threshold,
        'Variance': intra_pss_variance,
        'AUC-PSS': auc_intra_pss,
    }

    return {
        'ka_scores': ka_scores,
        'additional_metrics': additional_metrics
    }


def print_metrics(results, metric_colors):
    for name, metrics in results.items():
        print(f"Metrics for {name}:")
        for metric, value in metrics['additional_metrics'].items():
            color = metric_colors.get(metric, 'white')
            if isinstance(value, (list, tuple)):
                value = ', '.join(map(str, value))
            print(colored(f"  {metric}: {value}", color))
        print()


def create_ranking_plot(results, save_path, color_palette):
    data = []

    for name, metrics in results.items():
        for metric, value in metrics['additional_metrics'].items():
            if metric == 'Iteration threshold' and value is None:
                value = 'None'
            data.append({'Dataset': name, 'Metric': metric, 'Value': value})

    df = pd.DataFrame(data)
    
    # Split the dataframe into numeric and non-numeric for ranking
    df_numeric = df[df['Value'] != 'None']
    df_non_numeric = df[df['Value'] == 'None']
    
    # Convert numeric values to float for ranking
    df_numeric.loc[:, 'Value'] = df_numeric['Value'].astype(float)
    df_numeric.loc[:, 'Rank'] = df_numeric.groupby('Metric')['Value'].rank("dense", ascending=False)
    
    # Assign 'None' rank as the highest rank and replace 'None' with max iteration + 1
    max_iter = df_numeric['Value'].max()
    df_non_numeric.loc[:, 'Rank'] = 0
    df_non_numeric.loc[:, 'Value'] = max_iter + 1
    
    # Combine back the dataframes
    df_combined = pd.concat([df_numeric, df_non_numeric], ignore_index=True)
    
    # Define color palette for facets
    facet_colors = {
        'Variance': 'grey', 
        'Iteration threshold': 'grey',
        'Weighted mean': 'grey',
        'AUC-PSS': 'grey'
    }
    
    metrics_order = ['Weighted mean', 'AUC-PSS', 'Variance', 'Iteration threshold']
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for ax, metric in zip(axes, metrics_order):
        metric_df = df_combined[df_combined['Metric'] == metric]
        metric_df_sorted = metric_df.sort_values('Rank', ascending=True)
        color = facet_colors[metric]
        for rank in metric_df_sorted['Rank'].unique():
            subset = metric_df_sorted[metric_df_sorted['Rank'] == rank]
            alpha_value = 0.3 + 0.7 * (rank / metric_df_sorted['Rank'].max())  # Increasing alpha by rank
            sns.barplot(data=subset, y="Dataset", x="Value", 
                        palette=[color_palette[name] for name in subset['Dataset']], ax=ax, alpha=alpha_value, linewidth=0)

        ax.set_ylabel("Dataset", fontsize=12)
        ax.set_xlabel("Value", fontsize=12)
        ax.set_title(metric, fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.grid(True, axis='x', linestyle='--', linewidth=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        # Bold axis lines
        ax.axhline(y=0, color='black', linewidth=1.5)
        ax.axvline(x=0, color='black', linewidth=1.5)
    
    # Add a note below the plot
    fig.text(0.5, -0.05, "Note: 'None' values in Iteration threshold are replaced with max iterations + 1", ha='center', fontsize=12, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    files = {
        'Tweets (Rep. Dem.)': 'data/annotated/tweets_rd_within.csv',
        'Tweets (Populism)': 'data/annotated/tweets_pop_within.csv',
        'News': 'data/annotated/news_within.csv',
        'News (Short)': 'data/annotated/news_short_within.csv',
        'Manifestos': 'data/annotated/manifestos_within.csv',
        'Manifestos Multi': 'data/annotated/manifestos_multi_within.csv',
        'Stance': 'data/annotated/stance_within.csv',
        'Stance (Long)': 'data/annotated/stance_long_within.csv',
        'MII': 'data/annotated/mii_within.csv',
        'MII (Long)': 'data/annotated/mii_long_within.csv',
        'Synthetic': 'data/annotated/synth_within.csv',
        'Synthetic (Short)': 'data/annotated/synth_short_within.csv'
    }

    metric_colors = {
        'Weighted mean': 'green',
        'Iteration threshold': 'red',
        'Variance': 'blue',
        'AUC-PSS': 'yellow',
    }

    color_palette = {
        'Tweets (Rep. Dem.)': 'darkcyan',
        'Tweets (Populism)': 'cyan',
        'News': 'orange',
        'News (Short)': 'darkorange',
        'Manifestos': 'green',
        'Manifestos Multi': 'red',
        'Stance': 'hotpink',
        'Stance (Long)': 'deeppink',
        'MII': 'mediumseagreen',
        'MII (Long)': 'seagreen',
        'Synthetic': 'indianred',
        'Synthetic (Short)': 'brown'
    }

    results = {}
    for name, file_path in files.items():
        try:
            print(f"Processing {name}...")
            metrics = process_file(name, file_path)
            results[name] = metrics
        except ValueError as e:
            print(colored(f"Error processing {name}: {e}", 'red'))

    # Print the results
    print_metrics(results, metric_colors)
    
    # Create the ranking plot
    create_ranking_plot(results, save_path='plots/metrics_within.png', color_palette=color_palette)
