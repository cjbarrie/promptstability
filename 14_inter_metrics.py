import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simps
from termcolor import colored
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_weighted_mean_pss(ka_scores):
    temperatures = list(ka_scores.keys())
    pss_values = np.array([ka_scores[temp]['Average Alpha'] for temp in temperatures])
    weights = 1 / np.array(temperatures)  # More weight to lower temperatures
    weighted_mean_pss = np.average(pss_values, weights=weights)
    return weighted_mean_pss


def find_temperature_threshold(ka_scores, threshold=0.8):
    if not all('ka_upper' in ka_scores[temp] for temp in ka_scores):
        raise ValueError("No bootstrap estimates found for temperature threshold calculation.")
    
    temperatures = list(ka_scores.keys())
    ka_upper_values = [ka_scores[temp]['ka_upper'] for temp in temperatures]

    for temp, ka_upper in zip(temperatures, ka_upper_values):
        if ka_upper < threshold:
            return temp
    return None  # No temperature found below the threshold


def calculate_auc_pss(ka_scores):
    temperatures = list(ka_scores.keys())
    pss_values = [ka_scores[temp]['Average Alpha'] for temp in temperatures]

    # Use numerical integration to calculate the area under the curve
    auc_pss = simps(pss_values, temperatures)
    return auc_pss


def calculate_pss_variance(ka_scores):
    pss_values = np.array([ka_scores[temp]['Average Alpha'] for temp in ka_scores])
    pss_variance = np.var(pss_values)
    return pss_variance


def load_annotated_data(file_path):
    annotated_data = pd.read_csv(file_path)
    return annotated_data


def calculate_ka_scores(annotated_data):
    ka_scores = {}
    grouped = annotated_data.groupby('temperature')
    for temp, group in grouped:
        mean_alpha = group['ka_mean'].mean()
        ka_lower = group['ka_lower'].mean()
        ka_upper = group['ka_upper'].mean()
        ka_scores[temp] = {'Average Alpha': mean_alpha, 'ka_lower': ka_lower, 'ka_upper': ka_upper}
    return ka_scores


def process_file(name, file_path):
    annotated_data = load_annotated_data(file_path)
    ka_scores = calculate_ka_scores(annotated_data)

    # Calculate additional metrics
    weighted_mean_pss = calculate_weighted_mean_pss(ka_scores)
    temperature_threshold = find_temperature_threshold(ka_scores, threshold=0.8)
    auc_pss = calculate_auc_pss(ka_scores)
    pss_variance = calculate_pss_variance(ka_scores)
    additional_metrics = {
        'Weighted Mean PSS': weighted_mean_pss,
        'Temperature threshold': temperature_threshold,
        'AUC-PSS': auc_pss,
        'Variance': pss_variance}

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
            if metric == 'Temperature threshold' and value is None:
                value = 'None'
            data.append({'Dataset': name, 'Metric': metric, 'Value': value})

    df = pd.DataFrame(data)
    
    # Split the dataframe into numeric and non-numeric for ranking
    df_numeric = df[df['Value'] != 'None']
    df_non_numeric = df[df['Value'] == 'None']
    
    # Convert numeric values to float for ranking
    df_numeric.loc[:, 'Value'] = df_numeric['Value'].astype(float)
    df_numeric.loc[:, 'Rank'] = df_numeric.groupby('Metric')['Value'].rank("dense", ascending=False)
    
    # Assign 'None' rank as the highest rank and replace 'None' with max temperature + 1
    max_temp = df_numeric['Value'].max()
    df_non_numeric.loc[:, 'Rank'] = 0
    df_non_numeric.loc[:, 'Value'] = max_temp + 1
    
    # Combine back the dataframes
    df_combined = pd.concat([df_numeric, df_non_numeric], ignore_index=True)
    
    # Define color palette for facets
    facet_colors = {
        'Variance': 'grey', 
        'Temperature threshold': 'grey',
        'Weighted Mean PSS': 'grey',
        'AUC-PSS': 'grey'
    }
    
    metrics_order = ['Weighted Mean PSS', 'AUC-PSS', 'Variance', 'Temperature threshold']
    
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
    fig.text(0.5, -0.05, "Note: 'None' values in Temperature threshold are replaced with max temperature + 1", ha='center', fontsize=12, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

# In your main script, pass the color_palette dictionary to the create_ranking_plot function

if __name__ == '__main__':
    files = {
        'Tweets (Rep. Dem.)': 'data/annotated/tweets_rd_between.csv',
        'Tweets (Populism)': 'data/annotated/tweets_pop_between.csv',
        'News': 'data/annotated/news_between.csv',
        'News (Short)': 'data/annotated/news_short_between.csv',
        'Manifestos': 'data/annotated/manifestos_between.csv',
        'Manifestos Multi': 'data/annotated/manifestos_multi_between.csv',
        'Stance': 'data/annotated/stance_between.csv',
        'Stance (Long)': 'data/annotated/stance_long_between.csv',
        'MII': 'data/annotated/mii_between.csv',
        'MII (Long)': 'data/annotated/mii_long_between.csv',
        'Synthetic': 'data/annotated/synth_between.csv',
        'Synthetic (Short)': 'data/annotated/synth_short_between.csv'
    }

    metric_colors = {
        'Weighted Mean PSS': 'green',
        'Temperature threshold': 'red',
        'AUC-PSS': 'blue',
        'Variance': 'yellow',
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
        print(f"Processing {name}...")
        metrics = process_file(name, file_path)
        results[name] = metrics

    # Print the results
    print_metrics(results, metric_colors)
    
    # Create the ranking plot
    create_ranking_plot(results, save_path='plots/metrics_between.png', color_palette=color_palette)

