import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define file paths and labels for the datasets
within_files = {
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

between_files = {
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

# Define a color palette
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

def combine_within_files(files):
    combined_data = []
    for label, file_path in files.items():
        # Load the annotated data
        annotated_data = pd.read_csv(file_path)
        # Ensure iteration is of integer type for correct plotting
        annotated_data['iteration'] = annotated_data['iteration'].astype(int)
        # Drop rows where overall_KA is NaN
        annotated_data = annotated_data.dropna(subset=['ka_mean'])
        # Add a column for the label
        annotated_data['label'] = label
        combined_data.append(annotated_data)
    return pd.concat(combined_data, ignore_index=True)

def combine_between_files(files):
    combined_data = []
    for label, file_path in files.items():
        # Load the annotated data
        annotated_data = pd.read_csv(file_path)
        # Ensure temperature is of float type for correct plotting
        annotated_data['temperature'] = annotated_data['temperature'].astype(float)
        # Drop rows where KA is NaN
        annotated_data = annotated_data.dropna(subset=['ka_mean'])
        # Add a column for the label
        annotated_data['label'] = label
        combined_data.append(annotated_data)
    return pd.concat(combined_data, ignore_index=True)

def plot_combined_within(data, order, save_path=None):
    # Set the style for a minimalistic look
    sns.set_style("white")

    # Create the FacetGrid
    g = sns.FacetGrid(data, col="label", col_wrap=3, height=5, aspect=1, sharey=False, col_order=order)

    # Map the lineplot to the FacetGrid
    g.map_dataframe(sns.lineplot, x='iteration', y='ka_mean', marker='o', linewidth=1.5)

    # Iterate over each axis to add the error bars and other customizations
    for ax, label in zip(g.axes.flatten(), order):
        subset = data[data['label'] == label]
        ci_lowers = subset['ka_mean'] - subset['ka_lower']
        ci_uppers = subset['ka_upper'] - subset['ka_mean']
        color = color_palette[label]
        ax.errorbar(subset['iteration'], subset['ka_mean'], yerr=[ci_lowers, ci_uppers], fmt='o', linestyle='-', color=color, ecolor=color, capsize=3)
        avg_ka = subset['ka_mean'].mean()
        ax.axhline(y=avg_ka, color='red', linestyle='--', linewidth=1.5, label=f'Average KA: {avg_ka:.2f}')
        ax.axhline(y=0.80, color='black', linestyle=':', linewidth=1.5, label='Threshold KA: 0.80')
        ax.legend(fontsize=10, frameon=False)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

    # Customize the plot for a professional look
    g.set_axis_labels('Iteration', "Krippendorff's Alpha (KA)", fontsize=16, fontweight='bold')
    g.set_titles("{col_name} Within", size=18, fontweight='bold')
    g.set(xticks=data['iteration'].unique())
    g.set(yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")
    plt.show()

def plot_combined_between(data, order, save_path=None):
    # Calculate the average KA score for each temperature and label
    average_ka_per_temp = data.groupby(['temperature', 'label']).agg({
        'ka_mean': 'mean',
        'ka_lower': 'mean',
        'ka_upper': 'mean'
    }).reset_index()

    # Set the style for a minimalistic look
    sns.set_style("white")

    # Create the FacetGrid
    g = sns.FacetGrid(average_ka_per_temp, col="label", col_wrap=3, height=5, aspect=1, sharey=False, col_order=order)

    # Map the lineplot to the FacetGrid
    g.map_dataframe(sns.lineplot, x='temperature', y='ka_mean', marker='o', linewidth=1.5)

    # Iterate over each axis to add the error bars and other customizations
    for ax, label in zip(g.axes.flatten(), order):
        subset = average_ka_per_temp[average_ka_per_temp['label'] == label]
        ci_lowers = subset['ka_mean'] - subset['ka_lower']
        ci_uppers = subset['ka_upper'] - subset['ka_mean']
        color = color_palette[label]
        ax.errorbar(subset['temperature'], subset['ka_mean'], yerr=[ci_lowers, ci_uppers], fmt='o', linestyle='-', color=color, ecolor=color, capsize=3)
        avg_ka = subset['ka_mean'].mean()
        ax.axhline(y=avg_ka, color='red', linestyle='--', linewidth=1.5, label=f'Average KA: {avg_ka:.2f}')
        ax.axhline(y=0.80, color='black', linestyle=':', linewidth=1.5, label='Threshold KA: 0.80')
        ax.legend(fontsize=10, frameon=False)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

    # Customize the plot for a professional look
    g.set_axis_labels('Temperature', "Krippendorff's Alpha (KA)", fontsize=16, fontweight='bold')
    g.set_titles("{col_name} Between", size=18, fontweight='bold')
    g.set(xticks=average_ka_per_temp['temperature'].unique())
    g.set(yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")
    plt.show()

# Combine "within" datasets and order them by average KA
combined_within_data = combine_within_files(within_files)
within_order = combined_within_data.groupby('label')['ka_mean'].mean().sort_values(ascending=False).index.tolist()
plot_combined_within(combined_within_data, within_order, save_path="plots/combined_within.png")

# Combine "between" datasets and order them by average KA
combined_between_data = combine_between_files(between_files)
between_order = combined_between_data.groupby('label')['ka_mean'].mean().sort_values(ascending=False).index.tolist()
plot_combined_between(combined_between_data, between_order, save_path="plots/combined_between.png")
