import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define file paths and labels for the datasets
within_files = {
    'Tweets': 'data/annotated/tweets_within.csv',
    'News': 'data/annotated/news_within.csv',
    'Manifestos': 'data/annotated/manifestos_within.csv',
    'Manifestos Multi': 'data/annotated/manifestos_multi_within.csv'
}

between_files = {
    'Tweets': 'data/annotated/tweets_between.csv',
    'News': 'data/annotated/news_between.csv',
    'Manifestos': 'data/annotated/manifestos_between.csv',
    'Manifestos Multi': 'data/annotated/manifestos_multi_between.csv'
}

def combine_within_files(files):
    combined_data = []
    for label, file_path in files.items():
        # Load the annotated data
        annotated_data = pd.read_csv(file_path)
        # Ensure iteration is of integer type for correct plotting
        annotated_data['iteration'] = annotated_data['iteration'].astype(int)
        # Drop rows where overall_KA is NaN
        annotated_data = annotated_data.dropna(subset=['overall_KA'])
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
        annotated_data = annotated_data.dropna(subset=['KA'])
        # Add a column for the label
        annotated_data['label'] = label
        combined_data.append(annotated_data)
    return pd.concat(combined_data, ignore_index=True)

def plot_combined_within(data, save_path=None):
    # Set the style for a minimalistic look
    sns.set_style("white")

    # Create the FacetGrid
    g = sns.FacetGrid(data, col="label", col_wrap=2, height=4, aspect=2.5, sharey=True, col_order=['Tweets', 'News', 'Manifestos', 'Manifestos Multi'])

    # Map the lineplot to the FacetGrid
    g.map(sns.lineplot, 'iteration', 'overall_KA', marker='o', linewidth=1, color='black')

    # Add average KA as horizontal lines
    for ax, label in zip(g.axes.flatten(), ['Tweets', 'News', 'Manifestos', 'Manifestos Multi']):
        avg_ka = data[data['label'] == label]['overall_KA'].mean()
        ax.axhline(y=avg_ka, color='red', linestyle='--', linewidth=1, label=f'Average KA: {avg_ka:.2f}')
        ax.axhline(y=0.80, color='black', linestyle=':', linewidth=1, label='Threshold KA: 0.80')
        ax.legend(fontsize=10, frameon=False)

    # Customize the plot for a minimalistic look
    g.set_axis_labels('Iteration', "Krippendorff's Alpha (KA)", fontsize=12, fontweight='bold')
    g.set_titles("{col_name} Within", size=14, fontweight='bold')
    g.set(xticks=data['iteration'].unique())
    g.set(yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_combined_between(data, save_path=None):
    # Calculate the average KA score for each temperature and label
    average_ka_per_temp = data.groupby(['temperature', 'label'])['KA'].mean().reset_index()

    # Set the style for a minimalistic look
    sns.set_style("white")

    # Create the FacetGrid
    g = sns.FacetGrid(average_ka_per_temp, col="label", col_wrap=2, height=4, aspect=2.5, sharey=True, col_order=['Tweets', 'News', 'Manifestos', 'Manifestos Multi'])

    # Map the lineplot to the FacetGrid
    g.map(sns.lineplot, 'temperature', 'KA', marker='o', linewidth=1, color='black')

    # Add average KA as horizontal lines
    for ax, label in zip(g.axes.flatten(), ['Tweets', 'News', 'Manifestos', 'Manifestos Multi']):
        avg_ka = data[data['label'] == label]['KA'].mean()
        ax.axhline(y=avg_ka, color='red', linestyle='--', linewidth=1, label=f'Average KA: {avg_ka:.2f}')
        ax.axhline(y=0.80, color='black', linestyle=':', linewidth=1, label='Threshold KA: 0.80')
        ax.legend(fontsize=10, frameon=False)

    # Customize the plot for a minimalistic look
    g.set_axis_labels('Temperature', "Krippendorff's Alpha (KA)", fontsize=12, fontweight='bold')
    g.set_titles("{col_name} Between", size=14, fontweight='bold')
    g.set(xticks=average_ka_per_temp['temperature'].unique())
    g.set(yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# Combine and plot "within" datasets
combined_within_data = combine_within_files(within_files)
plot_combined_within(combined_within_data, save_path="plots/combined_within.png")

# Combine and plot "between" datasets
combined_between_data = combine_between_files(between_files)
plot_combined_between(combined_between_data, save_path="plots/combined_between.png")
