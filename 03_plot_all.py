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

# Define a color palette
color_palette = {
    'Tweets': 'blue',
    'News': 'orange',
    'Manifestos': 'green',
    'Manifestos Multi': 'red'
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
    g = sns.FacetGrid(data, col="label", col_wrap=2, height=5, aspect=2, sharey=False, col_order=['Tweets', 'News', 'Manifestos', 'Manifestos Multi'])

    # Map the lineplot to the FacetGrid
    for ax, label in zip(g.axes.flatten(), ['Tweets', 'News', 'Manifestos', 'Manifestos Multi']):
        sns.lineplot(data=data[data['label'] == label], x='iteration', y='overall_KA', marker='o', linewidth=1.5, color=color_palette[label], ax=ax, alpha=0.7)
        avg_ka = data[data['label'] == label]['overall_KA'].mean()
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
    plt.show()

def plot_combined_between(data, save_path=None):
    # Calculate the average KA score for each temperature and label
    average_ka_per_temp = data.groupby(['temperature', 'label'])['KA'].mean().reset_index()

    # Set the style for a minimalistic look
    sns.set_style("white")

    # Create the FacetGrid
    g = sns.FacetGrid(average_ka_per_temp, col="label", col_wrap=2, height=5, aspect=2, sharey=False, col_order=['Tweets', 'News', 'Manifestos', 'Manifestos Multi'])

    # Map the lineplot to the FacetGrid
    for ax, label in zip(g.axes.flatten(), ['Tweets', 'News', 'Manifestos', 'Manifestos Multi']):
        sns.lineplot(data=average_ka_per_temp[average_ka_per_temp['label'] == label], x='temperature', y='KA', marker='o', linewidth=1.5, color=color_palette[label], ax=ax, alpha=0.7)
        avg_ka = data[data['label'] == label]['KA'].mean()
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
    plt.show()

# Combine and plot "within" datasets
combined_within_data = combine_within_files(within_files)
plot_combined_within(combined_within_data, save_path="plots/combined_within.png")

# Combine and plot "between" datasets
combined_between_data = combine_between_files(between_files)
plot_combined_between(combined_between_data, save_path="plots/combined_between.png")
