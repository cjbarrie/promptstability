import pandas as pd
import os
import re

# --- Step 1: Define the file paths (converted from your R tibble) ---
file_paths = [
    {"file": "data/annotated/reannotated/comparison/cleaned_manifestos_filtered.csv",     "dataset": "manifestos",      "type": "Filtered"},
    {"file": "data/annotated/reannotated/comparison/cleaned_manifestos_multi_filtered.csv", "dataset": "manifestos_multi", "type": "Filtered"},
    {"file": "data/annotated/reannotated/comparison/cleaned_mii_filtered.csv",            "dataset": "mii",             "type": "Filtered"},
    {"file": "data/annotated/reannotated/comparison/cleaned_mii_long_filtered.csv",       "dataset": "mii_long",        "type": "Filtered"},
    {"file": "data/annotated/reannotated/comparison/cleaned_news_filtered.csv",           "dataset": "news",            "type": "Filtered"},
    {"file": "data/annotated/reannotated/comparison/cleaned_news_short_filtered.csv",     "dataset": "news_short",      "type": "Filtered"},
    {"file": "data/annotated/reannotated/comparison/cleaned_stance_filtered.csv",         "dataset": "stance",          "type": "Filtered"},
    {"file": "data/annotated/reannotated/comparison/cleaned_stance_long_filtered.csv",    "dataset": "stance_long",     "type": "Filtered"},
    {"file": "data/annotated/reannotated/comparison/cleaned_synth_filtered.csv",          "dataset": "synth",           "type": "Filtered"},
    {"file": "data/annotated/reannotated/comparison/cleaned_synth_short_filtered.csv",    "dataset": "synth_short",     "type": "Filtered"},
    {"file": "data/annotated/reannotated/comparison/cleaned_tweets_pop_filtered.csv",     "dataset": "tweets_pop",      "type": "Filtered"},
    {"file": "data/annotated/reannotated/comparison/cleaned_tweets_rd_filtered.csv",      "dataset": "tweets_rd",       "type": "Filtered"}
]

# --- Step 2: Process each prompt file ---
all_prompt_variants = []  # To hold all prompt variants

for info in file_paths:
    file = info["file"]
    dataset_name = info["dataset"]
    
    try:
        # Read the CSV file (each file must have at least 'prompt' and 'temperature')
        df = pd.read_csv(file)
    except Exception as e:
        print(f"Error reading file {file}: {e}")
        continue
    
    # Check that required columns exist
    if 'temperature' not in df.columns or 'prompt' not in df.columns:
        print(f"File {file} is missing required columns.")
        continue

    # Round the temperature column to 1 decimal place
    df['temperature'] = df['temperature'].round(1)
    
    # Extract unique prompt–temperature combinations
    unique_combinations = df[['prompt', 'temperature']].drop_duplicates()
    
    # Escape LaTeX special characters in the prompt (here, escaping underscores)
    unique_combinations['prompt'] = unique_combinations['prompt'].str.replace('_', r'\_', regex=True)
    
    # Add a dataset column
    unique_combinations['dataset'] = dataset_name
    all_prompt_variants.append(unique_combinations)

# Combine all prompt variants into one DataFrame.
all_results = pd.concat(all_prompt_variants, ignore_index=True)

# For merging purposes, ensure that temperature is numeric.
all_results['temperature'] = all_results['temperature'].astype(float)

# --- Step 3: Read the separate ka_results CSV ---
ka_results_file = "/Users/christopherbarrie/Dropbox/nyu_projects/promptstability/data/annotated/reannotated/comparison/ka_results_combined.csv"
try:
    ka_results = pd.read_csv(ka_results_file)
except Exception as e:
    print(f"Error reading ka_results file {ka_results_file}: {e}")
    ka_results = pd.DataFrame()

# Ensure temperature is numeric in ka_results.
if not ka_results.empty and 'temperature' in ka_results.columns:
    ka_results['temperature'] = ka_results['temperature'].astype(float)
else:
    print("ka_results is empty or missing the 'temperature' column.")

# --- Step 4: Merge the prompt data with the ka_results ---
# We merge on both 'dataset' and 'temperature'.
merged_results = pd.merge(all_results, ka_results[['temperature', 'ka_mean', 'dataset']], 
                          on=['dataset', 'temperature'], how='left')

# --- Step 5: Filter for poorly performing prompts (ka_mean < 0.8) ---
poor_results = merged_results[merged_results['ka_mean'] < 0.8].copy()

# --- Step 6: Produce LaTeX output for all prompt variants ---
# (For LaTeX, we convert temperature to a string with 1 decimal place.)
all_results['temperature'] = all_results['temperature'].map('{:.1f}'.format)
latex_all_lines = [
    r"\lstset{breaklines=true, breakatwhitespace=true}",
    r"\begin{lstlisting}[label=lst:promptvariants]",
    r"% Format: Dataset \t Temperature \t Prompt"
]
for idx, row in all_results.iterrows():
    latex_all_lines.append(f"{row['dataset']}\t{row['temperature']}\t{row['prompt']}")
latex_all_lines.append(r"\end{lstlisting}")

output_all_file = os.path.join("data", "output", "prompt_variants_expanded.tex")
os.makedirs(os.path.dirname(output_all_file), exist_ok=True)
with open(output_all_file, 'w') as f:
    f.write('\n'.join(latex_all_lines))
print(f"LaTeX file for all prompt variants created successfully at {output_all_file}!")

# --- Step 7: Produce LaTeX output for poorly performing prompts ---
# (Here we group by dataset and temperature, and list only the prompt text.)
# For LaTeX display, we also format temperature as a string.
poor_results['temperature'] = poor_results['temperature'].map('{:.1f}'.format)
grouped = poor_results.groupby(['dataset', 'temperature'])

latex_poor_lines = [
    r"\lstset{breaklines=true, breakatwhitespace=true}",
    r"\begin{lstlisting}[label=lst:poor_performing_prompts]",
    r"% Poorly performing prompts (ka\_mean < 0.8) grouped by dataset and temperature:"
]

current_dataset = None
for (dataset, temperature), group in grouped:
    if dataset != current_dataset:
        if current_dataset is not None:
            latex_poor_lines.append("")  # blank line between datasets
        latex_poor_lines.append(f"Dataset: {dataset}")
        current_dataset = dataset
    prompts = ", ".join(group['prompt'].tolist())
    latex_poor_lines.append(f"\tTemperature {temperature}: {prompts}")
latex_poor_lines.append(r"\end{lstlisting}")

output_poor_tex = os.path.join("data", "output", "poor_performing_prompts.tex")
os.makedirs(os.path.dirname(output_poor_tex), exist_ok=True)
with open(output_poor_tex, 'w') as f:
    f.write('\n'.join(latex_poor_lines))
print(f"LaTeX file for poorly performing prompts created successfully at {output_poor_tex}!")

# --- Step 8: Produce CSV output for poorly performing prompts ---
# The CSV will include prompt, temperature, dataset, and the corresponding ka_mean.
output_poor_csv = os.path.join("data", "output", "poor_performing_prompts.csv")
os.makedirs(os.path.dirname(output_poor_csv), exist_ok=True)
poor_results.to_csv(output_poor_csv, index=False)
print(f"CSV file for poorly performing prompts created successfully at {output_poor_csv}!")