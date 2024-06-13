import pandas as pd
import glob
import re

# Directory containing the files
directory = 'data/annotated/'

# Find all _between files in the directory
files = glob.glob(f'{directory}/*_between.csv')

# Initialize a list to hold the results
results = []

for file in files:
    # Read the CSV file
    df = pd.read_csv(file)
    
    # Extract the dataset name from the file name
    dataset_name = file.split('/')[-1]
    
    # Round temperature to 1 decimal place
    df['temperature'] = df['temperature'].round(1)
    
    # Select the unique combinations of 'prompt' and 'temperature'
    unique_combinations = df[['prompt', 'temperature']].drop_duplicates()
    
    # Escape LaTeX special characters in the prompt column
    unique_combinations['prompt'] = unique_combinations['prompt'].str.replace('_', r'\_', regex=True)
    
    # Add the dataset name to each entry
    unique_combinations['dataset'] = dataset_name
    
    # Append to the results list
    results.append(unique_combinations)

# Concatenate all results into a single DataFrame
all_results = pd.concat(results, ignore_index=True)

# Convert temperature to string with 1 decimal place
all_results['temperature'] = all_results['temperature'].map('{:.1f}'.format)

# Format the output
output_lines = [
    r"\lstset{breaklines=true, breakatwhitespace=true}",
    r"\begin{lstlisting}",
    r"% Format: Dataset \t Temperature \t Prompt"
]
for index, row in all_results.iterrows():
    output_lines.append(f"{row['dataset']}\t{row['temperature']}\t{row['prompt']}")

# Add the closing line for the listing environment
output_lines.append(r"\end{lstlisting}")

# Save to a .txt file
with open('data/output/prompt_variants.tex', 'w') as f:
    f.write('\n'.join(output_lines))

print("LaTeX table created successfully!")
