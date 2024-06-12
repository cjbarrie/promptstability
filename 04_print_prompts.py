import os
import re

# List of file names
files = [
    "00a_tweets_rd_example.py",
    "00b_tweets_pop_example.py",
    "01_news_example.py",
    "02a_manifestos_example.py",
    "02b_manifestos_multi_example.py"
]

# Function to extract the required strings
def extract_strings(file_content):
    # Pattern to match the CSV file path
    csv_pattern = re.compile(r'"([^"]*\.csv)"')
    original_text_pattern = re.compile(r"original_text\s*=\s*\((.*?)\)", re.DOTALL)
    prompt_postfix_pattern = re.compile(r"prompt_postfix\s*=\s*\((.*?)\)|prompt_postfix\s*=\s*['\"](.*?)['\"]", re.DOTALL)

    # Find the CSV file name
    csv_match = csv_pattern.search(file_content)
    dataset_name = os.path.basename(csv_match.group(1)) if csv_match else "Unknown"

    # Extract original_text and prompt_postfix
    original_text_match = original_text_pattern.search(file_content)
    prompt_postfix_match = prompt_postfix_pattern.search(file_content)
    
    if original_text_match:
        original_text = ''.join(original_text_match.group(1).split('\n')).strip().replace("'", "").replace('"', '')
    else:
        original_text_match = re.search(r"original_text\s*=\s*['\"](.*?)['\"]", file_content, re.DOTALL)
        original_text = original_text_match.group(1) if original_text_match else ""

    if prompt_postfix_match:
        if prompt_postfix_match.group(1):  # Check if we captured the multi-line case
            prompt_postfix = ''.join(prompt_postfix_match.group(1).split('\n')).strip().replace("'", "").replace('"', '')
        else:  # Single-line case
            prompt_postfix = prompt_postfix_match.group(2).strip()
    else:
        prompt_postfix = ""

    return dataset_name, original_text, prompt_postfix

# Loop through each file and extract the strings
latex_output = [
    "\\lstset{breaklines=true, breakatwhitespace=true}",
    "\\begin{lstlisting}",
    "% Format: Dataset \t Prompt"
]

for file in files:
    with open(file, 'r') as f:
        content = f.read()
        dataset_name, original_text, prompt_postfix = extract_strings(content)
        prompt = f"{original_text} {prompt_postfix}".strip()
        latex_output.append(f"{dataset_name}\t{prompt}")

latex_output.append("\\end{lstlisting}")

# Ensure the output directory exists
os.makedirs("data/output", exist_ok=True)

# Write the LaTeX output to a file in the data/output/ directory
output_file_path = "data/output/original_prompts.tex"
with open(output_file_path, 'w') as f:
    f.write("\n".join(latex_output))

print(f"LaTeX output has been written to {output_file_path}")
