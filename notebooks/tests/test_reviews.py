import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
from psa_temp import LLMWrapper, PromptStabilityAnalysis

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import simpledorff

import seaborn as sns
import matplotlib.pyplot as plt

# Data
df = pd.read_csv('sst2_reviews.csv')
df_small = df.sample(10)
texts = list(df_small['sentence'].values)

# Model
APIKEY = os.getenv("OPENAI_API_KEY")
MODEL = 'gpt-3.5-turbo'
llm = LLMWrapper(model = MODEL, apikey=APIKEY)
psa = PromptStabilityAnalysis(llm, texts, parse_function=lambda x: float(x), metric_fn = simpledorff.metrics.nominal_metric)

# Prompt
prompt = 'The text provided is a single sentence extracted from a movie review. Your task is to assess the sentiment of the review as either positive or negative.'
prompt_postfix = '[Respond 0 for negative, or 1 for positive. Respond nothing else.]'

# Run
temperatures = [0.1, 1.0, 2.0, 4.0]

# Initialize a DataFrame to store the comprehensive results
all_results_df = pd.DataFrame()

for temp in temperatures:
    print(f"Analyzing at temperature: {temp}")
    res, df, rel_vs_sim, poor_prompts = psa.interprompt_stochasticity(prompt, prompt_postfix, nr_variations=5, temperature=temp, iterations=1)

    # Append the current rel_by_sim DataFrame to the comprehensive results DataFrame
    all_results_df = pd.concat([all_results_df, rel_vs_sim], ignore_index=True)

print(all_results_df)

sns.set_theme(style="whitegrid", font_scale=1.5)
plt.figure(figsize=(10, 6), dpi=300)  # Larger plot with high resolution
sns.lineplot(data=all_results_df, x='temperature', y='KA_by_temp', marker='o', color='b', linewidth=2, markersize=8)
plt.title("Krippendorff's Alpha vs. Temperature", fontsize=20)
plt.xlabel("Temperature", fontsize=15)
plt.ylabel("Krippendorff's Alpha (KA)", fontsize=15)
plt.grid(True, which='both', linestyle='--', linewidth=.7)
plt.ylim(0.0, 1.05)
plt.axhline(y=0.80, color='black', linestyle='--', linewidth=.5)
plt.savefig('graphs/ka_temp_reviews_binary.pdf')
plt.show()
