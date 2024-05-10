import os
import pandas as pd
from utils import LLMWrapper
from utils import PromptStabilityAnalysis
from transformers import AutoModelForCausalLM, AutoTokenizer
import simpledorff

import seaborn as sns
import matplotlib.pyplot as plt

# Data
df = pd.read_csv('data/tweets.csv')
df = df.sample(10)
texts = list(df['text'].values)

# TODO: add into main library functions
# Get API key from environment variables
def get_openai_api_key():
    """Retrieve OpenAI API key from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
    return api_key


# Baseline stochasticity

## Usage example
APIKEY = get_openai_api_key()
MODEL = 'gpt-3.5-turbo'

# Data
df = pd.read_csv('data/tweets.csv')
df = df.sample(10)
example_data = list(df['text'].values)

llm = LLMWrapper(apikey=APIKEY, model=MODEL)
psa = PromptStabilityAnalysis(llm=llm, data=example_data)

# Step 2: Construct the Prompt
original_text = 'The following is a Twitter message written either by a Republican or a Democrat before the 2020 election. Your task is to guess whether the author is Republican or Democrat.'
prompt_postfix = '[Respond 0 for Democrat, or 1 for Republican. Guess if you do not know. Respond nothing else.]'

# Run baseline_stochasticity
KA, df, ka_scores, iterrations_no = psa.baseline_stochasticity(original_text, prompt_postfix, iterations=20)

# Model
llm = LLMWrapper(model = MODEL, apikey=APIKEY)
psa = PromptStabilityAnalysis(llm, texts, parse_function=lambda x: float(x), metric_fn = simpledorff.metrics.nominal_metric)

# Prompt
prompt = 'The following is a Twitter message written either by a Republican or a Democrat before the 2020 election. Your task is to guess whether the author is Republican or Democrat.'
prompt_postfix = '[Respond 0 for Democrat, or 1 for Republican. Guess if you do not know. Respond nothing else.]'

# TODO: add this into library as function
# Function to plot KA scores with integer x-axis labels
def plot_ka_scores(ka_scores, overall_ka):
    iterations = list(range(2, 2 + len(ka_scores)))

    plt.figure(figsize=(10, 5))
    plt.plot(iterations, ka_scores, marker='o', linestyle='-', color='b', label='KA Score per Iteration')
    plt.axhline(y=overall_ka, color='r', linestyle='--', label=f'Overall KA: {overall_ka:.2f}')
    plt.xlabel('Iteration')
    plt.ylabel('Krippendorff\'s Alpha (KA)')
    plt.title('Krippendorff\'s Alpha Scores Across Iterations')
    plt.xticks(iterations)  # Set x-axis ticks to be whole integers
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot the KA scores
plot_ka_scores(ka_scores, KA)







# Interprompt stochasticity
## Usage example
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
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.show()
