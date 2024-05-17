import os
import pandas as pd
from psa_temp import LLMWrapper
from psa_temp import PromptStabilityAnalysis
from transformers import AutoModelForCausalLM, AutoTokenizer
import simpledorff

import seaborn as sns
import matplotlib.pyplot as plt

# Data
MODEL = 'gpt-3.5-turbo'

try:
    # Try initializing the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    MAX_TOKENS = tokenizer.model_max_length
except OSError:
    # If the model identifier is not valid, set MAX_TOKENS to 16385
    MAX_TOKENS = 16385

df = pd.read_csv('UK_Manifestos.csv')
df_small = df.sample(10)
texts = [text[:MAX_TOKENS] for text in df_small['content'].values]

# Model
APIKEY = os.getenv("OPENAI_API_KEY")
MODEL = 'gpt-3.5-turbo'
llm = LLMWrapper(model = MODEL, apikey=APIKEY)
psa = PromptStabilityAnalysis(llm, texts, parse_function=lambda x: float(x), metric_fn = simpledorff.metrics.nominal_metric)

# Prompt
prompt = 'The text provided is a party manifesto for a political party in the United Kingdom. Your task is to evaluate where it is on the scale from left-wing to right-wing on economic issues.'
prompt_postfix = '[Respond with a number from 1 to 10. 1 corresponds to most left-wing. 10 corresponds to most right-wing. Guess if you do not know. Respond nothing else.]'


# Run
temperatures = [0.1, 1.0, 2.0, 4.0]

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
plt.savefig('graphs/ka_temp_manifestos.pdf')
plt.show()


# change to shorter sentences (chris file)
# run simpledorff with interval
