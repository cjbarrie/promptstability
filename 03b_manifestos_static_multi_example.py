import pandas as pd
from utils import LLMWrapper, PromptStabilityAnalysis, get_openai_api_key
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import simpledorff


# Baseline stochasticity

## Usage example
APIKEY = get_openai_api_key()
MODEL = 'gpt-3.5-turbo'

# Data
try:
    # Try initializing the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    MAX_TOKENS = tokenizer.model_max_length
except OSError:
    # If the model identifier is not valid, set MAX_TOKENS to 16385
    MAX_TOKENS = 16385

df = pd.read_csv('data/manifestos_static.csv')
df = df.sample(10)
example_data = list(df['sentence_context'].values)

llm = LLMWrapper(apikey=APIKEY, model=MODEL)
psa = PromptStabilityAnalysis(llm=llm, data=example_data, parse_function=lambda x: float(x), metric_fn=simpledorff.metrics.interval_metric)

# Step 2: Construct the Prompt
original_text = (
    'The text provided is part of a party manifesto of a political party in the United Kingdom.'
    'In each piece of text, there are several sentences. One of these is capitalized and this is the text you should focus on.'
    'The other sentences are there solely to provide some context.'
    'Your task is to evaluate where the capitalized sentence is on the scale from left-wing to right-wing on economic issues.'
)
prompt_postfix = (
    'Respond with a number from 1 to 10. 1 corresponds to most left-wing. 10 corresponds to most right-wing. '
    'Respond nothing else.'
)

# Run baseline_stochasticity
KA, df, ka_scores, iterrations_no = psa.baseline_stochasticity(original_text, prompt_postfix, iterations=20)

# Function to plot KA scores with integer x-axis labels
iterations = list(range(2, 2 + len(ka_scores)))

plt.figure(figsize=(10, 5))
plt.plot(iterations, ka_scores, marker='o', linestyle='-', color='b', label='KA Score per Iteration')
plt.axhline(y=KA, color='r', linestyle='--', label=f'Overall KA: {KA:.2f}')
plt.xlabel('Iteration')
plt.ylabel("Krippendorff's Alpha (KA)")
plt.title("Krippendorff's Alpha Scores Across Iterations")
plt.xticks(iterations)  # Set x-axis ticks to be whole integers
plt.legend()
plt.grid(True)
plt.axhline(y=0.8, color='black', linestyle='--', linewidth=.5)
plt.savefig('plots/03b_manifestos_static_multi_within.pdf')
plt.show()

# Interprompt stochasticity
## Usage example

# Set temperatures
temperatures = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,  5.0]

# Get KA scores across different temperature paraphrasings
ka_scores, annotated_data = psa.interprompt_stochasticity(original_text, prompt_postfix, nr_variations=10, temperatures=temperatures, iterations = 10)

# Extract temperatures (keys) and KA scores (values)
temperatures = list(ka_scores.keys())
ka_values = list(ka_scores.values())

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(temperatures, ka_values, marker='o', linestyle='-', color='b')
plt.xlabel('Temperature')
plt.ylabel('Krippendorff\'s Alpha (KA)')
plt.title('Krippendorff\'s Alpha Scores Across Temperatures')
plt.xticks(temperatures)  # Set x-axis ticks to be whole integers
plt.grid(True)
plt.ylim(0.0, 1.05)
plt.axhline(y=0.80, color='black', linestyle='--', linewidth=.5)
plt.savefig('plots/03b_manifestos_static_multi_between.pdf')
plt.show()
