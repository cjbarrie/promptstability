import pandas as pd
from utils import LLMWrapper, PromptStabilityAnalysis, get_openai_api_key
import matplotlib.pyplot as plt

# Baseline stochasticity

## Usage example
APIKEY = get_openai_api_key()
MODEL = 'gpt-3.5-turbo'

# Data
df = pd.read_csv('data/reviews.csv')
df = df.sample(10)
example_data = list(df['sentence'].values)

llm = LLMWrapper(apikey=APIKEY, model=MODEL)
psa = PromptStabilityAnalysis(llm=llm, data=example_data)

# Step 2: Construct the Prompt
original_text = (
    'The text provided is a single sentence extracted from a movie review. '
    'Your task is to assess the sentiment of the text as either positive or negative.'
)
prompt_postfix = '[Respond 0 for negative, or 1 for positive. Respond nothing else.]'

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
plt.savefig('plots/01_reviews_within.pdf')
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
plt.savefig('plots/01_reviews_between.pdf')
plt.show()
