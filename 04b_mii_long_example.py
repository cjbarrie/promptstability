import pandas as pd
from utils import PromptStabilityAnalysis, get_openai_api_key
import matplotlib.pyplot as plt
import simpledorff
from openai import OpenAI

# Example: We here use the OpenAI API. You can provide any annotation function.
APIKEY = get_openai_api_key()
MODEL = 'gpt-3.5-turbo'
client = OpenAI(
    api_key = get_openai_api_key()
)

def annotate(text, prompt, temperature=0.1):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=temperature,
            messages=[
                {"role": "system", "content": prompt}, 
                {"role": "user", "content": text}
            ]
        )
    except Exception as e:
        print(f"Caught exception: {e}")
        raise e

    return ''.join(choice.message.content for choice in response.choices)


# Data
df = pd.read_csv('data/mii_long.csv')
df = df.sample(100, random_state=123)
example_data = list(df['MII_textW'].values)

# Update parse_function to return nominal values
def parse_function(x):
    try:
        return int(x.strip())  # Ensure the result is treated as nominal
    except ValueError:
        return None  # Handle cases where conversion fails

psa = PromptStabilityAnalysis(annotation_function=annotate, data=example_data, parse_function=parse_function, metric_fn=simpledorff.metrics.nominal_metric)

# Step 2: Construct the Prompt
original_text = (
    'Here are some open-ended responses from a scientific study of voters to the question "what is the most important issue facing the country?". Please assign one of the following categories to each open ended text response.'
)
prompt_postfix = '[Respond 48 for Coronavirus, 15 for Europe, 32 for Living costs, 40 for Environment, 26 for Economy-general, 12 for Immigration, 4 for Pol-neg i.e., complaints about politics, 1 for Health, 31 for Inflation, 22 for War, 5 for Partisan-neg i.e., complaints about a party of politician, and 14 for Crime. Respond nothing else.]'

# Run baseline_stochasticity
ka_scores, annotated_data = psa.baseline_stochasticity(original_text, prompt_postfix, iterations=20, plot=True, save_path='plots/04b_mii_within.png', save_csv="data/annotated/mii_long_within.csv")

# Run interprompt_stochasticity
# Set temperatures
temperatures = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,  5.0]

# Get KA scores across different temperature paraphrasings
ka_scores, annotated_data = psa.interprompt_stochasticity(original_text, prompt_postfix, nr_variations=10, temperatures=temperatures, iterations = 1, print_prompts=False, plot=True, save_path='plots/04b_mii_between.png', save_csv = 'data/annotated/mii_long_between.csv')