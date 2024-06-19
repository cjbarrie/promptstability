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
df = pd.read_csv('data/news_short.csv') # Same dataset but with txtlengths of less than 1000
df = df.sample(100, random_state=123) 
example_data = list(df['body'].values)

# Update parse_function to return nominal values
def parse_function(x):
    try:
        return int(x.strip())  # Ensure the result is treated as nominal
    except ValueError:
        return None  # Handle cases where conversion fails

psa = PromptStabilityAnalysis(annotation_function=annotate, data=example_data, parse_function=parse_function, metric_fn=simpledorff.metrics.nominal_metric)

# Step 2: Construct the Prompt
original_text = (
    'The text provided is some newspaper text. Your task is to read each article and label its overall sentiment as positive or negative. Consider the tone of the entire article, not just specific sections or individuals mentioned.'
)
prompt_postfix = '[Respond 0 for negative, 1 for positive, and 2 for neutral. Respond nothing else.]'

# Run baseline_stochasticity
ka_scores, annotated_data = psa.baseline_stochasticity(original_text, prompt_postfix, iterations=20, plot=True, save_path='plots/01b_news_within.png', save_csv="data/annotated/news_short_within.csv")

# Run interprompt_stochasticity
# Set temperatures
temperatures = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,  5.0]

# Get KA scores across different temperature paraphrasings
ka_scores, annotated_data = psa.interprompt_stochasticity(original_text, prompt_postfix, nr_variations=10, temperatures=temperatures, iterations = 1, print_prompts=False, plot=True, save_path='plots/01b_news_between.png', save_csv = 'data/annotated/news_short_between.csv')
