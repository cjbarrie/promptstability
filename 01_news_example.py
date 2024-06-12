import pandas as pd
from utils import LLMWrapper, PromptStabilityAnalysis, get_openai_api_key
import matplotlib.pyplot as plt
import simpledorff

APIKEY = get_openai_api_key()
MODEL = 'gpt-3.5-turbo'

# Data
df = pd.read_csv('data/news.csv')
df = df.sample(100, random_state=123) #TODO change back to 100
example_data = list(df['body'].values)

# Update parse_function to return nominal values
def parse_function(x):
    try:
        return int(x.strip())  # Ensure the result is treated as nominal
    except ValueError:
        return None  # Handle cases where conversion fails

llm = LLMWrapper(apikey=APIKEY, model=MODEL)
psa = PromptStabilityAnalysis(llm=llm, data=example_data, parse_function=parse_function, metric_fn=simpledorff.metrics.nominal_metric)

# Step 2: Construct the Prompt
original_text = (
    'The text provided is some newspaper text. Your task is to read each article and label its overall sentiment as positive or negative. Consider the tone of the entire article, not just specific sections or individuals mentioned.'
)
prompt_postfix = '[Respond 0 for negative, 1 for positive, and 2 for neutral. Respond nothing else.]'

# Run baseline_stochasticity
ka_scores, annotated_data = psa.baseline_stochasticity(original_text, prompt_postfix, iterations=20, plot=True, save_path='plots/01_news_within.png', save_csv="data/annotated/news_within.csv")

# Run interprompt_stochasticity
# Set temperatures
temperatures = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,  5.0]

# Get KA scores across different temperature paraphrasings
ka_scores, annotated_data = psa.interprompt_stochasticity(original_text, prompt_postfix, nr_variations=10, temperatures=temperatures, iterations = 1, print_prompts=False, plot=True, save_path='plots/01_news_between.png', save_csv = 'data/annotated/news_between.csv')
