import pandas as pd
from utils import PromptStabilityAnalysis, get_openai_api_key
import matplotlib.pyplot as plt
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
df = pd.read_csv('data/stance_long.csv')
df = df.sample(100, random_state=123)
example_data = list(df['Tweet'].values)

psa = PromptStabilityAnalysis(annotation_function=annotate, data=example_data)

# Step 2: Construct the Prompt
original_text = (
    'The text provided come from some tweets about Donald Trump. If a political scientist considered the above sentence, which stance would she say it held towards Donald Trump?'
)
prompt_postfix = '[Respond 0 for negative, and 1 for positive. Respond nothing else.]'

# Run baseline_stochasticity
ka_scores, annotated_data = psa.baseline_stochasticity(original_text, prompt_postfix, iterations=20, plot=True, save_path='plots/03a_stance_within.png', save_csv="data/annotated/stance_long_within.csv")

# Run interprompt_stochasticity
# Set temperatures
temperatures = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,  5.0]

# Get KA scores across different temperature paraphrasings
ka_scores, annotated_data = psa.interprompt_stochasticity(original_text, prompt_postfix, nr_variations=10, temperatures=temperatures, iterations = 1, print_prompts=False, plot=True, save_path='plots/03a_stance_between.png', save_csv = 'data/annotated/stance_long_between.csv')