import pandas as pd
from utils import PromptStabilityAnalysis, get_openai_api_key
import matplotlib.pyplot as plt
from openai import OpenAI

APIKEY = get_openai_api_key()
MODEL = 'gpt-3.5-turbo'

# Example: We use the OpenAI API
client = OpenAI(
    api_key = get_openai_api_key()
)

def annotate(text, prompt, temperature=0.1):
    try:
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
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
df = pd.read_csv('data/profiles_short.csv')
df = df.sample(100, random_state=123)
example_data = list(df['profile'].values)


psa = PromptStabilityAnalysis(annotation_function=annotate, data=example_data)

# Step 2: Construct the Prompt
original_text = 'Let me know how you feel about the voters in the 2020 presidential election.' # One of the poor performing prompts at temp 3.0 
prompt_postfix = '[Respond 0 for Biden, or 1 for Trump. Guess if you do not know. Respond nothing else.]'

# Run baseline_stochasticity
ka_scores, annotated_data = psa.baseline_stochasticity(original_text, prompt_postfix, iterations=20, plot=True, save_path='plots/14_bad_synth_within.png', save_csv="data/annotated/bad_synth_short_within.csv")