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
df = pd.read_csv('data/profiles.csv')
df = df.sample(100, random_state=123)
example_data = list(df['profile'].values)


psa = PromptStabilityAnalysis(annotation_function=annotate, data=example_data)

# Step 2: Construct the Prompt
original_text = 'In the 2020 presidential election, Donald Trump is the Republican candidate, and Joe Biden is the Democratic candidate. The following is some information about an individual voter. I want you to tell me how you think they voted.'
prompt_postfix = '[Respond 0 for Biden, or 1 for Trump. Guess if you do not know. Respond nothing else.]'

# Run baseline_stochasticity
ka_scores, annotated_data = psa.baseline_stochasticity(original_text, prompt_postfix, iterations=20, plot=True, save_path='plots/05a_synth_within.png', save_csv="data/annotated/synth_within.csv")

# Run interprompt_stochasticity
# Set temperatures
temperatures = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,  5.0]

# Get KA scores across different temperature paraphrasings
ka_scores, annotated_data = psa.interprompt_stochasticity(original_text, prompt_postfix, nr_variations=10, temperatures=temperatures, iterations = 1, print_prompts=False, plot=True, save_path='plots/05a_synth_between.png', save_csv = 'data/annotated/synth_between.csv')