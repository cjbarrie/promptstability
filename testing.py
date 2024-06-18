import pandas as pd
from utils import PromptStabilityAnalysis, get_openai_api_key
import matplotlib.pyplot as plt
from openai import OpenAI


# Baseline stochasticity

## Usage example:  We use the OpenAI API
APIKEY = get_openai_api_key()
MODEL = 'gpt-3.5-turbo'

client = OpenAI(
    api_key = APIKEY
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
df = pd.read_csv('data/tweets.csv')
df = df.sample(10, random_state=123)
example_data = list(df['text'].values)

psa = PromptStabilityAnalysis(annotation_function=annotate, data=example_data)

# Step 2: Construct the Prompt
original_text = 'The following is a Twitter message written either by a Republican or a Democrat before the 2020 election. Your task is to guess whether the author is Republican or Democrat.'
prompt_postfix = '[Respond 0 for Democrat, or 1 for Republican. Guess if you do not know. Respond nothing else.]'

# Get KA scores across different temperature paraphrasings
temperatures = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,  5.0]

#ka_scores, annotated_data = psa.interprompt_stochasticity(original_text, prompt_postfix, nr_variations=5, temperatures=temperatures, iterations = 1, print_prompts=False, edit_prompts_path='/Users/ellipalaiologou/Downloads/test_prompts.csv',plot=True, save_path='/Users/ellipalaiologou/Downloads/test_plot.png', save_csv = '/Users/ellipalaiologou/Downloads/test_data.csv')
ka_scores, annotated_data = psa.manual_interprompt_stochasticity(edit_prompts_path='/Users/ellipalaiologou/Downloads/test_prompts.csv', plot=True, save_path='/Users/ellipalaiologou/Downloads/testmanual_plot.png', save_csv = '/Users/ellipalaiologou/Downloads/testmanual_data.csv')
