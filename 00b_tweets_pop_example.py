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
df = pd.read_csv('data/tweets.csv')
# df = df.sample(100, random_state=123)
sample_size = min(500, len(df))
df = df.sample(sample_size, random_state=123)
example_data = list(df['text'].values)

psa = PromptStabilityAnalysis(annotation_function=annotate, data=example_data)

# Step 2: Construct the Prompt
original_text = 'The following is a Twitter message written either by a Republican or a Democrat before the 2020 election. Your task is to label whether or not it contains populist language.'
prompt_postfix = '[Respond 0 if it does not contain populist language, and 1 if it does contain populist language. Guess if you do not know. Respond nothing else.]'

# Run intra_pss
# ka_scores, annotated_data = psa.intra_pss(original_text, prompt_postfix, iterations=20, plot=True, save_path='plots/00b_tweets_pop_within.png', save_csv="data/annotated/tweets_pop_within.csv")
ka_scores, annotated_data = psa.intra_pss(original_text, prompt_postfix, iterations=30, plot=True, save_path='plots/00b_tweets_pop_within_expanded.png', save_csv="data/annotated/tweets_pop_within_expanded.csv")

# Run inter_pss
# Set temperatures
# temperatures = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,  5.0]
temperatures = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5, 4.8, 5.0]


# ka_scores, annotated_data = psa.inter_pss(original_text, prompt_postfix, nr_variations=10, temperatures=temperatures, iterations = 1, print_prompts=False, plot=True, save_path='plots/00b_tweets_pop_between.png', save_csv = 'data/annotated/tweets_pop_between.csv')
ka_scores, annotated_data = psa.inter_pss(original_text, prompt_postfix, nr_variations=10, temperatures=temperatures, iterations = 3, print_prompts=False, plot=True, save_path='plots/00b_tweets_pop_between_expanded.png', save_csv = 'data/annotated/tweets_pop_between_expanded.csv')
