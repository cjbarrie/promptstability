import pandas as pd
import matplotlib.pyplot as plt
import simpledorff
from utils import PromptStabilityAnalysis, get_openai_api_key
from openai import OpenAI

# Set up the OpenAI client to use GPT-4o
APIKEY = get_openai_api_key()
MODEL = 'gpt-4o-mini'  
client = OpenAI(api_key=APIKEY)

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

# Define common temperature settings and number of variations
temperatures = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
nr_variations = 10
iterations = 1

# -----------------------------------------------------------------------------
# Dataset 1: News
# -----------------------------------------------------------------------------
df_news = pd.read_csv('data/news_short.csv')
df_news = df_news.sample(100, random_state=123)
example_data_news = list(df_news['body'].values)

def parse_function(x):
    try:
        return int(x.strip())  # Ensure the result is treated as nominal
    except ValueError:
        return None  # In case conversion fails

psa_news = PromptStabilityAnalysis(
    annotation_function=annotate,
    data=example_data_news,
    parse_function=parse_function,
    metric_fn=simpledorff.metrics.nominal_metric
)

# Step 2: Construct the Prompt
original_text_news_short = (
    'The text provided is some newspaper text. Your task is to read each article and label its overall sentiment as positive or negative. Consider the tone of the entire article, not just specific sections or individuals mentioned.'
)
prompt_postfix_news_short = '[Respond 0 for negative, 1 for positive, and 2 for neutral. Respond nothing else.]'

ka_scores_news, annotated_data_news = psa_news.inter_pss(
    original_text=original_text_news_short,
    prompt_postfix=prompt_postfix_news_short,
    nr_variations=nr_variations,
    temperatures=temperatures,
    iterations=iterations,
    save_csv='data/annotated/news_short_between_updated.csv'
)

# # -----------------------------------------------------------------------------
# # Dataset 1: News
# # -----------------------------------------------------------------------------
# df_news = pd.read_csv('data/news.csv')
# df_news = df_news.sample(100, random_state=123)
# example_data_news = list(df_news['body'].values)

# def parse_function(x):
#     try:
#         return int(x.strip())  # Ensure the result is treated as nominal
#     except ValueError:
#         return None  # In case conversion fails

# psa_news = PromptStabilityAnalysis(
#     annotation_function=annotate,
#     data=example_data_news,
#     parse_function=parse_function,
#     metric_fn=simpledorff.metrics.nominal_metric
# )

# original_text_news = (
#     'The text provided is some newspaper text. Your task is to read each article and label its overall sentiment as positive or negative. '
#     'Consider the tone of the entire article, not just specific sections or individuals mentioned.'
# )
# prompt_postfix_news = '[Respond 0 for negative, 1 for positive, and 2 for neutral. Respond nothing else.]'

# ka_scores_news, annotated_data_news = psa_news.inter_pss(
#     original_text=original_text_news,
#     prompt_postfix=prompt_postfix_news,
#     nr_variations=nr_variations,
#     temperatures=temperatures,
#     iterations=iterations,
#     save_path='plots/01a_news_between_updated.png',
#     save_csv='data/annotated/news_between_updated.csv'
# )

# # -----------------------------------------------------------------------------
# # Dataset 2: Stance
# # -----------------------------------------------------------------------------
# df_stance = pd.read_csv('data/stance.csv')
# df_stance = df_stance.sample(100, random_state=123)
# example_data_stance = list(df_stance['Tweet'].values)

# psa_stance = PromptStabilityAnalysis(
#     annotation_function=annotate,
#     data=example_data_stance,
#     parse_function=parse_function,
#     metric_fn=simpledorff.metrics.nominal_metric
# )

# original_text_stance = (
#     'The text provided come from some tweets about Donald Trump. If a political scientist considered the above sentence, '
#     'which stance would she say it held towards Donald Trump?'
# )
# prompt_postfix_stance = '[Respond 0 for negative, 1 for positive, and 2 for none. Respond nothing else.]'

# ka_scores_stance, annotated_data_stance = psa_stance.inter_pss(
#     original_text=original_text_stance,
#     prompt_postfix=prompt_postfix_stance,
#     nr_variations=nr_variations,
#     temperatures=temperatures,
#     iterations=iterations,
#     save_path='plots/03b_stance_between_updated.png',
#     save_csv='data/annotated/stance_long_between_updated.csv'
# )

# # -----------------------------------------------------------------------------
# # Dataset 3: Profiles
# # -----------------------------------------------------------------------------
# df_profiles = pd.read_csv('data/profiles_short.csv')
# df_profiles = df_profiles.sample(100, random_state=123)
# example_data_profiles = list(df_profiles['profile'].values)

# # For this dataset, no parse_function or metric_fn was provided in your original code.
# psa_profiles = PromptStabilityAnalysis(
#     annotation_function=annotate,
#     data=example_data_profiles
# )

# original_text_profiles = (
#     'In the 2020 presidential election, Donald Trump is the Republican candidate, and Joe Biden is the Democratic candidate. '
#     'The following is some information about an individual voter. I want you to tell me how you think they voted.'
# )
# prompt_postfix_profiles = '[Respond 0 for Biden, or 1 for Trump. Guess if you do not know. Respond nothing else.]'

# ka_scores_profiles, annotated_data_profiles = psa_profiles.inter_pss(
#     original_text=original_text_profiles,
#     prompt_postfix=prompt_postfix_profiles,
#     nr_variations=nr_variations,
#     temperatures=temperatures,
#     iterations=iterations,
#     save_path='plots/05b_synth_between_updated.png',
#     save_csv='data/annotated/synth_short_between_updated.csv'
# )