import pandas as pd
from utils import PromptStabilityAnalysis, get_openai_api_key
from transformers import AutoModelForCausalLM, AutoTokenizer
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
try:
    # Try initializing the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    MAX_TOKENS = tokenizer.model_max_length
except OSError:
    # If the model identifier is not valid, set MAX_TOKENS to 16385
    MAX_TOKENS = 16385

df = pd.read_csv('data/manifestos.csv')
# Filter for rows where the "Scale" column is "Economic"
df = df[df['scale'] == 'Economic']
df = df.sample(100, random_state=123)
example_data = list(df['sentence_context'].values)

psa = PromptStabilityAnalysis(annotation_function=annotate, data=example_data)

# Step 2: Construct the Prompt
original_text = (
    'The text provided is a party manifesto for a political party in the United Kingdom. '
    'Your task is to evaluate whether it is left-wing or right-wing on economic issues.'
)
prompt_postfix = (
    'Respond with 0 for left-wing, or 1 for right-wing. '
    'Respond nothing else.'
)

# Run baseline_stochasticity
ka_scores, annotated_data = psa.baseline_stochasticity(original_text, prompt_postfix, iterations=20, plot=True, save_path='plots/02a_manifestos_within.png', save_csv="data/annotated/manifestos_within.csv")

# Run interprompt_stochasticity
# Set temperatures
temperatures = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,  5.0]

# Get KA scores across different temperature paraphrasings
ka_scores, annotated_data = psa.interprompt_stochasticity(original_text, prompt_postfix, nr_variations=10, temperatures=temperatures, iterations = 1, print_prompts=False, plot=True, save_path='plots/02a_manifestos_between.png', save_csv = 'data/annotated/manifestos_between.csv')
