import pandas as pd
from utils import LLMWrapper, PromptStabilityAnalysis, get_openai_api_key

# Get API key from environment variables
def get_openai_api_key():
    """Retrieve OpenAI API key from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
    return api_key

APIKEY = get_openai_api_key()
MODEL = 'gpt-3.5-turbo'

# Data
df = pd.read_csv('data/tweets.csv')
df = df.sample(10)
example_data = list(df['text'].values)

llm = LLMWrapper(apikey=APIKEY, model=MODEL)
psa = PromptStabilityAnalysis(llm=llm, data=example_data)

# Step 2: Construct the Prompt
original_text = 'The following is a Twitter message written either by a Republican or a Democrat before the 2020 election. Your task is to guess whether the author is Republican or Democrat.'
prompt_postfix = '[Respond 0 for Democrat, or 1 for Republican. Guess if you do not know. Respond nothing else.]'
prompt = f'{original_text} {prompt_postfix}'
print("Constructed Prompt:", prompt)

# Step 3: Annotate Data for One Iteration
annotated = []
iteration = 0

for j, d in enumerate(psa.data):
    annotation = psa.llm.annotate(d, prompt, parse_function=None)
    annotated.append({'id': j, 'text': d, 'annotation': annotation, 'iteration': iteration})

print("Annotated Data (Iteration 0):")
for entry in annotated:
    print(entry)

# Step 4: Convert Annotated Data to DataFrame
df = pd.DataFrame(annotated)
print("DataFrame of Annotated Data:")
print(df)

# Step 5: Calculate Krippendorff's Alpha
if iteration > 0:
    KA = simpledorff.calculate_krippendorffs_alpha_for_df(
        df,
        metric_fn=simpledorff.metrics.nominal_metric,
        experiment_col='id',
        annotator_col='iteration',
        class_col='annotation'
    )
    print("Krippendorff's Alpha (KA):", KA)
else:
    KA = None

# Step 6: Iterate Over Multiple Iterations
iterations = 10
ka_scores = []
iterrations_no = []

for i in range(1, iterations):
    print(f"Iteration {i}/{iterations}...", end='\r')
    sys.stdout.flush()
    for j, d in enumerate(psa.data):
        annotation = psa.llm.annotate(d, prompt, parse_function=None)
        annotated.append({'id': j, 'text': d, 'annotation': annotation, 'iteration': i})

    df = pd.DataFrame(annotated)
    KA = simpledorff.calculate_krippendorffs_alpha_for_df(
        df,
        metric_fn=simpledorff.metrics.nominal_metric,
        experiment_col='id',
        annotator_col='iteration',
        class_col='annotation'
    )
    ka_scores.append(KA)
    iterrations_no.append(i + 1)

print('Finished classifications.')
print(f'Within-prompt KA score for {i + 1} repetitions is {KA}')
print("KA Scores:", ka_scores)
print()

## Note:

# KA = simpledorff.calculate_krippendorffs_alpha_for_df(
#     df,
#     metric_fn=self.metric_fn,
#     experiment_col='id',
#     annotator_col='iteration',
#     class_col='annotation')

# This is treating each iteration as a new annotator. It is then getting the KA for each run, calculated as, the overall agreement between iteration
# 1 and iteration 0 then iteration 2 and iteration 0. 













import pandas as pd
from utils import LLMWrapper, get_openai_api_key
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import simpledorff

import pandas as pd
import numpy as np
import openai
import time
import sys
import os
from openai import OpenAI

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import sentencepiece


class PromptStabilityAnalysis:

    def __init__(self, llm, data, metric_fn=simpledorff.metrics.nominal_metric, parse_function=None) -> None:
        self.llm = llm
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        model_name = 'tuner007/pegasus_paraphrase'
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.torch_device)
        self.parse_function = parse_function
        self.data = data
        self.metric_fn = metric_fn

    def __paraphrase_sentence(self, input_text, num_return_sequences=10, num_beams=50, temperature=1.0):
        batch = self.tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(self.torch_device)
        translated = self.model.generate(**batch, max_length=60, num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=temperature, do_sample=True)
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    def __generate_paraphrases(self, original_text, prompt_postfix, nr_variations, temperature=1.0):
        phrases = self.__paraphrase_sentence(original_text, num_return_sequences=nr_variations, temperature=temperature)
        l = [{'phrase': f'{original_text} {prompt_postfix}', 'original': True}]
        for phrase in phrases:
            l.append({'phrase': f'{phrase} {prompt_postfix}', 'original': False})
        self.paraphrases = pd.DataFrame(l)
        return self.paraphrases
    
    def baseline_stochasticity(self, original_text, prompt_postfix, iterations=10):
        prompt = f'{original_text} {prompt_postfix}'
        annotated = []
        ka_scores = []
        iterrations_no = []

        for i in range(iterations):
            print(f"Iteration {i}/{iterations}...", end='\r')
            sys.stdout.flush()
            for j, d in enumerate(self.data):
                annotation = self.llm.annotate(d, prompt, parse_function=self.parse_function)
                annotated.append({'id': j, 'text': d, 'annotation': annotation, 'iteration': i})

            if i > 0:
                df = pd.DataFrame(annotated)
                KA = simpledorff.calculate_krippendorffs_alpha_for_df(
                    df,
                    metric_fn=self.metric_fn,
                    experiment_col='id',
                    annotator_col='iteration',
                    class_col='annotation')
                ka_scores.append(KA)
                iterrations_no.append(i + 1)
                # Add KA score to each row of this iteration
                for k in range(len(annotated)):
                    if annotated[k]['iteration'] == i:
                        annotated[k]['KA'] = KA

        print()
        print('Finished classifications.')
        print("KA Scores:", ka_scores)
        print(f'Within-prompt KA score for {i + 1} repetitions is {KA}')

        return KA, df, ka_scores, iterrations_no

    def interprompt_stochasticity(self, original_text, prompt_postfix, nr_variations=5, temperatures=[0.5, 0.7, 0.9], iterations=1):
        ka_scores = {}
        
        for temp in temperatures:
            paraphrases = self.__generate_paraphrases(original_text, prompt_postfix, nr_variations=nr_variations, temperature=temp)
            annotated = []
            for i, (paraphrase, original) in enumerate(zip(paraphrases['phrase'], paraphrases['original'])):
                print(f"Temperature {temp}, Iteration {i}/{nr_variations}...", end='\r')
                sys.stdout.flush()
                for j, d in enumerate(self.data):
                    annotation = self.llm.annotate(d, paraphrase, parse_function=self.parse_function)
                    annotated.append({'id': j, 'text': d, 'annotation': annotation, 'prompt_id': i, 'prompt': paraphrase, 'original': original})
            print()
            print('Finished classifications for temperature:', temp)
            annotated_data = pd.DataFrame(annotated)
            KA = simpledorff.calculate_krippendorffs_alpha_for_df(
                annotated_data, 
                metric_fn=self.metric_fn, 
                experiment_col='id', 
                annotator_col='prompt_id', 
                class_col='annotation')
            ka_scores[temp] = KA
        
        return ka_scores, annotated_data
    







    

# Example usage
api_key = get_openai_api_key()
llm = LLMWrapper(api_key, 'gpt-3.5-turbo')
# Data
df = pd.read_csv('data/tweets.csv')
df = df.sample(10)
example_data = list(df['text'].values)

psa = PromptStabilityAnalysis(llm, example_data)

temperatures = [0.1, 0.5, 1.0, 2.0, 4.0, 5.0]
original_text = 'The following is a Twitter message written either by a Republican or a Democrat before the 2020 election. Your task is to guess whether the author is Republican or Democrat.'
prompt_postfix = '[Respond 0 for Democrat, or 1 for Republican. Guess if you do not know. Respond nothing else.]'

ka_scores, annotated_data = psa.interprompt_stochasticity(original_text, prompt_postfix, nr_variations=5, temperatures=temperatures, iterations = 5)

scores = list(ka_scores.values())




def plot_ka_scores(ka_scores, temperatures):
  plt.plot(temperatures, ka_scores, marker='o', linestyle='-', color='b', label='KA Score')
  plt.xlabel('Temperatures')
  plt.ylabel('Krippendorff\'s Alpha (KA)')
  plt.title('KA Scores vs. Temperatures')
  plt.legend()
  plt.grid(True)
  plt.show()

  plot_ka_scores(ka_scores, temperatures)