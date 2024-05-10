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


class LLMWrapper:
    '''This is a wrapper class for LLMs, which provides a method called 'annotate' that annotates a given message using an LLM.
    '''

    def __init__(self,apikey, model, wait_time=0.8) -> None:
        self.apikey = apikey
        self.model = model

        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=apikey
        )


    def annotate(self, text, prompt, parse_function = None, temperature = 0.1):
        '''
        Annotate the given text in the way the prompt instructs you to.

        Parameters:
        - text (str): text you want classified
        - prompt (str): the classification prompt/instruction
        - temperature (float): how deterministic (low number) vs. random (higher number) your results should be
        - parse_function (method): a method that parses the resulting data.

        Returns:
        - model's response to prompt (classification outcome)
        '''
        failed = True
        tries = 0
        while(failed):
            try:
                response = self.client.chat.completions.create(
                    model = self.model,
                    temperature = temperature,
                    messages = [
                        {"role": "system", "content": f"'{prompt}'"}, #The system instruction tells the bot how it is supposed to behave
                        {"role": "user", "content": f"'{text}'"} #This provides the text to be analyzed.
                    ]
                )
                failed = False

            #Handle errors.
            #If the API gets an error, perhaps because it is overwhelmed, we wait 10 seconds and then we try again.
            # We do this 10 times, and then we give up.
            except openai.APIError as e:
                print(f"OpenAI API returned an API Error: {e}")

                if tries < 10:
                    print(f"Caught an APIError: {e}. Waiting 10 seconds and then trying again...")
                    failed = True
                    tries += 1
                    time.sleep(10)
                else:
                    print(f"Caught an APIError: {e}. Too many exceptions. Giving up.")
                    raise e

            except openai.APIConnectionError as e:
                print(f"Failed to connect to OpenAI API: {e}")
                pass
            except openai.RateLimitError as e:
                print(f"OpenAI API request exceeded rate limit: {e}")
                pass

            #If the text is too long, we truncate it and try again. Note that if you get this error, you probably want to chunk your texts.
            except openai.BadRequestError as e:
                #Shorten request text
                print(f"Received a InvalidRequestError. Request likely too long. {e}")
                raise e

            except Exception as e:
                print(f"Caught unhandled error. {e}")
                raise e

        result = ''
        for choice in response.choices:
            result += choice.message.content

        # Parse the result using provided function
        if parse_function is not None:
            result = parse_function(result)

        return result


class PromptStabilityAnalysis:

    def __init__(self,llm, data, metric_fn=simpledorff.metrics.nominal_metric, parse_function=None) -> None:

        self.llm = llm

        # Get a number for the similarity between two sentences
        self.embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        #Initiate paraphraser
        model_name = 'tuner007/pegasus_paraphrase'
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.torch_device)
        self.parse_function = parse_function  #The function through which to parse the result from the LLM
        self.data = data # The data to be analyzed. Should be a list of texts.
        self.metric_fn = metric_fn #Metric function for KA. e.g., simpledorff.metrics.interval_metric or nominal_metric metric_fn=simpledorff.metrics.nominal_metric

    # Uses Pegasus to paraphrase a sentence
    def __paraphrase_sentence(self, input_text, num_return_sequences=10, num_beams=50, temperature=1.0):
        batch = self.tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(self.torch_device)
        translated = self.model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=temperature, do_sample=True)
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    #Generate paraphrases
    #prompt_postfix is a fixed addition that is not paraphrased
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

    # CHANGE to include loop of temperatures or right new function that has interprompt stochasticity in loop (like __generate_paraphrases)
    def interprompt_stochasticity(self, original_text, prompt_postfix, nr_variations=5, temperature=1.0, iterations=1):
        paraphrases = self.__generate_paraphrases(original_text, prompt_postfix, nr_variations=nr_variations, temperature=temperature)
        annotated = []
        for i, (paraphrase, original) in enumerate(zip(paraphrases['phrase'], paraphrases['original'])):
            print(f"Iteration {i}/{nr_variations}...", end='\r')
            sys.stdout.flush()
            for j, d in enumerate(self.data):
                annotation = self.llm.annotate(d, paraphrase, parse_function=self.parse_function)
                annotated.append({'id': j, 'text': d, 'annotation': annotation, 'prompt_id': i, 'prompt': paraphrase, 'original': original})
        print()
        print('Finished classifications.')
        annotated_data = pd.DataFrame(annotated)
        self.interprompt_df = annotated_data
        KA = simpledorff.calculate_krippendorffs_alpha_for_df(annotated_data, metric_fn=self.metric_fn, experiment_col='id', annotator_col='prompt_id', class_col='annotation')
        if KA < 0.8:
            original_prompt = original_text + ' ' + prompt_postfix
            print(f'Original prompt:\n{original_prompt}')
            print('Prompts with poor performance:')
            print(annotated_data[annotated_data['KA'] < 0.8])
        return KA, annotated_data



