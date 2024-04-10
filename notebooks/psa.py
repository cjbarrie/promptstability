import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import pandas as pd
import openai
import numpy as np
import time
import simpledorff
from openai import OpenAI
import matplotlib.pyplot as plt


import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import os

import simpledorff

from transformers import AutoModelForCausalLM, AutoTokenizer

import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


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

    # Compares similarity between two sentences in sentence embedding space
    def __compare_similarity(self,sent1,sent2):
        emb1 = self.embedding_model.encode(sent1, convert_to_tensor=True)
        emb2 = self.embedding_model.encode(sent2, convert_to_tensor=True)

        return util.pytorch_cos_sim(emb1, emb2)

    # Uses Pegasus to paraphrase a sentence
    def __paraphrase_sentence(self, input_text, num_return_sequences=10, num_beams=50, temperature=1.5):
        batch = self.tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(self.torch_device)
        translated = self.model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=temperature)
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    #This generates paraphrases based on an original text and uses sentence embedding to measure how different they are from the original sentence.
    #prompt_postfix is a fixed addition that is not paraphrased
    def __generate_paraphrases(self,original_text,prompt_postfix,nr_variations,temperature=1.5):
        # Create paraphrases of sentence
        phrases = self.__paraphrase_sentence(original_text,num_return_sequences=nr_variations,temperature=temperature)

        # Measure distances between new and original
        l = [{'similarity':1.0,'phrase':f'{original_text} {prompt_postfix}','original':True}]
        for phrase in phrases:
            sim = self.__compare_similarity(original_text,phrase)
            l.append({'similarity':float(sim),'phrase':f'{phrase} {prompt_postfix}','original':False})

        # Store for future use
        self.paraphrases = pd.DataFrame(l).sort_values(['similarity'])
        display(self.paraphrases)
        return self.paraphrases

    def baseline_stochasticity(self,original_text,prompt_postfix,iterations=10, plot_type='plotly'):
        '''
        This measures the amount of stochasticity there is within the same prompt, running the original prompt <iterations> and
        measuring the KA reliability over runs.
        '''
        prompt = f'{original_text} {prompt_postfix}'
        annotated = []
        ka_scores = []
        iterrations_no = []

        # Run the LLM on the data
        for i in range(iterations):
            print(f"Iteration {i}...")
            for j,d in enumerate(self.data):
                annotation = self.llm.annotate(d, prompt,parse_function=self.parse_function)
                annotated.append({'id':j,'text':d,'annotation':annotation,'iteration':i})

            # Measure the intercoder reliability for each additional repetition (after second iteration)
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

        print('Finished annotation. Analyzing reliability')

        # plot
        if plot_type == 'sns':
            sns.set()
            # turn iteration_no into integers
            #iterration_no = [int(x) for x in iterration_no]
            sns.scatterplot(x=iterrations_no, y= ka_scores)
            plt.xlabel('Number of Prompt Repetitions')
            plt.ylabel('KA Score')
            plt.title('Reliability (KA) vs. Repetitions')
            plt.ylim(0.0, 1.05)
            plt.xticks(range(0, max(iterrations_no) + 1, 5))
            plt.axhline(y=0.80, color='black', linestyle='--', linewidth=.5)
            plt.show()
        elif plot_type == 'plotly':
            data = {'Repetitions': iterrations_no, 'KA Score': ka_scores}
            df = pd.DataFrame(data)
            # interactive plot
            fig = px.scatter(df, x='Repetitions', y='KA Score', hover_data={'Repetitions': True, 'KA Score': True}, labels={'Repetitions': 'Repetitions', 'KA Score': 'KA Score'})
            # horizontal line: min KA acceptable
            fig.add_trace(go.Scatter(x=[0, max(iterrations_no)], y=[0.80, 0.80], mode='lines', name='KA Threshold', line=dict(color='black', width=.5, dash='dash')))

            fig.update_layout(
                title='Reliability (KA) vs. Repetitions',
                xaxis_title='Number of Prompt Repetitions',
                yaxis_title='KA Score',
                yaxis=dict(range=[0.0, 1.05]),
                hovermode='closest'
            )
            fig.show()

            print(f'Within-prompt KA reliability score for {i + 1} repetitions is {KA}')

        return KA, df, ka_scores, iterrations_no


    def interprompt_stochasticity(self,original_text,prompt_postfix, nr_variations=5, temperature=1.0, iterations=1, plot_type='plotly'):
        '''
        This measures the amount of stochasticity while varying the prompt.
        prompt_postfix: A fixed addition to the prompt. This is not paraphrased. Used to specify output format.
        '''

        # Generate paraphrases
        paraphrases = self.__generate_paraphrases(original_text,prompt_postfix,nr_variations=nr_variations,temperature=temperature)

        annotated = []
        # Run the LLM on the data
        for i, (paraphrase,similarity,original) in enumerate(zip(paraphrases['phrase'],paraphrases['similarity'],paraphrases['original'])):
            print(f"Iteration {i}...")
            for j,d in enumerate(self.data):
                annotation = self.llm.annotate(d, paraphrase,parse_function=self.parse_function)
                annotated.append({'id':j,'text':d,'annotation':annotation,'prompt_id':i,'prompt':paraphrase,'similarity':similarity,'original':original})

        print('Finished annotation. Analyzing reliability')
        annotated_data = pd.DataFrame(annotated)

        self.interprompt_df = annotated_data

        # Measure the interprompt reliability
        KA = simpledorff.calculate_krippendorffs_alpha_for_df(annotated_data,metric_fn=self.metric_fn,experiment_col='id', annotator_col='prompt_id', class_col='annotation')

        rel_vs_sim = self.__calculate_reliability_as_function_of_similarity(annotated_data)
        poor_prompts = rel_vs_sim.loc[rel_vs_sim['KA'] < 0.8]
        original_prompt = original_text + ' ' + prompt_postfix

        if plot_type == 'sns':
            sns.set()
            sns.scatterplot(x=rel_vs_sim['similarity'], y= rel_vs_sim['KA'])
            plt.xlabel('Prompt Similarity')
            plt.ylabel('KA Score')
            plt.title('Reliability vs similarity')
            plt.ylim(0.0, 1.05)
            plt.xlim(0,1)
            plt.axhline(y=0.80, color='black', linestyle='--', linewidth=.5)
            plt.show()
        elif plot_type == 'plotly':
            data = {'Similarity': rel_vs_sim['similarity'], 'KA Score': rel_vs_sim['KA'], 'Prompt_id': rel_vs_sim['prompt_id']}
            df = pd.DataFrame(data)
            # interactive plot
            fig = px.scatter(df, x='Similarity', y='KA Score', hover_data={'Similarity': True, 'KA Score': True, 'Prompt_id': True}, labels={'Similarity': 'Prompt Similarity', 'KA Score': 'KA Score', 'Prompt': 'Prompt_id'})
            # horizontal line: min KA acceptable
            fig.add_trace(go.Scatter(x=[0, 1], y=[0.80, 0.80], mode='lines', name='KA Threshold', line=dict(color='black', width=.5, dash='dash')))
            fig.update_traces(hoverlabel=dict(font=dict(size=12), align='left', namelength=30))

            fig.update_layout(
                title='Reliability (KA) vs. Similarity',
                xaxis_title='Prompt Similarity',
                yaxis_title='KA Score',
                yaxis=dict(range=[0.0, 1.05]),
                xaxis=dict(range=[0, 1]),  # Set x-axis limits,
                hovermode='closest'
            )
            fig.show()

        print(f'Inter-prompt KA reliability score is {KA}.')
        print('\n#################################')
        print(f'Original prompt:\n{original_prompt}')
        print('#################################')
        print('Prompts with poor performance:')
        display(poor_prompts)
        return KA, annotated_data, rel_vs_sim, poor_prompts


    #This calculates the KA-R as a function of the similarity between the prompts
    # Takes the output of the interprompt_stochasticity calculation
    def __calculate_reliability_as_function_of_similarity(self,df=None):
        if df is None:
            df = self.interprompt_df

        l = []
        # This calculates the KA separately one-on-one between the prompts, and then uses the similarity between the prompts to say something.
        for prompt_id in df['prompt_id'].loc[df['original']==False].unique():
            # Go through them one at the time
            dff = df.loc[(df['prompt_id']==prompt_id) | (df['original']==True)]
            ka = simpledorff.calculate_krippendorffs_alpha_for_df(dff,metric_fn=self.metric_fn,experiment_col='id', annotator_col='prompt_id', class_col='annotation')

            # Get the similarity of the prompt. Ugly code.
            similarity = df.loc[df['prompt_id']==prompt_id]['similarity'].values[0]
            prompt_text = df.loc[df['prompt_id']==prompt_id]['prompt'].values[0]
            l.append({'prompt_id':prompt_id, 'prompt_text': prompt_text,'similarity':similarity, 'KA':ka})

        return pd.DataFrame(l)
