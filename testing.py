# Step 1: Initialize Classes
import os
import pandas as pd
from utils import LLMWrapper
from utils import PromptStabilityAnalysis
from transformers import AutoModelForCausalLM, AutoTokenizer
import simpledorff

import seaborn as sns
import matplotlib.pyplot as plt

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

# Run baseline_stochasticity
KA, df, ka_scores, iterrations_no = psa.baseline_stochasticity(original_text, prompt_postfix, iterations=20)


# Function to plot KA scores with integer x-axis labels
def plot_ka_scores(ka_scores, overall_ka):
    iterations = list(range(2, 2 + len(ka_scores)))

    plt.figure(figsize=(10, 5))
    plt.plot(iterations, ka_scores, marker='o', linestyle='-', color='b', label='KA Score per Iteration')
    plt.axhline(y=overall_ka, color='r', linestyle='--', label=f'Overall KA: {overall_ka:.2f}')
    plt.xlabel('Iteration')
    plt.ylabel('Krippendorff\'s Alpha (KA)')
    plt.title('Krippendorff\'s Alpha Scores Across Iterations')
    plt.xticks(iterations)  # Set x-axis ticks to be whole integers
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot the KA scores
plot_ka_scores(ka_scores, KA)

## Note:

# KA = simpledorff.calculate_krippendorffs_alpha_for_df(
#     df,
#     metric_fn=self.metric_fn,
#     experiment_col='id',
#     annotator_col='iteration',
#     class_col='annotation')

# This is treating each iteration as a new annotator. It is then getting the KA for each run, calculated as, the overall agreement between iteration
# 1 and iteration 0 then iteration 2 and iteration 0. 