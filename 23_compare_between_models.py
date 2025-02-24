import pandas as pd
import matplotlib.pyplot as plt
from utils import PromptStabilityAnalysis, get_openai_api_key
from openai import OpenAI
import ollama
import os
import time

#############################################
# 1. Load a subsubsample of the manifestos dataset
#############################################
df = pd.read_csv('data/manifestos.csv')
df = df[df['scale'] == 'Economic']
# Take 10% of the rows
sample_size = max(1, int(0.1 * len(df)))
df = df.sample(sample_size, random_state=123)
data = list(df['sentence_context'].values)

# Define the prompt texts
original_text = (
    "The text provided is a UK party manifesto. "
    "Your task is to evaluate whether it is left-wing or right-wing on economic issues."
)
prompt_postfix = "Respond with 0 for left-wing or 1 for right-wing. Only respond with a one token integer. Do not respond with anything else."

#############################################
# 2. ANALYSIS USING OPENAI
#############################################
# Define the OpenAI annotation function
APIKEY = get_openai_api_key()
OPENAI_MODEL = 'gpt-4o'
client = OpenAI(api_key=APIKEY)

def annotate_openai(text, prompt, temperature=0.1):
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=temperature,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
        )
    except Exception as e:
        print(f"OpenAI exception: {e}")
        raise e
    return ''.join(choice.message.content for choice in response.choices)

# Instantiate the analysis class using OpenAI’s annotation function
psa_openai = PromptStabilityAnalysis(annotation_function=annotate_openai, data=data)

# Run intra-prompt (baseline) analysis using the updated method name `intra_pss`
print("Running OpenAI intra-prompt (baseline) analysis...")
ka_openai_intra, annotated_openai_intra = psa_openai.intra_pss(
    original_text, 
    prompt_postfix, 
    iterations=20,
    plot=False
)
print("OpenAI intra-PSS:", ka_openai_intra)

# Run inter-prompt analysis using the updated method name `inter_pss`
temperatures = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
nr_variations = 3
print("Running OpenAI inter-prompt analysis...")
ka_openai_inter, annotated_openai_inter = psa_openai.inter_pss(
    original_text, 
    prompt_postfix, 
    nr_variations=3,  
    temperatures=temperatures,
    iterations=1,
    plot=False
)
print("OpenAI inter-PSS:", ka_openai_inter)

# Create the new subdirectory data/example if it doesn't exist
output_dir = os.path.join('data', 'example')
os.makedirs(output_dir, exist_ok=True)

# Save OpenAI intra-prompt annotations
openai_intra_csv = os.path.join(output_dir, 'openai_intra.csv')
annotated_openai_intra.to_csv(openai_intra_csv, index=False)
print(f"OpenAI intra-prompt annotations saved to {openai_intra_csv}")

# Save OpenAI inter-prompt annotations
openai_inter_csv = os.path.join(output_dir, 'openai_inter.csv')
annotated_openai_inter.to_csv(openai_inter_csv, index=False)
print(f"OpenAI inter-prompt annotations saved to {openai_inter_csv}")

#############################################
# 3. ANALYSIS USING OLLAMA (with local deepseek-r1:8b)
#############################################
# Define the Ollama annotation function.

OLLAMA_MODEL = 'deepseek-r1:8b'
def annotate_ollama(text, prompt, temperature=0.1):
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ])
    except Exception as e:
        print(f"Ollama exception: {e}")
        raise e
    return response['message']['content']

# Instantiate the analysis class using Ollama’s annotation function
psa_ollama = PromptStabilityAnalysis(annotation_function=annotate_ollama, data=data)

# Run intra-prompt (baseline) analysis for Ollama with few iterations
print("Running Ollama intra-prompt (baseline) analysis...")
ka_ollama_intra, annotated_ollama_intra = psa_ollama.intra_pss(
    original_text, 
    prompt_postfix, 
    iterations=20,
    plot=False
)
print("Ollama intra-PSS:", ka_ollama_intra)

# Output the annotations to the terminal
print("\nOllama Intra-Prompt Annotations:")
print(annotated_ollama_intra[['id', 'text', 'annotation']].head(20))  # adjust the number of rows to print as needed

# Run inter-prompt analysis for Ollama with a couple of temperatures
print("Running Ollama inter-prompt analysis...")
ka_ollama_inter, annotated_ollama_inter = psa_ollama.inter_pss(
    original_text, 
    prompt_postfix, 
    nr_variations=3,
    temperatures=temperatures,
    iterations=1,
    plot=False
)
print("Ollama inter-PSS:", ka_ollama_inter)

# Save Ollama intra-prompt annotations
ollama_intra_csv = os.path.join(output_dir, 'ollama_intra.csv')
annotated_ollama_intra.to_csv(ollama_intra_csv, index=False)
print(f"Ollama intra-prompt annotations saved to {ollama_intra_csv}")

# Save Ollama inter-prompt annotations
ollama_inter_csv = os.path.join(output_dir, 'ollama_inter.csv')
annotated_ollama_inter.to_csv(ollama_inter_csv, index=False)
print(f"Ollama inter-prompt annotations saved to {ollama_inter_csv}")