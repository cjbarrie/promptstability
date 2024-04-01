import pandas as pd

data = pd.read_csv('data.csv')
test = data.iloc[:2].copy()

# Iterate through each sentence
for index, row in test.iterrows():
    original_prompt = row['original_prompt']
    generated_prompt = row['prompt_text']

    # Display the sentence to the coder
    print(f"Original Prompt: {original_prompt}")
    print(f"Generated Prompt: {generated_prompt}")

    # Prompt the coder to input their rating
    similarity = input("Enter your similarity score between original and generated prompts (min 0, max 10): ")

    # Store the rating along with the coder's name in a new column
    coder_name = input("Enter your name (all lowercase): ")  # Replace with the coder's name
    test.loc[index, coder_name] = float(similarity)

print(test)
# Save the updated DataFrame back to CSV
#data.to_csv('data_validated.csv', index=False)
