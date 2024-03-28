import pandas as pd

data = pd.read_csv('data.csv')

# Iterate through each sentence
for index, row in data.iterrows():
    generated_prompt = row['prompt_text']

    # Display the sentence to the coder
    print(f"Prompt: {generated_prompt}")

    # Prompt the coder to input their rating
    similarity = input("Enter your similarity (from 0 to 10): ")

    # Store the rating along with the coder's name in a new column
    coder_name = input("Enter your name (all lowercase): ")  # Replace with the coder's name
    data.at[index, coder_name] = similarity

print(data)
# Save the updated DataFrame back to CSV
#data.to_csv('data_validated.csv', index=False)
