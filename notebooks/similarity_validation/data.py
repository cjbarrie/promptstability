import pandas as pd
import numpy as np

'''
# include task column to have unique task-prompt_id rows
data['task'] = 0
data.loc[:49, 'task'] = 1
data.loc[50:99, 'task'] = 2
data.loc[100:149, 'task'] = 3
data.to_csv('data.csv', index=False)
'''
data = pd.read_csv('data.csv')
print(data)

'''
# subset data.csv to exclude KA and similartiy columns
data_subset = data[['task', 'prompt_id', 'prompt_text', 'original_prompt']]
data_subset['similarity_coder'] = np.nan
data_subset.to_csv('data_subset.csv', index=False)
'''
data_subset = pd.read_csv('data_subset.csv')
print(data_subset)

# Download data
#data_subset.to_excel('/Users/ellipalaiologou/Downloads/data_subset.xlsx', index=False)
#data_subset.to_csv('/Users/ellipalaiologou/Downloads/data_subset.csv', index=False)
