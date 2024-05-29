import pandas as pd
import numpy as np
import simpledorff

# Data
annotated_data = pd.read_csv('data/annotated/tweets_within.csv')

def bootstrap_krippendorff(df, bootstrap_samples=1000, confidence_level=95):
    alpha_scores = []

    for _ in range(bootstrap_samples):
        # Bootstrap sample: sample with replacement from the existing sample
        bootstrap_sample = df.sample(n=len(df), replace=True)
        # Calculate Krippendorff's Alpha for the sample
        alpha = simpledorff.calculate_krippendorffs_alpha_for_df(
            bootstrap_sample,
            metric_fn=simpledorff.metrics.nominal_metric,
            experiment_col='id',
            annotator_col='iteration',
            class_col='annotation'
        )
        alpha_scores.append(alpha)

    alpha_scores = np.array(alpha_scores)
    # Calculate confidence intervals
    ci_lower = np.percentile(alpha_scores, (100 - confidence_level) / 2)
    ci_upper = np.percentile(alpha_scores, 100 - (100 - confidence_level) / 2)

    return np.mean(alpha_scores), (ci_lower, ci_upper)


# Use the existing DataFrame `annotated_data` that has already been populated by your API calls
results = {}

for iteration in sorted(annotated_data['iteration'].unique()):
    if iteration > 0:
        iter_data = annotated_data[annotated_data['iteration'] <= iteration]
        mean_alpha, (ci_lower, ci_upper) = bootstrap_krippendorff(iter_data)
        results[iteration] = {
            'Average Alpha': mean_alpha,
            'CI Lower': ci_lower,
            'CI Upper': ci_upper
        }

print(results)


import matplotlib.pyplot as plt

# Use the existing DataFrame `annotated_data` that has already been populated by your API calls

# Extracting results for plotting
iterations_list = sorted(results.keys())
ka_values = [results[iter]['Average Alpha'] for iter in iterations_list]
average_ka = np.mean(ka_values)
ci_lowers = [results[iter]['Average Alpha'] - results[iter]['CI Lower'] for iter in iterations_list]
ci_uppers = [results[iter]['CI Upper'] - results[iter]['Average Alpha'] for iter in iterations_list]

# Plotting
plt.figure(figsize=(10, 5))
plt.errorbar(iterations_list, ka_values, yerr=[ci_lowers, ci_uppers], fmt='o', linestyle='-', color='b', ecolor='gray', capsize=3)
plt.xlabel('Iterations')
plt.ylabel("Krippendorff's Alpha (KA)")
plt.title("Krippendorff's Alpha Scores Across Iterations with 95% CI")
plt.xticks(iterations_list)  # Set x-axis ticks to be appropriate temperatures
plt.grid(True)
plt.axhline(y=0.80, color='black', linestyle='--', linewidth=.5)
plt.axhline(y=average_ka, color='r', linestyle='--', label=f'Average KA: {average_ka:.2f}')
plt.legend()
plt.show()


'''
ERROR DIVISION BY ZERO
        if len(iter_data['annotation'].unique()) > 1:  # Ensure there's more than one unique annotation
            mean_alpha, (ci_lower, ci_upper) = bootstrap_krippendorff(iter_data)
            results[iteration] = {'Average Alpha': mean_alpha, 'CI Lower': ci_lower, 'CI Upper': ci_upper}
        else:
            results[iteration] = {'Error': 'Insufficient diversity in annotations to calculate Alpha'}
'''

'''
between = pd.read_csv('data/annotated/news_between.csv')

unique_prompts = between.drop_duplicates(subset=['temperature', 'prompt_id'], keep='first')

print(unique_prompts)
unique_prompts.to_csv('/Users/ellipalaiologou/Downloads/unique_prompts.csv', index=False)

'''
