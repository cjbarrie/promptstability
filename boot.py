import pandas as pd
import numpy as np
import simpledorff

# Data
annotated_data = pd.read_csv('data/annotated/tweets_between.csv')
print(np.mean(annotated_data['KA']))
def bootstrap_krippendorff(df, iterations=1000, confidence_level=95):
    alpha_scores = []
    for _ in range(iterations):
        # Bootstrap sample: sample with replacement from the existing sample
        bootstrap_sample = df.sample(n=len(df), replace=True)
        # Calculate Krippendorff's Alpha for the sample
        alpha = simpledorff.calculate_krippendorffs_alpha_for_df(
            bootstrap_sample,
            metric_fn=simpledorff.metrics.nominal_metric,
            experiment_col='id',
            annotator_col='prompt_id',
            class_col='annotation'
        )
        alpha_scores.append(alpha)

    alpha_scores = np.array(alpha_scores)
    # Calculate confidence intervals
    lower_bound = np.percentile(alpha_scores, (100 - confidence_level) / 2)
    upper_bound = np.percentile(alpha_scores, 100 - (100 - confidence_level) / 2)

    return np.mean(alpha_scores), (lower_bound, upper_bound)

# Use the existing DataFrame `annotated_data` that has already been populated by your API calls
#mean_alpha, conf_interval = bootstrap_krippendorff(annotated_data)
#print(f"Average Krippendorff's Alpha: {mean_alpha}")
#print(f"95% Confidence Interval: [{conf_interval[0]:.2f}, {conf_interval[1]:.2f}]")



'''
between = pd.read_csv('data/annotated/news_between.csv')

unique_prompts = between.drop_duplicates(subset=['temperature', 'prompt_id'], keep='first')

print(unique_prompts)
unique_prompts.to_csv('/Users/ellipalaiologou/Downloads/unique_prompts.csv', index=False)

'''
