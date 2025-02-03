library(dplyr)
library(ggplot2)
library(reticulate)
library(tidyr)

# Set the Python executable for the virtual environment
use_python("pssenv/bin/python", required = TRUE)

# --- Step 1: Define the file paths ---
file_paths <- tibble::tribble(
  ~file,                                    ~dataset,       ~type,
  "data/annotated/reannotated/comparison/cleaned_manifestos_filtered.csv",     "manifestos",      "Filtered",
  "data/annotated/reannotated/comparison/cleaned_manifestos_multi_filtered.csv", "manifestos_multi", "Filtered",
  "data/annotated/reannotated/comparison/cleaned_mii_filtered.csv",            "mii",             "Filtered",
  "data/annotated/reannotated/comparison/cleaned_mii_long_filtered.csv",       "mii_long",        "Filtered",
  "data/annotated/reannotated/comparison/cleaned_news_filtered.csv",           "news",            "Filtered",
  "data/annotated/reannotated/comparison/cleaned_news_short_filtered.csv",     "news_short",      "Filtered",
  "data/annotated/reannotated/comparison/cleaned_stance_filtered.csv",         "stance",          "Filtered",
  "data/annotated/reannotated/comparison/cleaned_stance_long_filtered.csv",    "stance_long",     "Filtered",
  "data/annotated/reannotated/comparison/cleaned_synth_filtered.csv",          "synth",           "Filtered",
  "data/annotated/reannotated/comparison/cleaned_synth_short_filtered.csv",    "synth_short",     "Filtered",
  "data/annotated/reannotated/comparison/cleaned_tweets_pop_filtered.csv",     "tweets_pop",      "Filtered",
  "data/annotated/reannotated/comparison/cleaned_tweets_rd_filtered.csv",      "tweets_rd",       "Filtered"
)

# --- Step 2: Define the output directory and file ---
output_dir <- "data/annotated/reannotated/comparison/"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
output_file <- file.path(output_dir, "ka_results_combined_subsamples.csv")

# --- Step 3: Build the Python script with injected file paths ---
# Create a string representing the Python list of dataset dictionaries
dataset_entries <- file_paths %>% 
  mutate(entry = paste0("{'file': '", file, "', 'type': '", 
                        type, "', 'dataset': '", dataset, "'}")) %>%
  pull(entry) %>%
  paste(collapse = ",\n    ")

dataset_list_str <- paste0("datasets = [\n    ", dataset_entries, "\n]")

python_script <- paste0(
  "import pandas as pd
from simpledorff import calculate_krippendorffs_alpha_for_df, metrics

def calculate_ka(df, dataset, annotator_col='prompt_id', class_col='annotation'):
    # Convert columns to numeric and drop missing values
    df[annotator_col] = pd.to_numeric(df[annotator_col], errors='coerce')
    df[class_col] = pd.to_numeric(df[class_col], errors='coerce')
    df = df.dropna(subset=[annotator_col, class_col])
    # Group by 'temperature'
    grouped = df.groupby('temperature')
    results = []
    # Choose the metric: interval for manifestos_multi, nominal for others
    metric_fn = metrics.interval_metric if dataset == 'manifestos_multi' else metrics.nominal_metric
    # Define subsample fractions
    subsample_fracs = [0.02, 0.05, 0.1, 0.25, 0.5, 0.75]
    for temp, group in grouped:
        for frac in subsample_fracs:
            try:
                if len(group) > 0:
                    sample = group.sample(frac=frac, random_state=42)
                    alpha = calculate_krippendorffs_alpha_for_df(
                        sample,
                        experiment_col='id',
                        annotator_col=annotator_col,
                        class_col=class_col,
                        metric_fn=metric_fn
                    )
                    results.append({'temperature': temp, 'ka_mean': alpha, 'frac': frac})
            except Exception as e:
                print(f'Error calculating KA for temperature {temp} with frac {frac}: {e}')
    return pd.DataFrame(results)

",
  # Inject the dataset list constructed from R:
  dataset_list_str, "\n\n",
  "results = []
for ds in datasets:
    data = pd.read_csv(ds['file'])
    ka_results = calculate_ka(data, dataset=ds['dataset'])
    ka_results['dataset'] = ds['dataset']
    ka_results['type'] = ds['type']
    results.append(ka_results)

output_file = '", output_file, "'
pd.concat(results).to_csv(output_file, index=False)
"
)

# --- Step 4: Run the Python script ---
reticulate::py_run_string(python_script)

# --- Step 5: Read back the results ---
ka_results <- read.csv(file.path(output_dir, "ka_results_combined_subsamples.csv"))

# Ensure 'frac' and 'temperature' are factors for plotting
ka_results$frac <- as.factor(ka_results$frac)
ka_results$temperature <- as.factor(ka_results$temperature)

# --- Step 6: Order facets and rename them ---
# Calculate the mean KA per dataset for ordering (highest mean first)
facet_order <- ka_results %>%
  group_by(dataset) %>%
  summarise(mean_ka = mean(ka_mean, na.rm = TRUE)) %>%
  arrange(desc(mean_ka)) %>%
  pull(dataset)

# Define a named vector with new facet labels
facet_labels <- c(
  "tweets_rd" = "Tweets (Rep. Dem.)",
  "tweets_pop" = "Tweets (Populism)",
  "news" = "News",
  "news_short" = "News (Short)",
  "manifestos" = "Manifestos",
  "manifestos_multi" = "Manifestos Multi",
  "stance" = "Stance",
  "stance_long" = "Stance (Long)",
  "mii" = "MII",
  "mii_long" = "MII (Long)",
  "synth" = "Synthetic",
  "synth_short" = "Synthetic (Short)"
)

# Update the dataset column to be a factor with the desired order
ka_results <- ka_results %>%
  mutate(dataset = factor(dataset, levels = facet_order))

# --- Step 7: Create the final styled plot ---
final_plot <- ggplot(ka_results, aes(x = temperature, y = ka_mean, color = frac, group =frac)) +
  geom_line(alpha = 0.8) +
  geom_point(alpha = 0.1) +
  labs(
    title = "",
    x = "Temperature",
    y = "inter-PSS",
    color = "Dataset"
  ) +
  ylim(0, 1) +
  facet_wrap(
    ~ dataset, 
    scales = "free", 
    ncol = 4, 
    labeller = as_labeller(facet_labels)
  ) + 
  theme_minimal() +
  theme(
    legend.position = "bottom",
    strip.text = element_text(size = 10)
  )

# Display the plot
print(final_plot)

# Save the plot to file
# ggsave("plots/combined_postpro.png", plot = final_plot, width = 16, height = 12, dpi = 300)
