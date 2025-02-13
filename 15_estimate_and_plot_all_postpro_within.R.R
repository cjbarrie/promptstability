library(dplyr)
library(ggplot2)
library(reticulate)
library(tidyr)

# Set the Python executable for the virtual environment
use_python("pssenv/bin/python", required = TRUE)

# Define the output directory (same as in filtering)
output_dir <- "data/annotated/reannotated/within"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Input: File paths tibble referencing the filtered CSVs from the filtering script
file_paths <- tibble::tribble(
  ~file, ~dataset, ~type,
  "data/annotated/reannotated/within/manifestos_multi_filtered.csv", "manifestos_multi", "Filtered",
  "data/annotated/reannotated/within/manifestos_filtered.csv", "manifestos", "Filtered",
  "data/annotated/reannotated/within/mii_long_filtered.csv", "mii_long", "Filtered",
  "data/annotated/reannotated/within/mii_filtered.csv", "mii", "Filtered",
  "data/annotated/reannotated/within/news_short_filtered.csv", "news_short", "Filtered",
  "data/annotated/reannotated/within/news_filtered.csv", "news", "Filtered",
  "data/annotated/reannotated/within/stance_long_filtered.csv", "stance_long", "Filtered",
  "data/annotated/reannotated/within/stance_filtered.csv", "stance", "Filtered",
  "data/annotated/reannotated/within/synth_short_filtered.csv", "synth_short", "Filtered",
  "data/annotated/reannotated/within/synth_filtered.csv", "synth", "Filtered",
  "data/annotated/reannotated/within/tweets_pop_filtered.csv", "tweets_pop", "Filtered",
  "data/annotated/reannotated/within/tweets_rd_filtered.csv", "tweets_rd", "Filtered",
  "data/annotated/reannotated/within/manifestos_multi_filtered_balanced.csv", "manifestos_multi", "Filtered & Balanced",
  "data/annotated/reannotated/within/manifestos_filtered_balanced.csv", "manifestos", "Filtered & Balanced",
  "data/annotated/reannotated/within/mii_long_filtered_balanced.csv", "mii_long", "Filtered & Balanced",
  "data/annotated/reannotated/within/mii_filtered_balanced.csv", "mii", "Filtered & Balanced",
  "data/annotated/reannotated/within/news_short_filtered_balanced.csv", "news_short", "Filtered & Balanced",
  "data/annotated/reannotated/within/news_filtered_balanced.csv", "news", "Filtered & Balanced",
  "data/annotated/reannotated/within/stance_long_filtered_balanced.csv", "stance_long", "Filtered & Balanced",
  "data/annotated/reannotated/within/stance_filtered_balanced.csv", "stance", "Filtered & Balanced",
  "data/annotated/reannotated/within/synth_short_filtered_balanced.csv", "synth_short", "Filtered & Balanced",
  "data/annotated/reannotated/within/synth_filtered_balanced.csv", "synth", "Filtered & Balanced",
  "data/annotated/reannotated/within/tweets_pop_filtered_balanced.csv", "tweets_pop", "Filtered & Balanced",
  "data/annotated/reannotated/within/tweets_rd_filtered_balanced.csv", "tweets_rd", "Filtered & Balanced"
)


# Step 2: Write Python script (the script loads the CSVs, calculates intra-KA, and writes combined results)
python_script <- "
import pandas as pd
from simpledorff import calculate_krippendorffs_alpha_for_df, metrics

def calculate_intra_ka(df, dataset, annotator_col='iteration', class_col='annotation'):
    # Convert columns to numeric and drop rows with missing values in these columns
    df[annotator_col] = pd.to_numeric(df[annotator_col], errors='coerce')
    df[class_col] = pd.to_numeric(df[class_col], errors='coerce')
    df = df.dropna(subset=[annotator_col, class_col])
    
    # Select the appropriate metric function
    metric_fn = metrics.interval_metric if dataset == 'manifestos_multi' else metrics.nominal_metric
    
    results = []
    # Determine the maximum iteration present in the data
    max_iter = int(df[annotator_col].max())
    # For each iteration from 1 to max_iter, calculate KA on all rows with iteration <= current iteration
    for i in range(1, max_iter+1):
        subset = df[df[annotator_col] <= i]
        try:
            alpha = calculate_krippendorffs_alpha_for_df(
                subset,
                experiment_col='id',
                annotator_col=annotator_col,
                class_col=class_col,
                metric_fn=metric_fn
            )
            results.append({'iteration': i, 'ka_mean': alpha})
        except Exception as e:
            print(f'Error calculating KA for iteration {i}: {e}')
    return pd.DataFrame(results)

# Build the list of dataset entries from the R tibble
datasets = ["

# Append dataset entries (using the cleaned CSV files from filtering)
file_paths %>%
  mutate(
    dataset_entry = paste0(
      "{'file': '", file, "', 'type': '", type, "', 'dataset': '", dataset, "'}"
    )
  ) %>%
  pull(dataset_entry) %>%
  paste(collapse = ",\n") %>%
  {python_script <<- paste0(python_script, ., "\n]")}

python_script <- paste0(
  python_script,
  "

results = []
for ds in datasets:
    data = pd.read_csv(ds['file'])
    ka_results = calculate_intra_ka(data, dataset=ds['dataset'])
    ka_results['dataset'] = ds['dataset']
    ka_results['type'] = ds['type']
    results.append(ka_results)

output_file = '", output_dir, "/ka_results_combined.csv'
pd.concat(results).to_csv(output_file, index=False)
"
)

# Execute Python script
reticulate::py_run_string(python_script)

# Step 4: Read back results and plot
ka_results <- read.csv(file.path(output_dir, "ka_results_combined.csv"))


file_paths_orig <- tibble::tribble(
  ~file, ~dataset, ~type,
  "data/annotated/manifestos_within_expanded.csv", "manifestos", "Original",
  "data/annotated/manifestos_multi_within_expanded.csv", "manifestos_multi", "Original",
  "data/annotated/mii_within_expanded.csv", "mii", "Original",
  "data/annotated/mii_long_within_expanded.csv", "mii_long", "Original",
  "data/annotated/news_within_expanded.csv", "news", "Original",
  "data/annotated/news_short_within_expanded.csv", "news_short", "Original",
  "data/annotated/stance_within_expanded.csv", "stance", "Original",
  "data/annotated/stance_long_within_expanded.csv", "stance_long", "Original",
  "data/annotated/synth_within_expanded.csv", "synth", "Original",
  "data/annotated/synth_short_within_expanded.csv", "synth_short", "Original",
  "data/annotated/tweets_pop_within_expanded.csv", "tweets_pop", "Original",
  "data/annotated/tweets_rd_within_expanded.csv", "tweets_rd", "Original"
)

original_results <- file_paths_orig %>%
  rowwise() %>%
  mutate(original_data = list(read.csv(file) %>% select(iteration, ka_mean))) %>%
  unnest(original_data) %>%
  distinct(dataset, type, iteration, ka_mean)

combined_results <- bind_rows(
  original_results,
  ka_results
)

# Calculate ordering for facets
facet_order <- ka_results %>%
  group_by(dataset) %>%
  summarise(mean_pss = mean(ka_mean, na.rm = TRUE)) %>%
  arrange(desc(mean_pss)) %>%
  pull(dataset)

# Facet label mapping
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

combined_results <- combined_results %>%
  mutate(dataset = factor(dataset, levels = facet_order))

final_plot <- ggplot(combined_results, aes(x = iteration, y = ka_mean, color = type)) +
  geom_line(alpha = .8) +
  geom_point(alpha = .1) +
  labs(
    title = "",
    x = "iteration",
    y = "intra-PSS",
    color = "Dataset"
  ) +
  ylim(.75, 1) +
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

print(final_plot)

ggsave("plots/combined_within_postpro.png", plot = final_plot, width = 8, height = 6, dpi = 300)
