library(dplyr)
library(reticulate)
library(readr)
library(stringr)
library(tibble)
library(ggplot2)
library(cowplot)

# Define output directory (adjust as needed)
output_dir <- "/Users/christopherbarrie/Dropbox/nyu_projects/promptstability/data/example"
use_python("pssenv/bin/python", required = TRUE)

# Define file paths for the input CSVs
inter_file <- "data/example/ollama_inter.csv"

# Read in the CSV files
df_inter <- read_csv(inter_file)

# Define a function to extract the integer following the final occurrence of "think>"
extract_integer_after_think <- function(annotation_text) {
  extracted <- sub(".*think>\\s*(\\d+).*", "\\1", annotation_text)
  return(as.numeric(extracted))
}

df_inter <- df_inter %>%
  mutate(annotation_cleaned = extract_integer_after_think(annotation)) %>%
  select(-ka_mean, -ka_lower, -ka_upper)

# Write the updated dataframes to new CSV files
output_inter <- file.path(output_dir, "ollama_inter_cleaned.csv")

write_csv(df_inter, output_inter)

# Define file_paths tibble with annotator column information
file_paths <- tribble(
  ~cleaned_file,                                                                                          ~type,    ~dataset,           ~annotator_col,
  "/Users/christopherbarrie/Dropbox/nyu_projects/promptstability/data/example/ollama_inter_cleaned.csv",     "inter",  "manifestos",       "prompt_id"
)

# Step 2: Build the Python script as a string.
python_script <- "
import pandas as pd
from simpledorff import calculate_krippendorffs_alpha_for_df, metrics

def calculate_ka(df, dataset, annotator_col='prompt_id', class_col='annotation_cleaned'):
    df = df.copy()  # Force a copy to avoid SettingWithCopyWarning
    # Convert columns to numeric
    df[annotator_col] = pd.to_numeric(df[annotator_col], errors='coerce')
    df[class_col] = pd.to_numeric(df[class_col], errors='coerce')
    df = df.dropna(subset=[annotator_col, class_col])
    
    # If 'temperature' column is missing, create it with a default value (e.g., 0)
    if 'temperature' not in df.columns:
        df.loc[:, 'temperature'] = 0

    grouped = df.groupby('temperature')
    results = []
    
    # Use interval metric for manifestos_multi; nominal for others
    metric_fn = metrics.interval_metric if dataset == 'manifestos_multi' else metrics.nominal_metric

    for temp, group in grouped:
        try:
            alpha = calculate_krippendorffs_alpha_for_df(
                group,
                experiment_col='id',
                annotator_col=annotator_col,
                class_col=class_col,
                metric_fn=metric_fn
            )
            results.append({'temperature': temp, 'ka_mean': alpha})
        except Exception as e:
            print(f'Error calculating KA for temperature {temp}: {e}')
    return pd.DataFrame(results)

datasets = [
"

# Append datasets with type info from file_paths
file_paths <- file_paths %>%
  mutate(
    dataset_entry = paste0(
      "{'file': '", cleaned_file, "', 'type': '", type, "', 'dataset': '", dataset, "', 'annotator_col': '", annotator_col, "'}"
    )
  )

python_script <- paste0(
  python_script,
  paste(file_paths$dataset_entry, collapse = ",\n"),
  "\n]\n\n",
  "results = []\n",
  "for ds in datasets:\n",
  "    data = pd.read_csv(ds['file'])\n",
  "    ka_results = calculate_ka(data, dataset=ds['dataset'], annotator_col=ds['annotator_col'])\n",
  "    ka_results['dataset'] = ds['dataset']\n",
  "    ka_results['type'] = ds['type']\n",
  "    results.append(ka_results)\n\n",
  "output_file = '", output_dir, "/ka_results_combined.csv'\n",
  "pd.concat(results).to_csv(output_file, index=False)\n"
)

# Optionally print the Python script to verify its contents
cat(python_script)

# Execute the Python script using reticulate
reticulate::py_run_string(python_script)

df1 <- read_csv("data/example/ka_results_combined.csv") %>%
  distinct(temperature, ka_mean) %>%
  mutate(model = "deepseek-r1-8b")

openai_file <- "data/example/openai_inter.csv"
df2 <- read_csv(openai_file) %>%
  distinct(temperature, ka_mean) %>%
  mutate(model = "gpt-4o")

df <- bind_rows(df1, df2)

# Load the intra plot from the first script
p_intra <- readRDS(file.path(output_dir, "intra_plot.rds"))

# Create (or assign) the inter plot to p_inter (if not already done)
p_inter <- ggplot(df, aes(x = temperature, y = ka_mean, color = model)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  labs(
    x = "Temperature",
    y = "Inter-PSS",
    color = "Model"
  ) +
  scale_color_manual(values = c("deepseek-r1-8b" = "#0072B2", "gpt-4o" = "#D55E00")) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  theme_minimal(base_size = 14) +
  ylim(0, 1) +
  theme(
    legend.position = "top",
    panel.grid.major = element_line(color = "grey80"),
    panel.grid.minor = element_blank()
  )

# Combine the two plots side-by-side with labels A and B
combined_plot <- plot_grid(p_intra, p_inter, labels = c("A", "B"), ncol = 2)

# Display the combined plot
print(combined_plot)

# Save the combined plot
ggsave(
  filename = file.path("plots/combined_model_comparison_plot.png"),
  plot = combined_plot,
  width = 12, height = 4, dpi = 300
)



