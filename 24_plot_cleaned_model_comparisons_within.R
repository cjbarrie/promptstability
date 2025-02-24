library(dplyr)
library(reticulate)
library(readr)
library(tibble)
library(ggplot2)

# Set up your Python environment
use_python("pssenv/bin/python", required = TRUE)

# Define output directory (adjust as needed)
output_dir <- "/Users/christopherbarrie/Dropbox/nyu_projects/promptstability/data/example"

# (Assuming you already created the intra cleaned CSV earlier.)
# For example, your intra file:
intra_file <- file.path(output_dir, "ollama_intra.csv")
# (If needed, you can run similar R code as for inter to create this file.)

# Read in the CSV files
df_intra <- read_csv(intra_file)

# Define a function to extract the integer following the final occurrence of "think>"
extract_integer_after_think <- function(annotation_text) {
  extracted <- sub(".*think>\\s*(\\d+).*", "\\1", annotation_text)
  return(as.numeric(extracted))
}

df_intra <- df_intra %>%
  mutate(annotation_cleaned = extract_integer_after_think(annotation)) %>%
  select(-ka_mean, -ka_lower, -ka_upper)

# Write the updated dataframes to new CSV files
intra_file <- file.path(output_dir, "ollama_intra_cleaned.csv")

write_csv(df_intra, intra_file)

# Build the Python script as a single pasted string.
python_script_intra <- paste0(
  "import pandas as pd\n",
  "from simpledorff import calculate_krippendorffs_alpha_for_df, metrics\n",
  "\n",
  "def calculate_intra_ka(file, dataset, annotator_col='iteration', class_col='annotation_cleaned'):\n",
  "    df = pd.read_csv(file).copy()\n",
  "    df[annotator_col] = pd.to_numeric(df[annotator_col], errors='coerce')\n",
  "    df[class_col] = pd.to_numeric(df[class_col], errors='coerce')\n",
  "    df = df.dropna(subset=[annotator_col, class_col])\n",
  "    iterations = int(df[annotator_col].max()) + 1\n",
  "    results = {}\n",
  "    for i in range(1, iterations):\n",
  "        subset = df[df[annotator_col] <= i]\n",
  "        try:\n",
  "            alpha = calculate_krippendorffs_alpha_for_df(\n",
  "                subset,\n",
  "                experiment_col='id',\n",
  "                annotator_col=annotator_col,\n",
  "                class_col=class_col,\n",
  "                metric_fn=metrics.nominal_metric\n",
  "            )\n",
  "        except Exception as e:\n",
  "            print(f'Error calculating KA for iteration {i}: {e}')\n",
  "            alpha = None\n",
  "        results[i] = alpha\n",
  "        df.loc[df[annotator_col] == i, 'ka_mean'] = alpha\n",
  "    return df, results\n",
  "\n",
  "intra_file = '", intra_file, "'\n",
  "df_intra, ka_results = calculate_intra_ka(intra_file, dataset='manifestos')\n",
  "output_intra_file = '", file.path(output_dir, "ka_results_intra.csv"), "'\n",
  "df_intra.to_csv(output_intra_file, index=False)\n",
  "print(f'Intra KA results written to {output_intra_file}')\n"
)

# Optionally, print the Python script to verify its contents
cat(python_script_intra)

# Execute the Python script via reticulate
reticulate::py_run_string(python_script_intra)

# ----- Read and plot the intra KA results in R -----
df_intra_ka <- read_csv(file.path(output_dir, "ka_results_intra.csv"))

# Optionally, add a model column (if you want to overlay with other models)
df_intra_ka <- df_intra_ka %>% mutate(model = "deepseek-r1-8b")

df1 <- read_csv("data/example/ka_results_intra.csv") %>%
  distinct(iteration, ka_mean) %>%
  mutate(model = "deepseek-r1-8b")

openai_file <- "data/example/openai_intra.csv"
df2 <- read_csv(openai_file) %>%
  distinct(iteration, ka_mean) %>%
  mutate(model = "gpt-4o")

df <- bind_rows(df1, df2)

# Create the intra plot and assign it to p_intra
p_intra <- ggplot(df, aes(x = iteration, y = ka_mean, color = model)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  labs(
    x = "Iteration",
    y = "Intra-PSS",
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

saveRDS(p_intra, file = file.path(output_dir, "intra_plot.rds"))
