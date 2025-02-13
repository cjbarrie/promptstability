library(dplyr)
library(knitr)
library(kableExtra)
library(tidyr)

# --- Step 1: Define the file paths ---
file_paths <- tibble::tribble(
  ~file,                                    ~dataset,       ~type,
  "data/annotated/reannotated/between/manifestos_multi_filtered.csv", "manifestos_multi", "Filtered",
  "data/annotated/reannotated/between/manifestos_filtered.csv", "manifestos", "Filtered",
  "data/annotated/reannotated/between/mii_long_filtered.csv", "mii_long", "Filtered",
  "data/annotated/reannotated/between/mii_filtered.csv", "mii", "Filtered",
  "data/annotated/reannotated/between/news_short_filtered.csv", "news_short", "Filtered",
  "data/annotated/reannotated/between/news_filtered.csv", "news", "Filtered",
  "data/annotated/reannotated/between/stance_long_filtered.csv", "stance_long", "Filtered",
  "data/annotated/reannotated/between/stance_filtered.csv", "stance", "Filtered",
  "data/annotated/reannotated/between/synth_short_filtered.csv", "synth_short", "Filtered",
  "data/annotated/reannotated/between/synth_filtered.csv", "synth", "Filtered",
  "data/annotated/reannotated/between/tweets_pop_filtered.csv", "tweets_pop", "Filtered",
  "data/annotated/reannotated/between/tweets_rd_filtered.csv", "tweets_rd", "Filtered"
)

# --- Output directory ---
output_dir <- "data/annotated/reannotated/comparison/"
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

# --- Function to load and process datasets ---
process_data <- function(file) {
  data <- read.csv(file)
  data %>%
    mutate(temperature = as.numeric(temperature)) %>%  # Ensure temperature is numeric
    filter(!is.na(temperature))                         # Remove rows with missing temperatures
}

# --- Combine all datasets into a single DataFrame ---
combined_data <- bind_rows(
  lapply(seq_len(nrow(file_paths)), function(i) {
    dataset <- file_paths$dataset[i]
    file <- file_paths$file[i]
    data <- process_data(file)
    data %>% mutate(Dataset = dataset)  # Add a column to identify the dataset
  })
)

# --- Define the explicit subsample fractions ---
fractions <- c(0.02, 0.05, 0.1, 0.25, 0.5, 0.75)

# --- Calculate total annotated rows per dataset (ignoring temperature) ---
total_rows_df <- combined_data %>%
  group_by(Dataset) %>%
  summarise(TotalRows = n(), .groups = "drop")

# --- For each dataset and each fraction, compute the number of rows that would be selected ---
dataset_summary <- total_rows_df %>%
  crossing(Fraction = fractions) %>%         # Create all combinations of (Dataset, Fraction)
  mutate(SubsampleRows = round(TotalRows * Fraction)) %>%
  mutate(Fraction = paste0("Frac_", Fraction * 100, "%")) %>%  # Create nicer column names
  pivot_wider(names_from = Fraction, values_from = SubsampleRows)

# --- Use the wide table as the final summary (it already includes TotalRows) ---
final_summary <- dataset_summary

# --- Output the final summary table as LaTeX (first version) ---
latex_table1 <- final_summary %>%
  kbl(format = "latex", booktabs = TRUE,
      caption = "Total Annotated Rows and Subsample Counts for Each Dataset",
      # For counts, we show no decimals:
      digits = 0) %>%
  kable_styling(latex_options = c("hold_position", "scale_down"))

# Print to console and save to a .tex file:
cat(latex_table1)
writeLines(latex_table1, file.path(output_dir, "final_summary_rows_subsamples.tex"))

# --- Save the final summary as a CSV file ---
write.csv(final_summary, file.path(output_dir, "final_summary_rows_subsamples.csv"), row.names = FALSE)
print(final_summary)

# --- Define a simple token count function ---
count_tokens <- function(x) {
  # Remove any NA, convert to character, split on whitespace, and count tokens.
  sapply(x, function(str) {
    if (is.na(str)) return(0)
    length(unlist(strsplit(as.character(str), "\\s+")))
  })
}

# --- Compute total input and output tokens per dataset ---
token_summary <- combined_data %>%
  group_by(Dataset) %>%
  summarise(
    total_input_tokens  = sum(count_tokens(text)),
    total_output_tokens = sum(count_tokens(annotation)),
    .groups = "drop"
  )

final_summary <- final_summary %>%
  left_join(token_summary, by = "Dataset") %>%
  # Calculate average input tokens per API call (rounded to an integer)
  mutate(avg_input_tokens_per_api_call = round(total_input_tokens / TotalRows, 0)) %>%
  # Calculate cost estimates
  mutate(cost_input  = (total_input_tokens / 1e6) * 0.50,
         cost_output = (total_output_tokens / 1e6) * 1.50,
         total_cost  = cost_input + cost_output) %>%
  # Reorder columns: Dataset, TotalRows, token columns, cost columns, then fraction columns.
  select(Dataset, TotalRows, total_input_tokens, total_output_tokens, avg_input_tokens_per_api_call, 
         cost_input, cost_output, total_cost, everything())

# --- Output the updated final summary table as LaTeX ---
# Here we want to display cost values with 2 decimals and counts as integers.
latex_table2 <- final_summary %>%
  kbl(format = "latex", booktabs = TRUE,
      caption = "Total Annotated Rows, Token Counts, and Subsample Counts for Each Dataset",
      # Specify digits for columns that are numeric. Since kbl applies digits to all numeric columns,
      # you might instead pre-format columns or use format.args.
      digits = 2,
      format.args = list(nsmall = 2)) %>%  
  kable_styling(latex_options = c("hold_position", "scale_down"))

# Print and save the table:
cat(latex_table2)
writeLines(latex_table2, file.path(output_dir, "final_summary_rows_tokens_subsamples.tex"))

# --- Save the final summary as a CSV file ---
write.csv(final_summary, file.path(output_dir, "final_summary_rows_tokens_subsamples.csv"), row.names = FALSE)
print(final_summary)

# --- Create a totals row for cost estimates ---
fractions_numeric <- c("Frac_2%" = 0.02,
                       "Frac_5%" = 0.05,
                       "Frac_10%" = 0.10,
                       "Frac_25%" = 0.25,
                       "Frac_50%" = 0.50,
                       "Frac_75%" = 0.75)

grand_total_cost <- sum(final_summary$total_cost)
final_fraction_costs <- sapply(fractions_numeric, function(frac) frac * grand_total_cost)

final_row <- tibble(
  Dataset = "Total",
  TotalRows = sum(final_summary$TotalRows),
  total_input_tokens  = sum(final_summary$total_input_tokens),
  total_output_tokens = sum(final_summary$total_output_tokens),
  avg_input_tokens_per_api_call = NA,  # or compute weighted average if needed
  cost_input  = sum(final_summary$cost_input),
  cost_output = sum(final_summary$cost_output),
  total_cost  = grand_total_cost
)

for (col_name in names(final_fraction_costs)) {
  final_row[[col_name]] <- final_fraction_costs[[col_name]]
}

final_summary <- bind_rows(final_summary, final_row)

# --- Output the final summary (with totals) as LaTeX ---
latex_table3 <- final_summary %>%
  kbl(format = "latex", booktabs = TRUE,
      caption = "Total Annotated Rows, Subsample Counts, and Cost Estimates (with Grand Totals)",
      digits = 2,
      format.args = list(nsmall = 2)) %>%
  kable_styling(latex_options = c("hold_position", "scale_down"))

cat(latex_table3)
writeLines(latex_table3, file.path(output_dir, "final_summary_rows_subsamples_with_totals.tex"))
write.csv(final_summary, file.path(output_dir, "final_summary_rows_subsamples_with_totals.csv"), row.names = FALSE)
print(final_summary)

# --- Compute grand totals across all datasets using token_summary ---
grand_totals <- token_summary %>%
  summarise(
    grand_total_input_tokens  = sum(total_input_tokens),
    grand_total_output_tokens = sum(total_output_tokens)
  )

# --- Create a table of models and their pricing (cost per 1M tokens) ---
models_df <- tibble::tribble(
  ~Model,              ~input_price, ~output_price,
  "gpt-4o-mini",       0.15,         0.60,
  "gpt-4o",            2.50,         10.00,
  "o1-mini",           1.10,         4.40,
  "o1",                15.00,        60.00,
  "o3-mini",           1.10,         4.40,
  "claude-3.5-haiku",   0.8,          4.00,
  "claude-3.5-sonnet",  3.00,         15.00,
  "deepseek-v3",       0.14,         0.28,
  "deepseek-r1",       0.55,         2.19,
  "gemini-1.5-flash",  0.15,         0.60,
  "gemini-1.5-pro",    1.25,         5.00,
  "mistral-small",     0.20,         0.60,
  "mistral-large",     2.00,         6.00
)

# --- Calculate cost estimates for each model ---
models_df <- models_df %>%
  mutate(
    cost_input  = (grand_totals$grand_total_input_tokens / 1e6) * input_price,
    cost_output = (grand_totals$grand_total_output_tokens / 1e6) * output_price,
    total_cost  = cost_input + cost_output
  )

# --- Format cost columns so that whole numbers appear as integers and non-integers have two decimals ---
models_df <- models_df %>%
  mutate(
    cost_input  = ifelse(cost_input %% 1 == 0,
                         as.character(as.integer(cost_input)),
                         as.character(round(cost_input, 2))),
    cost_output = ifelse(cost_output %% 1 == 0,
                         as.character(as.integer(cost_output)),
                         as.character(round(cost_output, 2))),
    total_cost  = ifelse(total_cost %% 1 == 0,
                         as.character(as.integer(total_cost)),
                         as.character(round(total_cost, 2)))
  )

# --- Output the models cost table as a LaTeX table ---
latex_models <- models_df %>%
  kbl(format = "latex", booktabs = TRUE,
      caption = "Estimated Total API Cost for Various Models") %>%
  kable_styling(latex_options = c("hold_position", "scale_down"))

cat(latex_models)
writeLines(latex_models, file.path(output_dir, "models_cost_estimate.tex"))

# --- Save the models cost table as a CSV file ---
write.csv(models_df, file.path(output_dir, "models_cost_estimate.csv"), row.names = FALSE)
print(models_df)
