library(dplyr)
library(readr)
library(tidyr)
library(stringr)
library(tidylog)

# Define file paths
between_files <- list(
  'tweets_rd' = 'data/annotated/tweets_rd_between_expanded.csv',
  'tweets_pop' = 'data/annotated/tweets_pop_between_expanded.csv',
  'news' = 'data/annotated/news_between_expanded.csv',
  'news_short' = 'data/annotated/news_short_between_expanded.csv',
  'manifestos' = 'data/annotated/manifestos_between_expanded.csv',
  'manifestos_multi' = 'data/annotated/manifestos_multi_between_expanded.csv',
  'stance' = 'data/annotated/stance_between_expanded.csv',
  'stance_long' = 'data/annotated/stance_long_between_expanded.csv',
  'mii' = 'data/annotated/mii_between_expanded.csv',
  'mii_long' = 'data/annotated/mii_long_between_expanded.csv',
  'synth' = 'data/annotated/synth_between_expanded.csv',
  'synth_short' = 'data/annotated/synth_short_between_expanded.csv'
)

# Define valid annotations for each dataset
valid_annotations_map <- list(
  'mii' = c(12, 15, 26, 32, 40, 48),
  'mii_long' = c(48, 15, 32, 40, 26, 12, 4, 1, 31, 22, 5, 14),
  'tweets_rd' = c(1, 0),
  'tweets_pop' = c(1, 0),
  'news' = c(0, 1, 2),
  'news_short' = c(0, 1, 2),
  'manifestos' = c(0, 1),
  'manifestos_multi' = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
  'stance' = c(0, 1),
  'stance_long' = c(0, 1, 2),
  'synth' = c(0, 1),
  'synth_short' = c(0, 1)
)

# Function to load data without altering data types
read_and_preserve_types <- function(file) {
  read_csv(file, col_types = cols(.default = "c"), show_col_types = FALSE)
}

# Combine datasets
combine_files_preserve <- function(files) {
  combined_data <- lapply(names(files), function(label) {
    file_path <- files[[label]]
    annotated_data <- read_and_preserve_types(file_path) %>%
      mutate(label = label)
    return(annotated_data)
  })
  bind_rows(combined_data)
}

# Combine the "between" datasets
combined_between_data <- combine_files_preserve(between_files)

# Function to debug filtering process
debug_filtering <- function(dataset, label, valid_annotations) {
  print(paste("Processing dataset:", label))
  print(paste("Rows before filtering:", nrow(dataset)))
  
  # Ensure `annotation` column is numeric
  dataset <- dataset %>%
    mutate(annotation = as.numeric(annotation))
  
  # Filter based on valid annotations
  filtered_data <- dataset %>%
    filter(annotation %in% valid_annotations)
  
  print(paste("Rows after filtering:", nrow(filtered_data)))
  return(filtered_data)
}

# Process each dataset
for (cur_label in names(valid_annotations_map)) {
  valid_annotations <- valid_annotations_map[[cur_label]]
  
  # Filter data for the current label
  dataset <- combined_between_data %>%
    filter(label == cur_label)
  
  # Debug filtering
  dataset <- debug_filtering(dataset, cur_label, valid_annotations)
  
  if (nrow(dataset) > 0) {
    # Count the number of annotations at each temperature
    annotation_counts <- dataset %>%
      group_by(temperature) %>%
      summarise(n_annotations = n(), .groups = "drop")
    
    # Find the smallest number of annotations across all temperatures
    min_n <- min(annotation_counts$n_annotations)
    
    # Sample the same number of annotations for each temperature
    balanced_data <- dataset %>%
      group_by(temperature) %>%
      slice_sample(n = min_n) %>%
      ungroup()
    
    # Save the filtered and balanced datasets
    filtered_path <- paste0("data/annotated/reannotated/", cur_label, "_filtered.csv")
    balanced_path <- paste0("data/annotated/reannotated/", cur_label, "_filtered_balanced.csv")
    
    write_csv(dataset, filtered_path)
    write_csv(balanced_data, balanced_path)
  }
}