library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)

# Define file paths and labels for the datasets
within_files <- list(
  'Tweets (Rep. Dem.)' = 'data/annotated/tweets_rd_within_expanded.csv',
  'Tweets (Populism)' = 'data/annotated/tweets_pop_within_expanded.csv',
  'News' = 'data/annotated/news_within_expanded.csv',
  'News (Short)' = 'data/annotated/news_short_within_expanded.csv',
  'Manifestos' = 'data/annotated/manifestos_within_expanded.csv',
  'Manifestos Multi' = 'data/annotated/manifestos_multi_within_expanded.csv',
  'Stance' = 'data/annotated/stance_within_expanded.csv',
  'Stance (Long)' = 'data/annotated/stance_long_within_expanded.csv',
  'MII' = 'data/annotated/mii_within_expanded.csv',
  'MII (Long)' = 'data/annotated/mii_long_within_expanded.csv',
  'Synthetic' = 'data/annotated/synth_within_expanded.csv',
  'Synthetic (Short)' = 'data/annotated/synth_short_within_expanded.csv'
)

between_files <- list(
  'Tweets (Rep. Dem.)' = 'data/annotated/tweets_rd_between_expanded.csv',
  'Tweets (Populism)' = 'data/annotated/tweets_pop_between_expanded.csv',
  'News' = 'data/annotated/news_between_expanded.csv',
  'News (Short)' = 'data/annotated/news_short_between_expanded.csv',
  'Manifestos' = 'data/annotated/manifestos_between_expanded.csv',
  'Manifestos Multi' = 'data/annotated/manifestos_multi_between_expanded.csv',
  'Stance' = 'data/annotated/stance_between_expanded.csv',
  'Stance (Long)' = 'data/annotated/stance_long_between_expanded.csv',
  'MII' = 'data/annotated/mii_between_expanded.csv',
  'MII (Long)' = 'data/annotated/mii_long_between_expanded.csv',
  'Synthetic' = 'data/annotated/synth_between_expanded.csv',
  'Synthetic (Short)' = 'data/annotated/synth_short_between_expanded.csv'
)

# Define a color palette
color_palette <- c(
  'Tweets (Rep. Dem.)' = 'darkcyan',
  'Tweets (Populism)' = 'cyan',
  'News' = 'orange',
  'News (Short)' = 'darkorange',
  'Manifestos' = 'green',
  'Manifestos Multi' = 'red',
  'Stance' = 'hotpink',
  'Stance (Long)' = 'deeppink',
  'MII' = 'mediumseagreen',
  'MII (Long)' = 'seagreen',
  'Synthetic' = 'indianred',
  'Synthetic (Short)' = 'brown'
)

# Function to read a dataset and ensure consistent column types
read_and_standardize <- function(file, types) {
  data <- read_csv(file, col_types = types, show_col_types = FALSE)
  return(data)
}

# Function to determine column types
determine_column_types <- function(files) {
  sample_data <- read_csv(files[[1]], show_col_types = FALSE)
  types <- sapply(sample_data, class)
  return(types)
}

# Function to convert column types to standard types
convert_column_types <- function(data, types) {
  for (col in names(types)) {
    if (types[col] == "numeric") {
      data[[col]] <- as.numeric(data[[col]])
    } else if (types[col] == "factor" || types[col] == "character") {
      data[[col]] <- as.character(data[[col]])
    }
  }
  return(data)
}

combine_files <- function(files) {
  types <- determine_column_types(files)
  combined_data <- lapply(names(files), function(label) {
    file_path <- files[[label]]
    annotated_data <- read_and_standardize(file_path, types)
    annotated_data <- annotated_data %>%
      drop_na(ka_mean) %>%
      mutate(label = label)
    annotated_data <- convert_column_types(annotated_data, types)
    return(annotated_data)
  })
  combined_data <- bind_rows(combined_data)
  return(combined_data)
}

plot_combined <- function(data, x, y, xlabel, ylabel, ylim_vals, save_path = NULL, within = TRUE) {
  if (within) {
    order <- data %>%
      group_by(label) %>%
      summarise(mean_ka = mean(ka_mean, na.rm = TRUE)) %>%
      arrange(desc(mean_ka)) %>%
      pull(label)
  } else {
    data <- data %>%
      group_by(temperature, label) %>%
      summarise(across(starts_with("ka_"), mean, na.rm = TRUE)) %>%
      ungroup()
    order <- data %>%
      group_by(label) %>%
      summarise(mean_ka = mean(ka_mean, na.rm = TRUE)) %>%
      arrange(desc(mean_ka)) %>%
      pull(label)
  }
  
  data$label <- factor(data$label, levels = order)
  
  if (within) {
    mean_pss <- data %>%
      group_by(label) %>%
      summarise(mean_pss = mean(ka_mean, na.rm = TRUE))
    
    aspect_ratio <- 100
  } else {
    mean_pss <- data %>%
      group_by(label) %>%
      summarise(mean_pss = mean(ka_mean, na.rm = TRUE))
    
    aspect_ratio <- 4
  }
  
  g <- ggplot(data, aes_string(x = x, y = y, color = "label")) +
    geom_line(aes(group = label, alpha = 0.1), size = 1) +
    geom_point(aes(alpha = 0.1), size = 1) +
    geom_errorbar(aes(ymin = ka_lower, ymax = ka_upper, alpha = 0.1), width = 0.2) +
    geom_smooth(method = "loess", se = FALSE, size = 1, span = 2) +
    geom_hline(data = mean_pss, aes(yintercept = mean_pss, color = label), linetype = "dashed", size = 0.8) +
    geom_hline(aes(yintercept = 0.8, color = "black"), linetype = "dashed", size = 0.8) +
    geom_text(data = mean_pss, aes(x = Inf, y = mean_pss, label = round(mean_pss, 2), color = "black"),
              hjust = 1.1, vjust = -0.5, size = 6, show.legend = FALSE) +
    scale_color_manual(values = color_palette) +
    facet_wrap(~ label, ncol = 4) +
    labs(x = xlabel, y = ylabel) +
    ylim(ylim_vals) +
    theme_minimal() +
    theme(strip.text = element_text(size = 14),
          axis.text = element_text(size = 12),
          axis.title = element_text(size = 16, face = "bold"),
          panel.grid.major = element_line(size = 0.1, linetype = 'solid', color = 'grey80'),
          panel.grid.minor = element_line(size = 0.1, linetype = 'solid', color = 'grey80'),
          legend.position = "none")
  
  if (!is.null(save_path)) {
    ggsave(save_path, plot = g, width = 16, height = 12, dpi = 300)
    cat(sprintf("Plot saved to %s\n", save_path))
  }
  print(g)
}

# Combine "within" datasets and plot
combined_within_data <- combine_files(within_files)

combined_within_data <- combined_within_data %>%
  distinct(iteration, ka_mean, label, .keep_all = T)

plot_combined(combined_within_data, "iteration", "ka_mean", "Iteration", "Intra-PSS", c(0.5, 1), 
              save_path = "plots/combined_within_expanded.png")

# Combine "between" datasets and plot
combined_between_data <- combine_files(between_files)

combined_between_data <- combined_between_data %>%
  distinct(temperature, ka_mean, label, .keep_all = T)

plot_combined(combined_between_data, "temperature", "ka_mean", "Temperature", "Inter-PSS", c(0, 1), 
              save_path = "plots/combined_between_expanded.png", within = FALSE)