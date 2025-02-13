library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)


between_files <- list(
  'News' = 'data/annotated/news_between_updated.csv',
  'Stance (Long)' = 'data/annotated/stance_long_between_updated.csv',
  'Synthetic (Short)' = 'data/annotated/synth_short_between_updated.csv'
)
# Define a color palette
color_palette <- c(
  'News' = 'orange',
  'Stance (Long)' = 'deeppink',
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
    theme(
          legend.position = "none")
  
  if (!is.null(save_path)) {
    ggsave(save_path, plot = g, width = 8, height = 3, dpi = 300)
    cat(sprintf("Plot saved to %s\n", save_path))
  }
  print(g)
}

# Combine "between" datasets and plot
combined_between_data <- combine_files(between_files)

combined_between_data <- combined_between_data %>%
  distinct(temperature, ka_mean, label, .keep_all = T)

plot_combined(combined_between_data, "temperature", "ka_mean", "Temperature", "Inter-PSS", c(0, 1), 
              save_path = "plots/combined_between_updated.png", within = FALSE)