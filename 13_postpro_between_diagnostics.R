library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)
library(cowplot)

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

read_and_preserve_types <- function(file) {
  data <- read_csv(file, col_types = cols(.default = "c"), show_col_types = FALSE)
  return(data)
}

combine_files_preserve <- function(files) {
  combined_data <- lapply(names(files), function(label) {
    file_path <- files[[label]]
    annotated_data <- read_and_preserve_types(file_path) %>%
      mutate(label = label)
    return(annotated_data)
  })
  combined_data <- bind_rows(combined_data)
  return(combined_data)
}

combined_between_data <- combine_files_preserve(between_files)


# Plot overall scores

order_by_pss <- combined_between_data %>%
  group_by(label) %>%
  summarise(mean_pss = mean(as.numeric(ka_mean), na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(mean_pss)) %>%
  pull(label)

inter_pss <- combined_between_data %>%
  group_by(label, temperature) %>%
  summarise(mean_inter_pss = mean(as.numeric(ka_mean), na.rm = TRUE), .groups = "drop",
            ka_upper = max(as.numeric(ka_upper),na.rm = TRUE),
            ka_lower = min(as.numeric(ka_lower), na.rm = TRUE))

mean_pss <- combined_between_data %>%
  group_by(label) %>%
  summarise(mean_pss = mean(as.numeric(ka_mean), na.rm = TRUE), .groups = "drop")

plot_data <- left_join(mean_pss, inter_pss, by = c("label")) %>%
  mutate(label = factor(label, levels = order_by_pss),
         temperature = as.numeric(temperature))

split_data <- plot_data %>%
  group_split(label)

plot_single_dataset <- function(df) {
  this_label <- unique(df$label)
  
  ggplot(df) +
    geom_line(
      aes(x = temperature, y = mean_inter_pss, color = label),
      size = 1, group = 1
    ) +
    geom_point(
      aes(x = temperature, y = mean_inter_pss, color = label),
      size = 1
    ) +
    geom_errorbar(aes(x = temperature, y = mean_inter_pss, ymin = ka_lower, ymax = ka_upper, alpha = 0.1), width = 0.2) +
    geom_hline(aes(yintercept = mean_pss, color = "gray"), linetype = "dashed", size = 0.8) +
    geom_hline(aes(yintercept = 0.8, color = "black"), linetype = "dashed", size = 0.8) +
    geom_text(aes(x = Inf, y = mean_pss, label = round(mean_pss, 2), color = "black"),
              hjust = 1.1, vjust = -0.5, size = 6, show.legend = FALSE) +
    scale_color_manual(values = color_palette) +
    labs(
      x = "temperature",
      y = "Inter-PSS",
      title = paste(this_label)
    ) +
    ylim(0,1) +
    theme_minimal() +
    theme(
      legend.position = "none",
      strip.text = element_text(size = 10)
    )
}

plot_list <- lapply(split_data, plot_single_dataset)

final_plot <- cowplot::plot_grid(plotlist = plot_list, ncol = 4)

print(final_plot)

ggsave("plots/combined_between_expanded.png", plot = final_plot, width = 18, height = 10, dpi = 300)

# Plot overall scores and unique counts to check annotation performance

unique_counts <- combined_between_data %>%
  group_by(label, temperature) %>%
  summarise(unique_annotations_count = n_distinct(annotation), .groups = "drop")

inter_pss <- combined_between_data %>%
  group_by(label, temperature) %>%
  summarise(mean_inter_pss = mean(as.numeric(ka_mean), na.rm = TRUE), .groups = "drop")

plot_data <- left_join(unique_counts, inter_pss, by = c("label", "temperature")) %>%
  mutate(temperature = factor(temperature, levels = unique(temperature)))

order_by_pss <- plot_data %>%
  group_by(label) %>%
  summarise(mean_pss = mean(mean_inter_pss, na.rm = TRUE)) %>%
  arrange(desc(mean_pss)) %>%
  pull(label)

plot_data <- plot_data %>%
  mutate(label = factor(label, levels = order_by_pss))

split_data <- plot_data %>%
  group_split(label)

plot_single_dataset <- function(df) {
  this_label <- unique(df$label)
  
  # Safeguard for dividing by zero
  max_unique <- max(df$unique_annotations_count, na.rm = TRUE)
  max_pss    <- max(df$mean_inter_pss, na.rm = TRUE)
  if (max_unique == 0) max_unique <- 1
  if (max_pss == 0)    max_pss    <- 1
  
  # Ratio to align Inter-PSS onto the left axis
  ratio <- max_unique / max_pss
  
  ggplot(df, aes(x = temperature)) +
    geom_bar(
      aes(y = unique_annotations_count, fill = label),
      stat = "identity", alpha = 0.5
    ) +
    geom_line(
      aes(y = mean_inter_pss * ratio, color = "black"),
      size = 1, group = 1
    ) +
    geom_point(
      aes(y = mean_inter_pss * ratio, color = "black"),
      size = 1
    ) +
    scale_y_continuous(
      name = "Unique annotations",         # left-axis label
      sec.axis = sec_axis(
        trans = ~ . / ratio,               # map back to original Inter-PSS
        name = "Inter-PSS"                 # right-axis label
      )
    ) +
    scale_color_manual(values = color_palette) +
    scale_fill_manual(values = color_palette) +
    labs(
      x = "Temperature",
      title = paste(this_label)
    ) +
    theme_minimal() +
    theme(
      strip.text       = element_text(size = 14),
      axis.text.x      = element_text(size = 12, angle = 90, vjust = 0.5, hjust = 1),
      axis.text.y      = element_text(size = 12),
      axis.title       = element_text(size = 16),
      panel.grid.major = element_line(size = 0.1, linetype = 'solid', color = 'grey80'),
      panel.grid.minor = element_line(size = 0.1, linetype = 'solid', color = 'grey80'),
      legend.position  = "none",
      plot.title       = element_text(size = 14, hjust = 0.5)
    )
}

plot_list <- lapply(split_data, plot_single_dataset)

final_plot <- cowplot::plot_grid(plotlist = plot_list, ncol = 4)

print(final_plot)

ggsave("plots/combined_postpro_between_diagnostics.png", plot = final_plot, width = 16, height = 12, dpi = 300)
