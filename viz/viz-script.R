if (!require("pacman")) install.packages("pacman")
# remotes::install_github("ricardo-bion/ggradar")
pacman::p_load(tidyverse, ggthemes, ggradar, scales, jsonlite)

ggplot2::theme_set(
    theme_fivethirtyeight() +
        theme(
            text = element_text(family = "Roboto Condensed"),
            title = element_text(size = 14),
            plot.subtitle = element_text(size = 12),
            plot.caption = element_text(size = 10),
            axis.title = element_text(size = 14),
            axis.text = element_text(size = 12),
            panel.grid.minor.x = element_blank()
        )
)

train <- fromJSON("data_creation/dataset/eli5-category-train.json")
valid1 <- fromJSON("data_creation/dataset/eli5-category-validation-1.json")
valid2 <- fromJSON("data_creation/dataset/eli5-category-validation-2.json")
test <- fromJSON("data_creation/dataset/eli5-category-test.json")


train_scores <- tibble(
    category = train$category,
    scores = train$answers$score
) %>%
    rowwise() %>%
    mutate(top_score = max(unlist(scores)),
           answer_num = length(scores)) %>%
    group_by(category) %>%
    summarize(
        mean_top_score = mean(top_score),
        mean_all_score = mean(unlist(scores)),
        mean_answer_num = mean(answer_num)
    )

v1_scores <- tibble(
    category = valid1$category,
    scores = valid1$answers$score
) %>%
    rowwise() %>%
    mutate(top_score = max(unlist(scores)),
           answer_num = length(scores)) %>%
    ungroup() %>%
    summarize(
        mean_top_score = mean(top_score),
        mean_all_score = mean(unlist(scores)),
        mean_answer_num = mean(answer_num)
    ) %>%
    mutate(category = "Culture") %>%
    relocate(category, .before = mean_top_score)


v2_scores <- tibble(
    category = valid2$category,
    scores = valid2$answers$score
) %>%
    rowwise() %>%
    mutate(top_score = max(unlist(scores)),
           answer_num = length(scores)) %>%
    ungroup() %>%
    summarize(
        mean_top_score = mean(top_score),
        mean_all_score = mean(unlist(scores)),
        mean_answer_num = mean(answer_num)
    ) %>%
    mutate(category = "Repost") %>%
    relocate(category, .before = mean_top_score)

test_scores <- tibble(
    category = test$category,
    scores = test$answers$score
) %>%
    rowwise() %>%
    mutate(top_score = max(unlist(scores)),
           answer_num = length(scores)) %>%
    ungroup() %>%
    summarize(
        mean_top_score = mean(top_score),
        mean_all_score = mean(unlist(scores)),
        mean_answer_num = mean(answer_num)
    ) %>%
    mutate(category = "Engineering") %>%
    relocate(category, .before = mean_top_score)

scores_dat <- bind_rows(train_scores, v1_scores, v2_scores, test_scores)

scores_dat

scores_dat <- scores_dat %>%
    mutate_at(vars(-category), rescale)


scores_radar <- scores_dat %>%
    ggradar(
        font.radar = "Roboto Condensed",
        grid.label.size = 10,
        axis.label.size = 5,
        group.point.size = 3
    ) +
    labs(title = "Radar plot of summary stats of posts by category") + 
    theme_fivethirtyeight() +
    theme(
        legend.text = element_text(size = 14, family = "Roboto Condensed"),
        legend.key = element_rect(fill = NA, color = NA),
        legend.background = element_blank(),
        plot.title.position = "plot"
    )
scores_radar
ggsave(
    filename = here::here("viz", "radar-category.png"),
    plot = scores_radar,
    width = 12,
    height = 9,
    device = "png"
)

loss_dat1 <- read_csv("exp_record/bart-chem-loss-1e-4.csv") %>%
    mutate(setup = "bart-chem-loss-1e-4")
loss_dat2 <- read_csv("exp_record/bart-chem-loss-5e-5.csv") %>%
    mutate(setup = "bart-chem-loss-5e-5")
loss_dat3 <- read_csv("exp_record/bart-random-loss-5e-5.csv") %>%
    mutate(setup = "bart-random-loss-5e-5")
loss_dat <- bind_rows(loss_dat1, loss_dat2, loss_dat3)


ggplot(loss_dat, aes(x=iteration, y=loss, color=setup)) +
    geom_line(size = 1.2, alpha = 0.7) +
    facet_wrap(~epoch) +
    scale_color_brewer(palette = "Set1") +
    labs(
        title = "Loss functions in model training"
    )
ggsave(
    filename = here::here("viz", "loss-functions.png"),
    plot = last_plot(),
    width = 12,
    height = 9,
    device = "png"
)
