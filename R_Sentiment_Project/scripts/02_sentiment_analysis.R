# ------------------------------------------------------------------
# 02_sentiment_analysis.R
# End-to-end sentiment analysis of 50,000 product reviews in R.
# Stages: load -> clean -> visualise -> feature engineering ->
#         split -> train (LM, Random Forest, Logistic) -> compare
# Run from the project root:  Rscript scripts/02_sentiment_analysis.R
# ------------------------------------------------------------------
suppressPackageStartupMessages({
  library(ggplot2); library(corrplot); library(caret)
  library(randomForest); library(pROC)
})
options(width = 105, digits = 5)
dir.create("console", showWarnings = FALSE); dir.create("plots", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

# helper: run code, echo it like the R console, and save the transcript
snap <- function(name, code) {
  out <- capture.output(source(exprs = parse(text = code, keep.source = TRUE), echo = TRUE,
                               max.deparse.length = Inf, local = globalenv()))
  writeLines(out, file.path("console", paste0(name, ".txt")))
  invisible(out)
}
save_plot <- function(p, name, w = 7, h = 4.5)
  ggsave(file.path("plots", name), p, width = w, height = h, dpi = 150, bg = "white")
cols <- c(Negative = "#C0504D", Neutral = "#9AA5AB", Positive = "#4F8A6F")

# ================= LOADING & UNDERSTANDING THE DATA ================
snap("03_load",      'reviews <- read.csv("data/product_reviews_50k.csv", stringsAsFactors = FALSE)
head(reviews[, c(1, 3:9, 22, 23)], 6)')
snap("04_dim_names", 'dim(reviews)
names(reviews)')
snap("05_str",       'str(reviews)')
snap("06_summary",   'summary(reviews[, c("price_usd","discount_pct","delivery_days","reviewer_age","star_rating","sentiment_score")])')

# ================= STAGE 1: DATA CLEANING ===========================
snap("07_missing",   'na_summary <- colSums(is.na(reviews))
na_summary[na_summary > 0]')
snap("08_impute",    'for (col in c("reviewer_age", "delivery_days", "reviewer_review_count")) {
  reviews[[col]][is.na(reviews[[col]])] <- median(reviews[[col]], na.rm = TRUE)
}
cat("Missing values after median imputation:", sum(is.na(reviews)), "\\n")')
snap("09_duplicates",'dup_count <- sum(duplicated(reviews))
cat("Duplicate rows:", dup_count, "\\n")
cat("Duplicate review IDs:", sum(duplicated(reviews$review_id)), "\\n")')
snap("10_types",     'reviews$review_date     <- as.Date(reviews$review_date)
reviews$sentiment_label <- factor(reviews$sentiment_label, levels = c("Negative","Neutral","Positive"))
for (f in c("product_category","brand","device_type","region")) reviews[[f]] <- factor(reviews[[f]])
table(reviews$sentiment_label)
round(prop.table(table(reviews$sentiment_label)) * 100, 1)')
snap("11_outliers",  'q1 <- quantile(reviews$price_usd, 0.25); q3 <- quantile(reviews$price_usd, 0.75)
iqr_out <- sum(reviews$price_usd > q3 + 1.5 * (q3 - q1))
cat("Price outliers (IQR rule):", iqr_out, "\\n")
cap <- quantile(reviews$price_usd, 0.99)
reviews$price_usd <- pmin(reviews$price_usd, cap)
cat("Price capped at 99th percentile:", round(cap, 2), "\\n")')

# ================= STAGE 2: DATA VISUALISATION ======================
save_plot(ggplot(reviews, aes(sentiment_label, fill = sentiment_label)) + geom_bar(width = .6) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.4, size = 3.5) +
  scale_fill_manual(values = cols) + labs(title = "Distribution of Sentiment Labels", x = NULL, y = "Reviews") +
  theme_minimal(base_size = 12) + theme(legend.position = "none"), "plot_01_sentiment_distribution.png")
save_plot(ggplot(reviews, aes(sentiment_score, fill = sentiment_label)) + geom_histogram(bins = 50) +
  scale_fill_manual(values = cols) + labs(title = "Distribution of Sentiment Score", x = "Sentiment score (-1 to +1)", y = "Count", fill = NULL) +
  theme_minimal(base_size = 12), "plot_02_score_histogram.png")
save_plot(ggplot(reviews, aes(product_category, fill = sentiment_label)) + geom_bar(position = "fill") +
  scale_fill_manual(values = cols) + scale_y_continuous(labels = scales::percent) +
  labs(title = "Sentiment Mix by Product Category", x = NULL, y = "Share of reviews", fill = NULL) +
  theme_minimal(base_size = 11) + theme(axis.text.x = element_text(angle = 30, hjust = 1)), "plot_03_category_sentiment.png")
save_plot(ggplot(reviews, aes(factor(star_rating), fill = sentiment_label)) + geom_bar(position = "fill") +
  scale_fill_manual(values = cols) + scale_y_continuous(labels = scales::percent) +
  labs(title = "Star Rating vs Sentiment", x = "Star rating", y = "Share of reviews", fill = NULL) +
  theme_minimal(base_size = 12), "plot_04_rating_vs_sentiment.png")
delivery <- aggregate(sentiment_score ~ delivery_days, reviews, mean)
save_plot(ggplot(delivery, aes(delivery_days, sentiment_score)) + geom_line(colour = "#4F8A6F", linewidth = 1) + geom_point(size = 2.5, colour = "#4F8A6F") +
  labs(title = "Average Sentiment Score by Delivery Time", x = "Delivery days", y = "Mean sentiment score") + theme_minimal(base_size = 12), "plot_05_delivery_trend.png")

snap("12_corr", 'num_vars <- reviews[, sapply(reviews, is.numeric)]
cor_mat  <- cor(num_vars, use = "pairwise.complete.obs")
dim(cor_mat)
round(sort(cor_mat[, "sentiment_score"], decreasing = TRUE), 3)')
png("plots/plot_06_correlation.png", width = 1500, height = 1500, res = 150)
corrplot(cor_mat, method = "color", tl.cex = 0.95, tl.col = "black", tl.srt = 45, addCoef.col = "black", number.cex = 0.5,
         col = colorRampPalette(c("#C0504D", "white", "#2E75B6"))(200), mar = c(0, 0, 1, 0))
dev.off()
box_df <- stack(as.data.frame(scale(num_vars[, c("price_usd","discount_pct","delivery_days","reviewer_age",
                  "reviewer_review_count","helpful_votes","total_votes","star_rating")])))
save_plot(ggplot(box_df, aes(ind, values)) + geom_boxplot(fill = "#B7C4CB", outlier.size = .6, outlier.alpha = .4) +
  labs(title = "Boxplots for Checking Outliers (standardised)", x = NULL, y = "z-score") +
  theme_minimal(base_size = 11) + theme(axis.text.x = element_text(angle = 30, hjust = 1)), "plot_07_boxplots.png", 8, 4.5)

# ================= STAGE 3: FEATURE ENGINEERING (TEXT) ===============
pos_lex <- c("excellent","amazing","fantastic","great","wonderful","superb","impressive","perfect","outstanding",
             "reliable","smooth","beautiful","comfortable","good","solid","recommend","happy","worth")
neg_lex <- c("terrible","awful","poor","disappointing","horrible","cheap","flimsy","useless","defective","broken",
             "bad","noisy","overpriced","unreliable","faulty","waste","refund","regret","disappointed")
neg_words <- c("not","never","no","hardly")
snap("13_features", 'txt    <- reviews$review_text
tokens <- strsplit(tolower(txt), "[^a-z]+")
tokens <- lapply(tokens, function(t) t[t != ""])
count_in <- function(lex) vapply(tokens, function(t) sum(t %in% lex), numeric(1))

reviews$word_count          <- lengths(tokens)
reviews$char_count          <- nchar(txt)
reviews$avg_word_length     <- round(vapply(tokens, function(t) mean(nchar(t)), numeric(1)), 3)
reviews$sentence_count      <- nchar(gsub("[^.!?]", "", txt))
reviews$exclamation_count   <- nchar(gsub("[^!]", "", txt))
reviews$question_count      <- nchar(gsub("[^?]", "", txt))
reviews$uppercase_word_count<- vapply(strsplit(txt, " "), function(w) sum(grepl("^[A-Z]{3,}[.!?]*$", w)), numeric(1))
reviews$pos_word_count      <- count_in(pos_lex)
reviews$neg_word_count      <- count_in(neg_lex)
reviews$negation_count      <- count_in(neg_words)
reviews$pos_ratio           <- round(reviews$pos_word_count / reviews$word_count, 4)
reviews$neg_ratio           <- round(reviews$neg_word_count / reviews$word_count, 4)
reviews$net_price           <- round(reviews$price_usd * (1 - reviews$discount_pct / 100), 2)
reviews$helpful_ratio       <- round(reviews$helpful_votes / (reviews$total_votes + 1), 4)
dim(reviews)')
snap("14_features_head", 'head(reviews[, c("word_count","char_count","exclamation_count","pos_word_count","neg_word_count","negation_count","pos_ratio","neg_ratio")], 6)')
long <- stack(reviews[, c("pos_word_count","neg_word_count")]); long$label <- rep(reviews$sentiment_label, 2)
save_plot(ggplot(long, aes(label, values, fill = label)) + geom_boxplot(outlier.alpha = .3) + facet_wrap(~ ind) +
  scale_fill_manual(values = cols) + labs(title = "Lexicon Word Counts by Sentiment", x = NULL, y = "Words per review") +
  theme_minimal(base_size = 11) + theme(legend.position = "none"), "plot_08_text_features.png", 8, 4.2)

# ================= STAGE 4: DATA SPLITTING ===========================
snap("15_split", 'set.seed(123)
trainIndex <- createDataPartition(reviews$sentiment_label, p = 0.8, list = FALSE)
train <- reviews[trainIndex, ]
test  <- reviews[-trainIndex, ]
cat("Train rows:", nrow(train), "| Test rows:", nrow(test), "\\n")')

features <- setdiff(names(reviews), c("review_id","review_text","review_date","sentiment_score","sentiment_label"))
snap("16_features_list", 'length(features)
features')

# ================= STAGE 5: MODEL TRAINING ===========================
# ---- 5a. Linear Regression (predict continuous sentiment score) ----
snap("17_lm", 'lm_model <- lm(sentiment_score ~ ., data = train[, c(features, "sentiment_score")])
lm_sum   <- summary(lm_model)
cat("Multiple R-squared (train):", round(lm_sum$r.squared, 4), "\\n")
cat("Predictors used:", length(coef(lm_model)) - 1, "\\n")
top <- coef(lm_sum)[order(-abs(coef(lm_sum)[, "t value"])), ]
round(top[2:9, ], 4)')
snap("18_lm_eval", 'lm_pred  <- predict(lm_model, newdata = test)
lm_res   <- postResample(lm_pred, test$sentiment_score)
round(lm_res, 4)')
set.seed(1); idx <- sample(nrow(test), 3000)
save_plot(ggplot(data.frame(Actual = test$sentiment_score[idx], Predicted = lm_pred[idx]), aes(Actual, Predicted)) +
  geom_point(alpha = .25, colour = "#2E75B6", size = 1) + geom_abline(colour = "#C0504D", linewidth = 1) +
  labs(title = "Linear Regression: Actual vs Predicted", x = "Actual sentiment score", y = "Predicted") +
  theme_minimal(base_size = 12), "plot_09_lr_actual_vs_pred.png", 6, 4.8)

# ---- 5b. Random Forest (classify Negative / Neutral / Positive) ----
snap("19_rf", 'set.seed(42)
rf_model <- randomForest(sentiment_label ~ ., data = train[, c(features, "sentiment_label")],
                         ntree = 100, mtry = 6, nodesize = 5, importance = TRUE)
print(rf_model)')
snap("20_rf_eval", 'rf_pred <- predict(rf_model, newdata = test)
rf_cm   <- confusionMatrix(rf_pred, test$sentiment_label)
rf_cm$table
round(rf_cm$overall[c("Accuracy","Kappa")], 4)')
imp <- data.frame(Feature = rownames(importance(rf_model)), Gain = importance(rf_model)[, "MeanDecreaseGini"])
imp <- head(imp[order(-imp$Gain), ], 15)
save_plot(ggplot(imp, aes(reorder(Feature, Gain), Gain)) + geom_col(fill = "#4F8A6F") + coord_flip() +
  labs(title = "Random Forest: Top 15 Important Features", x = NULL, y = "Mean decrease in Gini") + theme_minimal(base_size = 12),
  "plot_10_rf_importance.png", 7, 5)
cm_df <- as.data.frame(rf_cm$table)
save_plot(ggplot(cm_df, aes(Reference, Prediction, fill = Freq)) + geom_tile() + geom_text(aes(label = Freq), size = 5) +
  scale_fill_gradient(low = "white", high = "#4F8A6F") + labs(title = "Random Forest: Confusion Matrix (Test Set)", x = "Actual", y = "Predicted") +
  theme_minimal(base_size = 12) + theme(legend.position = "none"), "plot_11_rf_confusion.png", 6, 4.6)

# ---- 5c. Logistic Regression (GLM): Positive vs Not Positive -------
snap("21_glm", 'train$is_positive <- factor(ifelse(train$sentiment_label == "Positive", "Positive", "Not_Positive"), levels = c("Not_Positive","Positive"))
test$is_positive  <- factor(ifelse(test$sentiment_label  == "Positive", "Positive", "Not_Positive"), levels = c("Not_Positive","Positive"))
glm_model <- glm(is_positive ~ ., data = train[, c(features, "is_positive")], family = binomial)
cat("AIC:", round(AIC(glm_model), 1), "\\n")
round(head(coef(summary(glm_model))[order(-abs(coef(summary(glm_model))[, "z value"])), ], 8), 4)')
snap("22_glm_eval", 'glm_prob <- predict(glm_model, newdata = test, type = "response")
glm_pred <- factor(ifelse(glm_prob > 0.5, "Positive", "Not_Positive"), levels = c("Not_Positive","Positive"))
glm_cm   <- confusionMatrix(glm_pred, test$is_positive, positive = "Positive")
glm_cm$table
glm_auc  <- as.numeric(auc(roc(test$is_positive, glm_prob, quiet = TRUE)))
cat("Accuracy:", round(glm_cm$overall["Accuracy"], 4), "| AUC:", round(glm_auc, 4), "\\n")')
roc_obj <- roc(test$is_positive, glm_prob, quiet = TRUE)
roc_df  <- data.frame(FPR = 1 - roc_obj$specificities, TPR = roc_obj$sensitivities)
save_plot(ggplot(roc_df, aes(FPR, TPR)) + geom_line(colour = "#2E75B6", linewidth = 1) + geom_abline(linetype = 2, colour = "grey50") +
  annotate("text", x = .7, y = .15, label = paste("AUC =", round(glm_auc, 3)), size = 5) +
  labs(title = "Logistic Regression: ROC Curve", x = "False positive rate", y = "True positive rate") +
  theme_minimal(base_size = 12), "plot_12_glm_roc.png", 6, 4.8)

# ================= STAGE 6: COMPARISON ==============================
cmp <- data.frame(
  Model   = c("Linear Regression", "Random Forest", "Logistic Regression"),
  Type    = c("Regression (score)", "Classification (3-class)", "Classification (binary)"),
  Metric  = c(round(lm_res["Rsquared"], 4), round(rf_cm$overall["Accuracy"], 4), round(glm_cm$overall["Accuracy"], 4)),
  MetricName = c("R-squared", "Accuracy", "Accuracy"),
  RMSE    = c(round(lm_res["RMSE"], 4), NA, NA),
  AUC_or_MAE = c(round(lm_res["MAE"], 4), NA, round(glm_auc, 4)),
  stringsAsFactors = FALSE)
rf_cls <- rf_cm$byClass
cmp$Model <- factor(cmp$Model, levels = cmp$Model)
cmp$F1 <- c(NA, round(mean(rf_cls[, "F1"]), 4), round(glm_cm$byClass["F1"], 4))
write.csv(cmp, "results/model_comparison.csv", row.names = FALSE)
snap("23_compare", 'print(cmp, row.names = FALSE)')
save_plot(ggplot(cmp, aes(Model, Metric, fill = Model)) + geom_col(width = .55) + geom_text(aes(label = Metric), vjust = -0.4) +
  scale_fill_manual(values = c("#2E75B6","#4F8A6F","#C98B3C")) + ylim(0, 1.05) +
  labs(title = "Model Comparison (R-squared / Accuracy on test set)", x = NULL, y = NULL) +
  theme_minimal(base_size = 12) + theme(legend.position = "none"), "plot_13_model_comparison.png")

write.csv(reviews[, setdiff(names(reviews), "review_text")][1:5, ], "results/sample_engineered_rows.csv", row.names = FALSE)
cat("\nDone.\n")
