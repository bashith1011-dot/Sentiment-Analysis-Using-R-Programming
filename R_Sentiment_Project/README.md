# Sentiment Analysis of Product Reviews Using R

R mini project: predicts the sentiment (Negative / Neutral / Positive) and sentiment score of
50,000 e-commerce product reviews using 32 features (18 order/reviewer features + 14 text features).

**Author:** Mohamed Abdul Bashith A, MCA, Karpagam College of Engineering

## Pipeline
Loading -> Stage 1 Cleaning -> Stage 2 Visualization -> Stage 3 Feature engineering ->
Stage 4 Split (80/20) -> Stage 5 Models (Linear Regression, Random Forest, Logistic Regression) -> Comparison

## Results (test set, 9,998 reviews)
| Model | Metric | Value |
|---|---|---|
| Linear Regression | R-squared / RMSE | 0.8218 / 0.2650 |
| Random Forest (3-class) | Accuracy | 0.8089 |
| Logistic Regression (binary) | Accuracy / AUC | 0.8978 / 0.9658 |

## Run
```
Rscript scripts/01_generate_dataset.R     # creates data/product_reviews_50k.csv
Rscript scripts/02_sentiment_analysis.R   # runs analysis, writes plots/, console/, results/
```
Required R packages: ggplot2, corrplot, caret, randomForest, pROC.

## Note on the data
The dataset is simulated by `01_generate_dataset.R` (seed 2026), so reported accuracy reflects the
simulated relationships, not real customer behaviour.
