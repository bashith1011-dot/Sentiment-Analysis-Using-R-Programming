# ------------------------------------------------------------------
# 01_generate_dataset.R
# Builds a 50,000-row e-commerce product-review dataset for sentiment
# analysis. Every row has review text plus reviewer / product / order
# metadata. Data is simulated with a fixed seed so results reproduce.
# ------------------------------------------------------------------
set.seed(2026)
n <- 50000

# ---- 1. Product / order / reviewer metadata -----------------------
categories <- c("Electronics","Fashion","Home_Kitchen","Beauty","Books","Sports","Toys","Grocery")
cat_q      <- c(0.20, 0.00, 0.10, -0.10, 0.30, 0.15, 0.25, -0.20)   # category quality effect
brands     <- paste0("Brand_", LETTERS[1:20])
brand_q    <- rnorm(20, 0, 0.5)

product_category <- sample(categories, n, replace = TRUE,
                           prob = c(.22,.18,.15,.10,.08,.10,.09,.08))
brand            <- sample(brands, n, replace = TRUE)
price_usd        <- round(rlnorm(n, meanlog = 3.6, sdlog = 0.9), 2)
discount_pct     <- ifelse(runif(n) < 0.35, 0, sample(5:70, n, replace = TRUE))
delivery_days    <- pmin(20, 1 + rpois(n, 4))
verified_purchase<- rbinom(n, 1, 0.80)
has_image        <- rbinom(n, 1, 0.15)
reviewer_age     <- pmin(75, pmax(18, round(rnorm(n, 34, 11))))
reviewer_review_count <- rgeom(n, 0.12) + 1
device_type      <- sample(c("Mobile","Desktop","Tablet"), n, TRUE, prob = c(.62,.30,.08))
region           <- sample(c("North","South","East","West","Central","Northeast"), n, TRUE)
review_date      <- as.Date("2019-01-01") + sample(0:2556, n, replace = TRUE)
review_year      <- as.integer(format(review_date, "%Y"))
review_month     <- as.integer(format(review_date, "%m"))
is_weekend       <- as.integer(format(review_date, "%u") %in% c("6","7"))

# ---- 2. Latent sentiment -> score, label ---------------------------
z <- cat_q[match(product_category, categories)] +
     brand_q[match(brand, brands)] -
     0.12 * (delivery_days - 5) +
     0.012 * discount_pct -
     0.10 * as.numeric(scale(log(price_usd))) +
     rnorm(n, 0, 0.8)
sentiment_score <- round(tanh(z), 4)                       # continuous, -1 .. +1
q <- quantile(sentiment_score, c(0.25, 0.45))
sentiment_label <- ifelse(sentiment_score <= q[1], "Negative",
                   ifelse(sentiment_score >= q[2], "Positive", "Neutral"))
star_rating     <- pmin(5, pmax(1, round(3 + 2 * sentiment_score + rnorm(n, 0, 0.8))))
return_flag     <- rbinom(n, 1, plogis(-2.5 - 2 * sentiment_score))
helpful_votes   <- rpois(n, exp(0.8 + 0.3 * abs(sentiment_score) + 0.5 * has_image))
total_votes     <- helpful_votes + rpois(n, 2)

# ---- 3. Review text -------------------------------------------------
pos_adj <- c("excellent","amazing","fantastic","great","wonderful","superb","impressive",
             "perfect","outstanding","reliable","smooth","beautiful","comfortable","good","solid")
neg_adj <- c("terrible","awful","poor","disappointing","horrible","cheap","flimsy",
             "useless","defective","broken","bad","noisy","overpriced","unreliable","faulty")
neu_adj <- c("okay","average","decent","fine","acceptable","ordinary","standard",
             "passable","adequate","normal","typical","moderate")
aspects <- c("quality","packaging","delivery","price","design","performance",
             "durability","finish","size","sound","fit","battery")
closers <- list(
  pos = c("Would buy again!","Highly recommend this product!","Very happy with it.","Totally worth the money."),
  neg = c("Would not recommend.","Total waste of money!","I want a refund.","Very disappointed, I regret buying it."),
  neu = c("It does the job.","Nothing special.","Might buy again.","Fair for the price."))

tp <- sentiment_score + rnorm(n, 0, 0.30)                  # "text polarity" (noisy view of score)
w_pos <- exp(2.5 * tp); w_neg <- exp(-2.5 * tp); w_neu <- rep(exp(0.2), n)
tot <- w_pos + w_neg + w_neu
p_pos <- w_pos / tot; p_neg <- w_neg / tot

make_sentence <- function() {
  u   <- runif(n)
  pol <- ifelse(u < p_pos, "pos", ifelse(u < p_pos + p_neg, "neg", "neu"))
  neg_flag <- (pol != "neu") & (runif(n) < 0.09)           # negation: "not good"
  adj <- character(n)
  pool_of <- function(p, flip) {
    if (flip) { if (p == "pos") neg_adj else pos_adj } else { if (p == "pos") pos_adj else if (p == "neg") neg_adj else neu_adj }
  }
  for (p in c("pos","neg","neu")) {
    for (f in c(FALSE, TRUE)) {
      idx <- which(pol == p & neg_flag == f)
      if (length(idx)) adj[idx] <- sample(pool_of(p, f), length(idx), TRUE)
    }
  }
  adj <- ifelse(neg_flag, paste("not", adj), adj)
  adj <- ifelse(runif(n) < 0.05, toupper(adj), adj)        # occasional SHOUTING
  asp <- sample(aspects, n, TRUE)
  tpl <- sample(1:3, n, TRUE)
  s <- ifelse(tpl == 1, paste0("The ", asp, " is ", adj, "."),
       ifelse(tpl == 2, paste0("I found the ", asp, " ", adj, "."),
                        paste0("Honestly the ", asp, " was ", adj, ".")))
  bang <- runif(n) < ifelse(abs(tp) > 0.5, 0.35, 0.10)
  ifelse(bang, sub("\\.$", "!", s), s)
}
k <- sample(2:4, n, TRUE, prob = c(.45, .35, .20))
s1 <- make_sentence(); s2 <- make_sentence(); s3 <- make_sentence(); s4 <- make_sentence()
s3[k < 3] <- ""; s4[k < 4] <- ""
cu  <- runif(n)
cl_pol <- ifelse(cu < p_pos, "pos", ifelse(cu < p_pos + p_neg, "neg", "neu"))
closer <- vapply(seq_len(n), function(i) sample(closers[[cl_pol[i]]], 1), character(1))
closer[runif(n) < 0.15] <- ""
quest <- ifelse(runif(n) < 0.04 & tp < 0, "Is this normal?", "")
review_text <- trimws(gsub("\\s+", " ", paste(s1, s2, s3, s4, closer, quest)))

reviews <- data.frame(
  review_id = sprintf("R%06d", 1:n), review_text, product_category, brand, price_usd,
  discount_pct, delivery_days, verified_purchase, has_image, reviewer_age,
  reviewer_review_count, device_type, region, review_date, review_year, review_month,
  is_weekend, star_rating, return_flag, helpful_votes, total_votes,
  sentiment_score, sentiment_label, stringsAsFactors = FALSE)

# ---- 4. Inject realistic data-quality problems ----------------------
reviews$reviewer_age[sample(n, 1000)]           <- NA   # 2.0 %
reviews$delivery_days[sample(n, 750)]           <- NA   # 1.5 %
reviews$reviewer_review_count[sample(n, 500)]   <- NA   # 1.0 %

write.csv(reviews, "data/product_reviews_50k.csv", row.names = FALSE)
cat("Rows:", nrow(reviews), " Columns:", ncol(reviews), "\n")
