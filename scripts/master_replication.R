# ==============================================================================
# PREDICTIVE VOLATILITY OF MACHINE LEARNING IN MICRO-SAMPLES: R SCRIPT

# 0. SETUP, DIRECTORIES & DEPENDENCY MANAGEMENT
# ==============================================================================
mainDir <- getwd()
dir.create(file.path(mainDir, "1. Findings"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(mainDir, "2. Simulation"), showWarnings = FALSE, recursive = TRUE)

# Required packages:
# install.packages(c(
#   "glmnet", "brms", "spdep", "BART", "xgboost", "shapviz",
#   "loo", "car", "ggplot2", "randomForest", "MASS",
#   "kernlab", "dplyr", "sf", "rnaturalearth", "tidyr",
#   "ggcorrplot", "bayesplot", "posterior"
# ))

library(glmnet)
library(brms)
library(spdep)
library(BART)
library(xgboost)
library(shapviz)
library(loo)
library(car)
library(ggplot2)
library(randomForest)
library(MASS)
library(kernlab)
library(dplyr)
library(sf)
library(rnaturalearth)
library(tidyr)
library(ggcorrplot)
library(bayesplot)
library(posterior)

options(mc.cores = max(1, parallel::detectCores() - 1))

# 0.a HELPER FUNCTIONS
# ==============================================================================
rmse <- function(y, yhat) sqrt(mean((y - yhat)^2, na.rm = TRUE))
mae  <- function(y, yhat) mean(abs(y - yhat), na.rm = TRUE)

safe_scale <- function(x_train, x_test = NULL) {
  mu <- colMeans(x_train)
  sds <- apply(x_train, 2, sd)
  sds[sds == 0] <- 1
  
  x_train_sc <- scale(x_train, center = mu, scale = sds)
  x_test_sc  <- NULL
  if (!is.null(x_test)) {
    x_test_sc <- scale(x_test, center = mu, scale = sds)
  }
  
  list(
    train  = as.matrix(x_train_sc),
    test   = if (!is.null(x_test_sc)) as.matrix(x_test_sc) else NULL,
    center = mu,
    scale  = sds
  )
}

backtransform_glmnet <- function(cvfit, scaler, s = "lambda.min") {
  beta_std <- as.numeric(coef(cvfit, s = s))
  intercept_std <- beta_std[1]
  beta_std <- beta_std[-1]
  
  beta_orig <- beta_std / scaler$scale
  intercept_orig <- intercept_std - sum((scaler$center / scaler$scale) * beta_std)
  
  names(beta_orig) <- names(scaler$center)
  list(intercept = intercept_orig, beta = beta_orig)
}

extract_estimate <- function(pred_obj) {
  if (is.matrix(pred_obj) && "Estimate" %in% colnames(pred_obj)) {
    return(as.numeric(pred_obj[, "Estimate"]))
  }
  if (is.data.frame(pred_obj) && "Estimate" %in% names(pred_obj)) {
    return(as.numeric(pred_obj$Estimate))
  }
  return(as.numeric(pred_obj))
}

squeeze01 <- function(x, eps = 1e-6) {
  pmin(pmax(x, eps), 1 - eps)
}

top1_fractional <- function(loss_mat) {
  out <- rep(0, ncol(loss_mat))
  names(out) <- colnames(loss_mat)
  for (i in seq_len(nrow(loss_mat))) {
    row_i <- loss_mat[i, ]
    min_i <- min(row_i, na.rm = TRUE)
    winners <- which(abs(row_i - min_i) < .Machine$double.eps^0.5)
    out[winners] <- out[winners] + 1 / length(winners)
  }
  out / nrow(loss_mat)
}

theme_pub <- function() {
  theme_classic(base_size = 14) %+replace% theme(
    plot.title = element_text(face = "bold", size = rel(1.1), margin = ggplot2::margin(b = 6)),
    legend.position = "bottom",
    axis.title = element_text(face = "bold"),
    panel.grid.major.x = element_line(color = "grey93", linetype = "dotted"),
    panel.grid.major.y = element_line(color = "grey93", linetype = "dotted")
  )
}

# 1. EMPIRICAL APPLICATION DATA PREPARATION
# ==============================================================================
cat("--- Step 1: Loading empirical application data ---\n")

df <- data.frame(
  Province = paste0("Provinsi", sprintf("%02d", 1:34)),
  Y  = c(14.75, 8.33, 6.04, 6.84, 7.70, 11.95, 14.34, 11.44, 4.61, 6.03, 4.61,
         7.98, 10.98, 11.49, 10.49, 6.24, 4.53, 13.82, 20.23, 6.81, 5.22, 4.61,
         6.44, 6.86, 7.34, 12.30, 8.66, 11.27, 15.51, 11.02, 16.23, 6.37, 21.43, 26.80),
  X1 = c(9.44, 9.71, 9.18, 9.22, 8.68, 8.37, 8.91, 8.18, 8.11, 10.37, 11.31,
         8.78, 7.93, 9.75, 8.03, 9.13, 9.39, 7.61, 7.70, 7.59, 8.65, 8.46,
         9.92, 9.27, 9.68, 8.89, 8.63, 9.25, 8.02, 8.08, 10.19, 9.24, 7.84, 7.02),
  X2 = c(70.18, 69.61, 69.90, 71.95, 71.50, 70.32, 69.69, 70.99, 70.98, 70.50, 73.32,
         73.52, 74.57, 75.08, 71.74, 70.39, 72.60, 67.07, 67.47, 71.02, 70.04, 69.13,
         74.62, 72.67, 72.08, 68.93, 70.97, 71.37, 68.51, 65.63, 66.45, 68.79, 66.46, 66.23),
  X3 = c(6.83, 5.96, 4.31, 5.82, 6.56, 4.70, 6.06, 7.33, 6.03, 3.96, 1.56,
         6.62, 6.61, 7.48, 6.20, 5.34, 2.75, 9.98, 7.22, 6.44, 5.18, 5.40,
         5.34, 8.90, 4.93, 7.93, 7.15, 10.21, 8.89, 8.50, 4.11, 5.19, 4.14, 2.16),
  X4 = c(0.291, 0.326, 0.292, 0.323, 0.335, 0.330, 0.315, 0.313, 0.255, 0.325, 0.412,
         0.412, 0.366, 0.459, 0.365, 0.377, 0.362, 0.374, 0.340, 0.311, 0.309, 0.309,
         0.317, 0.270, 0.359, 0.305, 0.365, 0.366, 0.423, 0.371, 0.306, 0.309, 0.384, 0.393),
  X5 = c(77.48, 82.30, 69.27, 84.06, 79.54, 78.62, 79.58, 83.65, 91.63, 87.74, 92.79,
         74.02, 84.37, 96.21, 81.13, 85.12, 95.94, 83.12, 73.70, 77.41, 74.33, 82.55,
         90.33, 82.22, 84.05, 75.01, 92.24, 87.07, 79.82, 78.88, 76.47, 79.39, 73.52, 40.34),
  X6 = c(89.70, 92.13, 85.23, 90.07, 79.19, 86.35, 73.07, 81.60, 80.96, 91.82, 97.93,
         93.04, 93.32, 96.50, 95.05, 92.71, 98.42, 96.50, 86.76, 80.43, 77.01, 76.18,
         87.14, 89.96, 94.15, 86.74, 91.96, 94.64, 96.16, 78.98, 92.10, 88.10, 81.57, 65.39),
  X7 = c(99.69, 98.70, 98.82, 95.20, 98.39, 96.55, 99.21, 98.66, 99.53, 97.96, 99.73,
         99.77, 99.96, 99.95, 99.59, 99.56, 99.92, 99.68, 85.58, 89.99, 85.36, 98.78,
         95.56, 94.16, 99.22, 95.26, 97.36, 97.02, 98.43, 92.85, 93.15, 90.98, 83.49, 43.04),
  X8 = c(6.17, 6.16, 6.28, 4.37, 4.59, 4.63, 3.59, 4.52, 4.77, 8.23, 7.18,
         8.31, 5.57, 4.06, 5.49, 8.09, 4.80, 2.89, 3.54, 5.11, 4.26, 4.74,
         5.71, 4.33, 6.61, 3.00, 4.51, 3.36, 2.58, 2.34, 6.88, 3.98, 5.37, 2.83),
  X9 = c(64.89, 74.44, 73.12, 77.63, 72.33, 69.12, 70.57, 72.00, 75.70, 91.14, 92.36,
         79.42, 76.95, 86.98, 73.63, 79.01, 80.77, 64.98, 57.78, 70.39, 73.94, 76.50,
         87.61, 84.80, 72.77, 63.71, 73.16, 71.84, 67.02, 64.92, 65.58, 59.45, 66.05, 29.82)
)

X_mat <- as.matrix(df[, paste0("X", 1:9)])
colnames(X_mat) <- paste0("X", 1:9)
Y <- df$Y

# Poverty rate mapped to (0,1); values are already interior, but squeeze defensively
df$Y_beta <- squeeze01(df$Y / 100)
df$spatial_id <- factor(seq_len(nrow(df)))

# Table 4: Descriptive statistics + VIF
cat("--- Step 1.b: Exporting descriptive statistics and VIFs ---\n")
vif_fit <- lm(Y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9, data = df)
vif_vals <- car::vif(vif_fit)

table4 <- data.frame(
  Variable = c("Y", paste0("X", 1:9)),
  Mean     = sapply(df[, c("Y", paste0("X", 1:9))], mean),
  SD       = sapply(df[, c("Y", paste0("X", 1:9))], sd),
  Min      = sapply(df[, c("Y", paste0("X", 1:9))], min),
  Median   = sapply(df[, c("Y", paste0("X", 1:9))], median),
  Max      = sapply(df[, c("Y", paste0("X", 1:9))], max),
  VIF      = c(NA, as.numeric(vif_vals))
)
write.csv(table4, file.path(mainDir, "1. Findings", "Table4_Descriptive_Statistics_VIF.csv"), row.names = FALSE)

# Absolute Provincial Coordinates Array Definition Block
real_coords <- data.frame(
  Lon = c(95.3167, 98.6667, 100.3667, 101.4500, 103.6167, 104.7500, 102.2667, 105.2580,
          106.1000, 104.4667, 106.8167, 107.6000, 110.4167, 110.3667, 112.7500, 106.1500,
          115.2167, 116.1167, 123.5833, 109.3333, 113.9167, 114.5833, 117.1500, 117.3667,
          124.8333, 119.8667, 119.4333, 122.5833, 123.0667, 118.8833, 128.1667, 127.5667,
          134.0833, 140.7167),
  Lat = c(5.5500, 3.5833, -0.9500, 0.5333, -1.6000, -2.9833, -3.8000, -5.4250,
          -2.1167, 0.9167, -6.2000, -6.9167, -6.9667, -7.8000, -7.2500, -6.1167,
          -8.6500, -8.5833, -10.1667, -0.0167, -2.2000, -3.3167, -0.5000, 2.8333,
          1.4833, -0.9000, -5.1500, -3.9667, 0.5333, -2.6667, -3.7000, 0.7333,
          -0.8667, -2.5333)
)

# Spatial graph (retained conservatively; manuscript does not fully specify k)
coords_matrix <- as.matrix(real_coords)
knn_nb <- knn2nb(knearneigh(coords_matrix, k = 2))
sym_nb <- make.sym.nb(knn_nb)
W_list <- nb2listw(sym_nb, style = "W", zero.policy = FALSE)
A <- nb2mat(sym_nb, style = "B", zero.policy = FALSE)
rownames(A) <- colnames(A) <- as.character(df$spatial_id)

# 2. FREQUENTIST COEFFICIENT ESTIMATION ON FULL DATA
# ==============================================================================
cat("--- Step 2: Fitting frequentist models on full sample for coefficient tables ---\n")

# OLS on raw scale
m1_ols_full <- lm(Y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9, data = df)
ols_orig <- coef(m1_ols_full)[paste0("X", 1:9)]

# Penalized models estimated on standardized X, then back-transformed
sc_full <- safe_scale(X_mat)
cv_ridge_full <- cv.glmnet(sc_full$train, Y, alpha = 0,   nfolds = nrow(df), standardize = FALSE)
cv_lasso_full <- cv.glmnet(sc_full$train, Y, alpha = 1,   nfolds = nrow(df), standardize = FALSE)
cv_enet_full  <- cv.glmnet(sc_full$train, Y, alpha = 0.5, nfolds = nrow(df), standardize = FALSE)

ridge_bt <- backtransform_glmnet(cv_ridge_full, sc_full)
lasso_bt <- backtransform_glmnet(cv_lasso_full, sc_full)
enet_bt  <- backtransform_glmnet(cv_enet_full,  sc_full)

freq_coefs <- data.frame(
  Predictor   = paste0("X", 1:9),
  OLS         = round(as.numeric(ols_orig), 3),
  Ridge       = round(as.numeric(ridge_bt$beta[paste0("X", 1:9)]), 3),
  LASSO       = round(as.numeric(lasso_bt$beta[paste0("X", 1:9)]), 3),
  Elastic_Net = round(as.numeric(enet_bt$beta[paste0("X", 1:9)]), 3)
)

write.csv(freq_coefs, file.path(mainDir, "1. Findings", "Table5_Frequentist_Coefficients.csv"), row.names = FALSE)

# 3. BAYESIAN PARAMETRIC ESTIMATION SUITE
# ==============================================================================
cat("--- Step 3: Compiling Bayesian models ---\n")

brms_form <- bf(Y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9)

# Common arguments
brms_common <- list(
  chains = 4,
  iter = 4000,
  warmup = 2000,
  cores = min(4, parallel::detectCores()),
  seed = 123,
  silent = 2,
  refresh = 0,
  save_pars = save_pars(all = TRUE)
)

# Priors aligned to manuscript where sufficiently specified
prior_m5 <- c(
  set_prior("normal(0,10)", class = "Intercept"),
  set_prior("normal(0,5)", class = "b"),
  set_prior("inv_gamma(3,2)", class = "sigma")
)

# Exact ridge hyper-scale is not explicitly coded in the uploaded manuscript;
# use a conservative Gaussian shrinkage prior instead of the incorrect Student-t.
prior_m6 <- c(
  set_prior("normal(0,10)", class = "Intercept"),
  set_prior("normal(0,2.5)", class = "b"),
  set_prior("inv_gamma(3,2)", class = "sigma")
)

prior_m7 <- c(
  set_prior("normal(0,10)", class = "Intercept"),
  set_prior("double_exponential(0,1)", class = "b"),
  set_prior("inv_gamma(3,2)", class = "sigma")
)

prior_m8 <- c(
  set_prior("normal(0,10)", class = "Intercept"),
  set_prior("horseshoe(1)", class = "b"),
  set_prior("inv_gamma(3,2)", class = "sigma")
)

prior_m10 <- c(
  set_prior("normal(0,10)", class = "Intercept"),
  set_prior("normal(0,5)", class = "b")
)

prior_m11 <- c(
  set_prior("normal(0,10)", class = "Intercept"),
  set_prior("normal(0,5)", class = "b"),
  set_prior("inv_gamma(3,2)", class = "sigma")
)

m5_brms <- do.call(
  brm,
  c(
    list(formula = brms_form, data = df, family = gaussian(), prior = prior_m5),
    brms_common
  )
)

m6_bridge <- do.call(
  brm,
  c(
    list(formula = brms_form, data = df, family = gaussian(), prior = prior_m6),
    brms_common
  )
)

m7_blasso <- do.call(
  brm,
  c(
    list(formula = brms_form, data = df, family = gaussian(), prior = prior_m7),
    brms_common
  )
)

m8_horseshoe <- do.call(
  brm,
  c(
    list(
      formula = brms_form,
      data = df,
      family = gaussian(),
      prior = prior_m8,
      control = list(adapt_delta = 0.995, max_treedepth = 15)
    ),
    brms_common
  )
)

m10_beta <- do.call(
  brm,
  c(
    list(
      formula = bf(Y_beta ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9),
      data = df,
      family = Beta(),
      prior = prior_m10
    ),
    brms_common
  )
)

# BYM2 retained as a robustness check; adjacency tuning not changed due limited detail.
m11_icar <- brm(
  formula = Y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 +
    car(A, gr = spatial_id, type = "bym2"),
  data = df,
  data2 = list(A = A),
  family = gaussian(),
  prior = prior_m11,
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = min(4, parallel::detectCores()),
  seed = 123,
  silent = 2,
  refresh = 0,
  save_pars = save_pars(all = TRUE)
)

# 3.a BAYESIAN POSTERIOR SUMMARY TABLE
# ==============================================================================
cat("--- Step 3.a: Exporting Bayesian posterior summaries ---\n")

extract_fixed_summary <- function(brms_model, model_name) {
  fx <- as.data.frame(summary(brms_model)$fixed)
  fx$Predictor <- rownames(fx)
  rownames(fx) <- NULL
  
  # standardize column names across brms versions
  lower_col <- intersect(c("l-95% CI", "Q2.5"), names(fx))[1]
  upper_col <- intersect(c("u-95% CI", "Q97.5"), names(fx))[1]
  se_col    <- intersect(c("Est.Error", "Est.Error"), names(fx))[1]
  
  out <- data.frame(
    Model     = model_name,
    Predictor = fx$Predictor,
    Estimate  = fx$Estimate,
    Est.Error = fx[[se_col]],
    Lower_95  = fx[[lower_col]],
    Upper_95  = fx[[upper_col]]
  )
  out
}

table6_bayes <- bind_rows(
  extract_fixed_summary(m5_brms, "M5_Bayes_LM_Gaussian"),
  extract_fixed_summary(m6_bridge, "M6_Bayesian_Ridge"),
  extract_fixed_summary(m7_blasso, "M7_Bayesian_LASSO"),
  extract_fixed_summary(m8_horseshoe, "M8_Horseshoe"),
  extract_fixed_summary(m10_beta, "M10_Beta_Regression"),
  extract_fixed_summary(m11_icar, "M11_Spatial_BYM2")
)

write.csv(table6_bayes, file.path(mainDir, "1. Findings", "Table6_Bayesian_Posterior_Summaries.csv"), row.names = FALSE)

# 3.b MCMC CONVERGENCE DIAGNOSTICS
# ==============================================================================
cat("--- Step 3.b: Extracting MCMC convergence diagnostics ---\n")

extract_convergence_bounds <- function(brms_model, model_name, model_id) {
  dd <- posterior::as_draws_df(brms_model)
  ss <- posterior::summarise_draws(dd, posterior::rhat, posterior::ess_bulk)
  
  rhats <- ss$rhat[is.finite(ss$rhat)]
  neffs <- ss$ess_bulk[is.finite(ss$ess_bulk)]
  
  data.frame(
    ID = model_id,
    Model = model_name,
    Min_Rhat = round(min(rhats), 3),
    Max_Rhat = round(max(rhats), 3),
    Min_ESS  = round(min(neffs), 0),
    Max_ESS  = round(max(neffs), 0)
  )
}

tab_convergence <- bind_rows(
  extract_convergence_bounds(m5_brms,  "Bayes LM (Gaussian)", "M5"),
  extract_convergence_bounds(m6_bridge,"Bayesian ridge",      "M6"),
  extract_convergence_bounds(m7_blasso,"Bayesian lasso",      "M7"),
  extract_convergence_bounds(m8_horseshoe, "Horseshoe",       "M8"),
  extract_convergence_bounds(m10_beta, "Beta regression",     "M10"),
  extract_convergence_bounds(m11_icar, "Spatial ICAR/BYM2",   "M11")
)

write.csv(tab_convergence, file.path(mainDir, "1. Findings", "TableS8_Convergence_Diagnostics.csv"), row.names = FALSE)

# 4. MANUAL EMPIRICAL LOO FOR FREQUENTIST + ML MODELS
# ==============================================================================
cat("--- Step 4: Running manual LOO with fold-wise scaling and nested tuning ---\n")

set.seed(99)
n_emp <- nrow(df)

preds_ols   <- numeric(n_emp)
preds_ridge <- numeric(n_emp)
preds_lasso <- numeric(n_emp)
preds_enet  <- numeric(n_emp)
preds_bart  <- numeric(n_emp)
preds_gp    <- numeric(n_emp)
preds_rf    <- numeric(n_emp)
preds_xgb   <- numeric(n_emp)

xgb_params <- list(
  eta = 0.05,
  max_depth = 3,
  colsample_bytree = 0.8,
  objective = "reg:squarederror"
)

for (i in seq_len(n_emp)) {
  tr <- setdiff(seq_len(n_emp), i)
  
  x_tr <- X_mat[tr, , drop = FALSE]
  x_te <- X_mat[i,  , drop = FALSE]
  y_tr <- Y[tr]
  
  # Fold-wise standardization, as stated in the paper
  sc_i <- safe_scale(x_tr, x_te)
  
  # OLS on standardized predictors within fold
  dat_tr_sc <- data.frame(y = y_tr, sc_i$train)
  colnames(dat_tr_sc) <- c("y", paste0("X", 1:9))
  dat_te_sc <- data.frame(sc_i$test)
  colnames(dat_te_sc) <- paste0("X", 1:9)
  
  ols_i <- lm(y ~ ., data = dat_tr_sc)
  preds_ols[i] <- as.numeric(predict(ols_i, newdata = dat_te_sc))
  
  # Nested tuning inside training split
  inner_folds <- length(tr)  # exact inner LOO for nested penalized tuning
  
  ridge_i <- cv.glmnet(sc_i$train, y_tr, alpha = 0, standardize = FALSE, nfolds = inner_folds)
  lasso_i <- cv.glmnet(sc_i$train, y_tr, alpha = 1, standardize = FALSE, nfolds = inner_folds)
  enet_i  <- cv.glmnet(sc_i$train, y_tr, alpha = 0.5, standardize = FALSE, nfolds = inner_folds)
  
  preds_ridge[i] <- as.numeric(predict(ridge_i, newx = sc_i$test, s = "lambda.min"))
  preds_lasso[i] <- as.numeric(predict(lasso_i, newx = sc_i$test, s = "lambda.min"))
  preds_enet[i]  <- as.numeric(predict(enet_i,  newx = sc_i$test, s = "lambda.min"))
  
  # BART
  capture.output(
    bart_i <- wbart(
      x.train = x_tr, y.train = y_tr, x.test = x_te,
      ntree = 50, ndpost = 2000, sparse = FALSE, printevery = 100000
    )
  )
  preds_bart[i] <- mean(bart_i$yhat.test)
  
  # Gaussian process on fold-wise standardized predictors
  gp_i <- gausspr(sc_i$train, y_tr, kernel = "rbfdot", variance.model = FALSE)
  preds_gp[i] <- as.numeric(predict(gp_i, sc_i$test))
  
  # Random forest
  rf_i <- randomForest(x_tr, y_tr, ntree = 500, mtry = 3)
  preds_rf[i] <- as.numeric(predict(rf_i, x_te))
  
  # XGBoost
  xgb_i <- xgb.train(
    params = xgb_params,
    data = xgb.DMatrix(data = x_tr, label = y_tr),
    nrounds = 250,
    verbose = 0
  )
  preds_xgb[i] <- as.numeric(predict(xgb_i, xgb.DMatrix(data = x_te)))
}

# Full-sample BART and RF for importance outputs
cat("--- Step 4.b: Fitting full-sample BART/RF/XGB objects for importance/SHAP ---\n")
set.seed(123)
m12_bart <- wbart(
  x.train = X_mat, y.train = Y,
  ntree = 50, ndpost = 10000,
  sparse = FALSE, printevery = 100000
)
bart_imp <- (m12_bart$varcount.mean / sum(m12_bart$varcount.mean)) * 100

m14_rf <- randomForest(X_mat, Y, ntree = 500, mtry = 3, importance = TRUE)
rf_imp <- (m14_rf$importance[, "%IncMSE"] / sum(m14_rf$importance[, "%IncMSE"])) * 100

fit_xgb_full <- xgb.train(
  params = xgb_params,
  data = xgb.DMatrix(data = X_mat, label = Y),
  nrounds = 250,
  verbose = 0
)

imp_df <- data.frame(
  Predictor = rep(colnames(X_mat), 2),
  Algorithm = rep(c("BART", "RF"), each = ncol(X_mat)),
  Importance = c(bart_imp, rf_imp)
)
write.csv(imp_df, file.path(mainDir, "1. Findings", "TableS7_Variable_Importance.csv"), row.names = FALSE)

# 5. BAYESIAN EMPIRICAL PERFORMANCE
# ==============================================================================
cat("--- Step 5: Computing Bayesian empirical predictive metrics ---\n")

# Note:
# The paper states PSIS-LOO with exact refitting when Pareto-k > 0.7.
# The code below records Pareto-k diagnostics and uses loo_predict() output
# for RMSE/MAE. If any model shows large Pareto-k, run a targeted reloo step
# before final manuscript numbers are frozen.

get_bayes_metrics <- function(mod, y_true, scale_back = 1) {
  loo_obj <- suppressWarnings(loo(mod))
  preds <- suppressWarnings(loo_predict(mod))
  est <- extract_estimate(preds) * scale_back
  
  max_k <- NA_real_
  if (!is.null(loo_obj$diagnostics$pareto_k)) {
    max_k <- max(loo_obj$diagnostics$pareto_k, na.rm = TRUE)
  }
  
  c(
    RMSE = rmse(y_true, est),
    MAE = mae(y_true, est),
    Max_Pareto_k = max_k
  )
}

mets_m5  <- get_bayes_metrics(m5_brms, Y, scale_back = 1)
mets_m6  <- get_bayes_metrics(m6_bridge, Y, scale_back = 1)
mets_m7  <- get_bayes_metrics(m7_blasso, Y, scale_back = 1)
mets_m8  <- get_bayes_metrics(m8_horseshoe, Y, scale_back = 1)
mets_m10 <- get_bayes_metrics(m10_beta, Y, scale_back = 100)
mets_m11 <- get_bayes_metrics(m11_icar, Y, scale_back = 1)

pareto_diag <- data.frame(
  ID = c("M5", "M6", "M7", "M8", "M10", "M11"),
  Model = c("Bayes LM (Gaussian)", "Bayesian ridge", "Bayesian lasso",
            "Horseshoe", "Beta regression", "Spatial ICAR/BYM2"),
  Max_Pareto_k = c(mets_m5["Max_Pareto_k"], mets_m6["Max_Pareto_k"], mets_m7["Max_Pareto_k"],
                   mets_m8["Max_Pareto_k"], mets_m10["Max_Pareto_k"], mets_m11["Max_Pareto_k"])
)
write.csv(pareto_diag, file.path(mainDir, "1. Findings", "TableS8b_Pareto_k_Diagnostics.csv"), row.names = FALSE)

# 5.b EMPIRICAL COMPARISON TABLE (Table 2)
# ==============================================================================
cat("--- Step 5.b: Writing empirical comparison table ---\n")

tab_empirical <- data.frame(
  ID = c("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M10", "M11", "M12", "M13", "M14", "M15"),
  Model = c(
    "OLS", "Ridge", "LASSO", "Elastic net",
    "Bayes LM (Gaussian)", "Bayesian ridge", "Bayesian LASSO", "Horseshoe",
    "Beta regression", "Spatial ICAR/BYM2",
    "BART", "Gaussian process", "Random forest", "XGBoost"
  ),
  LOO_RMSE = c(
    rmse(Y, preds_ols),
    rmse(Y, preds_ridge),
    rmse(Y, preds_lasso),
    rmse(Y, preds_enet),
    unname(mets_m5["RMSE"]),
    unname(mets_m6["RMSE"]),
    unname(mets_m7["RMSE"]),
    unname(mets_m8["RMSE"]),
    unname(mets_m10["RMSE"]),
    unname(mets_m11["RMSE"]),
    rmse(Y, preds_bart),
    rmse(Y, preds_gp),
    rmse(Y, preds_rf),
    rmse(Y, preds_xgb)
  ),
  LOO_MAE = c(
    mae(Y, preds_ols),
    mae(Y, preds_ridge),
    mae(Y, preds_lasso),
    mae(Y, preds_enet),
    unname(mets_m5["MAE"]),
    unname(mets_m6["MAE"]),
    unname(mets_m7["MAE"]),
    unname(mets_m8["MAE"]),
    unname(mets_m10["MAE"]),
    unname(mets_m11["MAE"]),
    mae(Y, preds_bart),
    mae(Y, preds_gp),
    mae(Y, preds_rf),
    mae(Y, preds_xgb)
  )
)

write.csv(tab_empirical, file.path(mainDir, "1. Findings", "Table2_Empirical_Comparison.csv"), row.names = FALSE)

# Optional ranked export for quick manuscript checking
tab_empirical_ranked <- tab_empirical %>% arrange(LOO_RMSE, LOO_MAE)
write.csv(tab_empirical_ranked, file.path(mainDir, "1. Findings", "Table2_Empirical_Comparison_Ranked.csv"), row.names = FALSE)

# 6. PRIOR SENSITIVITY PIPELINE
# ==============================================================================
cat("--- Step 6: Running prior sensitivity analysis ---\n")

# The manuscript discusses prior sensitivity for the ICT-skills coefficient (X9).
# We vary the prior variance scale and track the posterior estimate / interval for X9.
prior_var_scales <- c(1, 10, 25, 100, 1000)

sens_results <- data.frame(
  Prior_Variance = prior_var_scales,
  Prior_SD = sqrt(prior_var_scales),
  Mean = NA_real_,
  Lower_95 = NA_real_,
  Upper_95 = NA_real_
)

for (i in seq_along(prior_var_scales)) {
  sd_i <- sqrt(prior_var_scales[i])
  
  tmp_mod <- brm(
    formula = brms_form,
    data = df,
    family = gaussian(),
    prior = c(
      set_prior("normal(0,10)", class = "Intercept"),
      set_prior(paste0("normal(0,", sd_i, ")"), class = "b"),
      set_prior("inv_gamma(3,2)", class = "sigma")
    ),
    chains = 2,
    iter = 2000,
    warmup = 1000,
    cores = min(2, parallel::detectCores()),
    seed = 100 + i,
    silent = 2,
    refresh = 0
  )
  
  tmp_fix <- as.data.frame(summary(tmp_mod)$fixed)
  tmp_fix$Predictor <- rownames(tmp_fix)
  rownames(tmp_fix) <- NULL
  
  lower_col <- intersect(c("l-95% CI", "Q2.5"), names(tmp_fix))[1]
  upper_col <- intersect(c("u-95% CI", "Q97.5"), names(tmp_fix))[1]
  
  x9_row <- tmp_fix[tmp_fix$Predictor == "X9", ]
  
  sens_results$Mean[i]     <- x9_row$Estimate
  sens_results$Lower_95[i] <- x9_row[[lower_col]]
  sens_results$Upper_95[i] <- x9_row[[upper_col]]
}

write.csv(sens_results, file.path(mainDir, "1. Findings", "TableS6_Prior_Sensitivity_X9.csv"), row.names = FALSE)

p_sens <- ggplot(sens_results, aes(x = factor(Prior_Variance), y = Mean)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "#e74c3c", alpha = 0.8) +
  geom_point(size = 3, color = "#2980b9") +
  geom_errorbar(aes(ymin = Lower_95, ymax = Upper_95), width = 0.2, color = "#2980b9", linewidth = 0.8) +
  theme_classic(base_size = 14) %+replace% theme(
    plot.title = element_text(face = "bold", size = rel(1.1), margin = ggplot2::margin(b = 6)),
    axis.title = element_text(face = "bold"),
    panel.grid.major.y = element_line(color = "grey93", linetype = "dotted")
  ) +
  labs(
    title = "Prior sensitivity for ICT skills coefficient (X9)",
    x = "Prior variance",
    y = "Posterior estimate for X9"
  )

# 7. FIGURE GENERATION
# ==============================================================================
cat("--- Step 7: Exporting manuscript and supplementary figures ---\n")

# Figure 2: Predictor correlation heatmap
p_corr <- ggcorrplot(
  cor(X_mat),
  type = "lower",
  lab = TRUE,
  colors = c("#e74c3c", "white", "#3498db"),
  ggtheme = theme_pub()
) +
  labs(title = "Predictor correlation matrix")

ggsave(
  file.path(mainDir, "1. Findings", "Figure2_Corr_Heatmap.pdf"),
  plot = p_corr,
  width = 8,
  height = 7
)

# Figure 3: Spatial poverty map
# Note: province-name joins can vary across Natural Earth versions.
# This join is kept conservative and close to your original script.
world_states <- ne_download(
  scale = 50,
  type = "states",
  category = "cultural",
  returnclass = "sf"
)

indo_states <- world_states %>% filter(admin == "Indonesia")

prov_map_df <- data.frame(
  name = c(
    "Aceh", "Sumatera Utara", "Sumatera Barat", "Riau", "Jambi",
    "Sumatera Selatan", "Bengkulu", "Lampung", "Kepulauan Bangka Belitung",
    "Kepulauan Riau", "Jakarta Raya", "Jawa Barat", "Jawa Tengah",
    "Yogyakarta", "Jawa Timur", "Banten", "Bali", "Nusa Tenggara Barat",
    "Nusa Tenggara Timur", "Kalimantan Barat", "Kalimantan Tengah",
    "Kalimantan Selatan", "Kalimantan Timur", "Kalimantan Utara",
    "Sulawesi Utara", "Sulawesi Tengah", "Sulawesi Selatan",
    "Sulawesi Tenggara", "Gorontalo", "Sulawesi Barat",
    "Maluku", "Maluku Utara", "Papua Barat", "Papua"
  ),
  Y = df$Y
)

if ("name" %in% names(indo_states)) {
  indo_merged <- indo_states %>% left_join(prov_map_df, by = "name")
} else if ("name_en" %in% names(indo_states)) {
  names(prov_map_df)[1] <- "name_en"
  indo_merged <- indo_states %>% left_join(prov_map_df, by = "name_en")
} else {
  indo_merged <- indo_states
  indo_merged$Y <- NA_real_
}

p_map <- ggplot(data = indo_merged) +
  geom_sf(aes(fill = Y), color = "white", linewidth = 0.1) +
  scale_fill_gradientn(
    colours = c("#2980b9", "#f1c40f", "#e74c3c"),
    name = "Poverty %",
    na.value = "grey85"
  ) +
  theme_pub() +
  theme(panel.background = element_rect(fill = "#eef4f7")) +
  labs(title = "Spatial distribution of provincial poverty in Indonesia")

ggsave(
  file.path(mainDir, "1. Findings", "Figure3_Poverty_Map.pdf"),
  plot = p_map,
  width = 12,
  height = 6
)

# Figure 4: Variable importance
p_imp <- ggplot(imp_df, aes(x = Importance, y = Predictor, fill = Algorithm)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = c("#2c3e50", "#e67e22")) +
  theme_pub() +
  labs(
    title = "Relative variable importance",
    x = "Importance (%)",
    y = "Predictor"
  )

ggsave(
  file.path(mainDir, "1. Findings", "Figure4_Variable_Importance.pdf"),
  plot = p_imp,
  width = 8,
  height = 5.5
)

# Figure 5: SHAP beeswarm for XGBoost
shp <- shapviz(fit_xgb_full, X_pred = X_mat)

p_shap <- sv_importance(shp, kind = "beeswarm") +
  theme_pub() +
  theme(legend.position = "right") +
  labs(title = "SHAP summary for XGBoost")

ggsave(
  file.path(mainDir, "1. Findings", "Figure5_SHAP_XGBoost.pdf"),
  plot = p_shap,
  width = 9,
  height = 6
)

# Figure 6: Empirical predictive performance comparison
tab_long <- tab_empirical %>%
  drop_na(LOO_RMSE, LOO_MAE) %>%
  pivot_longer(cols = c(LOO_RMSE, LOO_MAE), names_to = "Metric", values_to = "Value") %>%
  mutate(Metric = ifelse(Metric == "LOO_RMSE", "LOO RMSE", "LOO MAE"))

model_order <- tab_empirical %>% arrange(desc(LOO_RMSE)) %>% pull(Model)
tab_long$Model <- factor(tab_long$Model, levels = model_order)

p_emp <- ggplot(tab_long, aes(x = Value, y = Model, fill = Metric)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65) +
  scale_fill_manual(values = c("#2980b9", "#2c3e50")) +
  theme_pub() +
  labs(
    title = "Leave-one-out predictive performance",
    x = "Error metric value (lower is better)",
    y = "Model",
    fill = "Metric"
  ) +
  theme(legend.position = "bottom", axis.text.y = element_text(face = "plain"))

ggsave(
  file.path(mainDir, "1. Findings", "Figure6_Empirical_Performance.pdf"),
  plot = p_emp,
  width = 9,
  height = 6.5
)

# Supplementary figures
cat("--- Step 7.b: Exporting supplementary figures ---\n")

m8_summary <- as.data.frame(summary(m8_horseshoe)$fixed)
m8_summary$Predictor <- rownames(m8_summary)
rownames(m8_summary) <- NULL
m8_summary <- m8_summary[m8_summary$Predictor != "Intercept", ]

lower_col_m8 <- intersect(c("l-95% CI", "Q2.5"), names(m8_summary))[1]
upper_col_m8 <- intersect(c("u-95% CI", "Q97.5"), names(m8_summary))[1]

forest_df <- data.frame(
  Predictor = factor(m8_summary$Predictor, levels = rev(m8_summary$Predictor)),
  Mean = m8_summary$Estimate,
  Lower = m8_summary[[lower_col_m8]],
  Upper = m8_summary[[upper_col_m8]]
)

p_forest <- ggplot(forest_df, aes(x = Mean, y = Predictor)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "#e74c3c", alpha = 0.8) +
  geom_point(size = 3, color = "#2c3e50") +
  geom_errorbarh(aes(xmin = Lower, xmax = Upper), height = 0.2, color = "#2c3e50", linewidth = 0.8) +
  theme_pub() +
  labs(
    title = "Horseshoe posterior means and 95% credible intervals",
    x = "Posterior mean with 95% credible interval",
    y = "Predictor"
  )

ggsave(
  file.path(mainDir, "1. Findings", "FigureS2_Horseshoe_Forest.pdf"),
  plot = p_forest,
  width = 8,
  height = 5
)

p_ppc <- pp_check(m8_horseshoe, ndraws = 100) + theme_pub() +
  labs(title = "Posterior predictive check for horseshoe model")

ggsave(
  file.path(mainDir, "1. Findings", "FigureS3_PPC_Horseshoe.pdf"),
  plot = p_ppc,
  width = 8,
  height = 5
)

ggsave(
  file.path(mainDir, "1. Findings", "FigureS4_Prior_Sensitivity_X9.pdf"),
  plot = p_sens,
  width = 8,
  height = 5.5
)

# 8. MONTE CARLO SIMULATION GRID
# ==============================================================================
cat("\n--- Step 8: Starting Monte Carlo simulation grid ---\n")

# The uploaded paper version states 500 replications per design point.
# If reproducing an older draft that used 100, change this line only.
n_reps <- 500

n_scenarios <- c(35, 75, 150, 500)
p_scenarios <- c(9, 30)
rho_scenarios <- c(0.20, 0.60, 0.85)
signal_scenarios <- c("S1", "S2", "S3")
models_list <- c("OLS", "Ridge", "LASSO", "ENet", "BART", "RF", "XGB")

sim_results_master <- data.frame()
total_iters <- length(signal_scenarios) * length(p_scenarios) * length(rho_scenarios) * length(n_scenarios)

pb <- txtProgressBar(min = 0, max = total_iters, style = 3, width = 50)
iter_count <- 0

set.seed(2026)

for (sig in signal_scenarios) {
  for (p in p_scenarios) {
    beta_true <- rep(0, p)
    beta_true[1] <- 1.5
    beta_true[4] <- -1.0
    beta_true[7] <- 0.5
    beta_true[8] <- 2.0
    
    for (rho in rho_scenarios) {
      Sigma_ex <- matrix(rho, nrow = p, ncol = p)
      diag(Sigma_ex) <- 1
      
      for (n in n_scenarios) {
        iter_count <- iter_count + 1
        
        cell_losses <- matrix(NA_real_, nrow = n_reps, ncol = length(models_list))
        colnames(cell_losses) <- models_list
        oracle_losses <- numeric(n_reps)
        
        for (r in seq_len(n_reps)) {
          X_tr <- mvrnorm(n = n, mu = rep(0, p), Sigma = Sigma_ex)
          X_te <- mvrnorm(n = 10000, mu = rep(0, p), Sigma = Sigma_ex)
          
          eta_tr <- as.numeric(X_tr %*% beta_true)
          eta_te <- as.numeric(X_te %*% beta_true)
          
          if (sig == "S1") {
            y_tr <- eta_tr + rnorm(n, mean = 0, sd = sqrt(2.25))
            y_te <- eta_te + rnorm(10000, mean = 0, sd = sqrt(2.25))
            oracle_mean_te <- eta_te
          }
          
          if (sig == "S2") {
            nonlin_tr <- 0.5 * (X_tr[, 1] * X_tr[, 2]) + 0.5 * (X_tr[, 3]^2 - 1)
            nonlin_te <- 0.5 * (X_te[, 1] * X_te[, 2]) + 0.5 * (X_te[, 3]^2 - 1)
            
            y_tr <- eta_tr + nonlin_tr + rnorm(n, mean = 0, sd = sqrt(2.25))
            y_te <- eta_te + nonlin_te + rnorm(10000, mean = 0, sd = sqrt(2.25))
            oracle_mean_te <- eta_te + nonlin_te
          }
          
          if (sig == "S3") {
            sig_tr <- sqrt(exp(0.5 * X_tr[, 1]))
            sig_te <- sqrt(exp(0.5 * X_te[, 1]))
            
            y_tr <- eta_tr + sig_tr * rt(n, df = 5)
            y_te <- eta_te + sig_te * rt(10000, df = 5)
            oracle_mean_te <- eta_te
          }
          
          oracle_losses[r] <- sqrt(mean((y_te - oracle_mean_te)^2))
          
          # OLS
          m_ols <- suppressWarnings(lm(y ~ ., data = data.frame(y = y_tr, X_tr)))
          pred_ols <- as.numeric(predict(m_ols, newdata = data.frame(X_te)))
          cell_losses[r, "OLS"] <- sqrt(mean((y_te - pred_ols)^2))
          
          # Penalized regressions with training-only scaling
          sc_sim <- safe_scale(X_tr, X_te)
          inner_folds_sim <- min(10, n)
          
          m_ridge <- cv.glmnet(sc_sim$train, y_tr, alpha = 0, nfolds = inner_folds_sim, standardize = FALSE)
          pred_ridge <- as.numeric(predict(m_ridge, newx = sc_sim$test, s = "lambda.min"))
          cell_losses[r, "Ridge"] <- sqrt(mean((y_te - pred_ridge)^2))
          
          m_lasso <- cv.glmnet(sc_sim$train, y_tr, alpha = 1, nfolds = inner_folds_sim, standardize = FALSE)
          pred_lasso <- as.numeric(predict(m_lasso, newx = sc_sim$test, s = "lambda.min"))
          cell_losses[r, "LASSO"] <- sqrt(mean((y_te - pred_lasso)^2))
          
          m_enet <- cv.glmnet(sc_sim$train, y_tr, alpha = 0.5, nfolds = inner_folds_sim, standardize = FALSE)
          pred_enet <- as.numeric(predict(m_enet, newx = sc_sim$test, s = "lambda.min"))
          cell_losses[r, "ENet"] <- sqrt(mean((y_te - pred_enet)^2))
          
          # BART
          capture.output(
            m_bart <- wbart(
              x.train = X_tr, y.train = y_tr, x.test = X_te,
              ntree = 20, ndpost = 300, printevery = 100000
            )
          )
          pred_bart <- colMeans(m_bart$yhat.test)
          cell_losses[r, "BART"] <- sqrt(mean((y_te - pred_bart)^2))
          
          # Random forest
          m_rf <- randomForest(X_tr, y_tr, ntree = 100, mtry = max(1, floor(p / 3)))
          pred_rf <- as.numeric(predict(m_rf, X_te))
          cell_losses[r, "RF"] <- sqrt(mean((y_te - pred_rf)^2))
          
          # XGBoost
          m_xgb <- xgb.train(
            params = list(
              eta = 0.1,
              max_depth = 3,
              objective = "reg:squarederror"
            ),
            data = xgb.DMatrix(data = X_tr, label = y_tr),
            nrounds = 100,
            verbose = 0
          )
          pred_xgb <- as.numeric(predict(m_xgb, xgb.DMatrix(data = X_te)))
          cell_losses[r, "XGB"] <- sqrt(mean((y_te - pred_xgb)^2))
        }
        
        top1_vec <- top1_fractional(cell_losses)
        
        for (m in models_list) {
          rep_losses <- cell_losses[, m]
          
          sim_results_master <- rbind(
            sim_results_master,
            data.frame(
              Scenario = sig,
              N = n,
              P = p,
              Rho = rho,
              Model = m,
              Mean_RMSE = mean(rep_losses, na.rm = TRUE),
              SD_RMSE = sd(rep_losses, na.rm = TRUE),
              QRMSE_90 = as.numeric(quantile(rep_losses, 0.90, na.rm = TRUE)),
              ERisk = mean(rep_losses - oracle_losses, na.rm = TRUE),
              TopOne = top1_vec[m]
            )
          )
        }
        
        setTxtProgressBar(pb, iter_count)
      }
    }
  }
}
close(pb)

# Full simulation export
write.csv(
  sim_results_master,
  file.path(mainDir, "2. Simulation", "TableS9_Simulation_Full_Results.csv"),
  row.names = FALSE
)

# Main-text scenario tables under strong collinearity (p = 9, rho = 0.85)
table1_sim <- sim_results_master %>%
  filter(Scenario == "S1", P == 9, abs(Rho - 0.85) < 1e-12) %>%
  arrange(N, Model)

table2_sim <- sim_results_master %>%
  filter(Scenario == "S2", P == 9, abs(Rho - 0.85) < 1e-12) %>%
  arrange(N, Model)

table3_sim <- sim_results_master %>%
  filter(Scenario == "S3", P == 9, abs(Rho - 0.85) < 1e-12) %>%
  arrange(N, Model)

write.csv(table1_sim, file.path(mainDir, "2. Simulation", "Table1_Simulation_S1.csv"), row.names = FALSE)
write.csv(table2_sim, file.path(mainDir, "2. Simulation", "Table2_Simulation_S2.csv"), row.names = FALSE)
write.csv(table3_sim, file.path(mainDir, "2. Simulation", "Table3_Simulation_S3.csv"), row.names = FALSE)

# Figure 1: Simulation risk profiles
sim_plot_df <- sim_results_master %>%
  filter(P == 9, abs(Rho - 0.85) < 1e-12) %>%
  mutate(Model = factor(Model, levels = c("OLS", "Ridge", "LASSO", "ENet", "BART", "RF", "XGB")))

p_sim <- ggplot(sim_plot_df, aes(x = factor(N), y = Mean_RMSE, group = Model, color = Model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  facet_wrap(~Scenario, scales = "free_y") +
  scale_color_brewer(palette = "Dark2") +
  theme_pub() +
  labs(
    title = "Simulation risk profiles under strong collinearity",
    x = "Sample size (n)",
    y = "Mean out-of-sample RMSE"
  )

ggsave(
  file.path(mainDir, "2. Simulation", "Figure1_Simulation_Risk_Profiles.pdf"),
  plot = p_sim,
  width = 11,
  height = 5.5
)

cat("\nDone. Full manuscript-analysis workflow completed successfully.\n")
