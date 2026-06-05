# 0. SETUP, DIRECTORIES & DEPENDENCY MANAGEMENT
# ==============================================================================
mainDir <- getwd()
dir.create(file.path(mainDir, '1. Findings'), showWarnings = FALSE)
dir.create(file.path(mainDir, '2. Simulation'), showWarnings = FALSE)

# Required Packages:
# install.packages(c("glmnet", "brms", "spdep", "BART", "xgboost", "shapviz", 
#                    "loo", "car", "ggplot2", "randomForest", "MASS", 
#                    "kernlab", "dplyr", "sf", "rnaturalearth", "tidyr", 
#                    "ggcorrplot", "bayesplot"))

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

# 1. EMPIRICAL APPLICATION DATA PREPARATION (INDONESIA 2021 BPS DATASET)
# ==============================================================================
cat("--- Processing Step 1: Loading Empirical Application Matrix ---\n")
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

X_mat <- as.matrix(df[, 3:11])
X_scaled <- scale(X_mat)
sd_factors <- apply(X_mat, 2, sd)
Y <- df$Y
df$Y_beta <- df$Y / 100 
df$spatial_id <- as.character(1:nrow(df))

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

# Construct Spatial Graph (BYM2 Adjacency Matrix)
coords_matrix <- as.matrix(real_coords)
knn_nb <- knn2nb(knearneigh(coords_matrix, k = 2))
sym_nb <- make.sym.nb(knn_nb)
W_list <- nb2listw(sym_nb, style = "W", zero.policy = FALSE)
A <- nb2mat(sym_nb, style = "B", zero.policy = FALSE)
rownames(A) <- colnames(A) <- df$spatial_id

# 2. FREQUENTIST PENALISED SHRINKAGE METHODS
# ==============================================================================
cat("--- Processing Step 2: Running Frequentist Shrinkage Loops ---\n")
set.seed(123)
m1_ols <- lm(Y ~ ., data = data.frame(Y = Y, X_scaled))

# Added 'keep = TRUE' to extract out-of-sample LOO predictions for MAE calculation
cv_ridge <- cv.glmnet(X_scaled, Y, alpha = 0, nfolds = 34, keep = TRUE)
cv_lasso <- cv.glmnet(X_scaled, Y, alpha = 1, nfolds = 34, keep = TRUE)
cv_enet  <- cv.glmnet(X_scaled, Y, alpha = 0.5, nfolds = 34, keep = TRUE) 

# Convert scaled metrics back to absolute operational scales
ols_orig <- coef(m1_ols)[-1] / sd_factors
ridge_orig <- as.numeric(coef(cv_ridge, s = "lambda.min")[-1]) / sd_factors
lasso_orig <- as.numeric(coef(cv_lasso, s = "lambda.min")[-1]) / sd_factors
enet_orig <- as.numeric(coef(cv_enet, s = "lambda.min")[-1]) / sd_factors

freq_coefs <- data.frame(
  Predictor = paste0("X", 1:9),
  OLS = round(ols_orig, 3), Ridge = round(ridge_orig, 3),
  LASSO = round(lasso_orig, 3), Elastic_Net = round(enet_orig, 3)
)

# 3. BAYESIAN PARAMETRIC ESTIMATION SUITE (brms Engines)
# ==============================================================================
cat("--- Processing Step 3: Compiling Bayesian Samplers ---\n")
brms_form <- Y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9

m5_brms <- brm(formula=brms_form, data=df, prior=set_prior("normal(0,5)", class="b"), chains=4, iter=4000, warmup=2000, cores=4, silent=2, refresh=0)
m6_bridge <- brm(formula=brms_form, data=df, prior=set_prior("student_t(1,0,1)", class="b"), chains=4, iter=4000, warmup=2000, cores=4, silent=2, refresh=0)
m7_blasso <- brm(formula=brms_form, data=df, prior=set_prior("double_exponential(0,1)", class="b"), chains=4, iter=4000, warmup=2000, cores=4, silent=2, refresh=0)
m8_horseshoe <- brm(formula = brms_form, data = df, prior = set_prior(horseshoe(1, 3/34), class="b"), chains = 4, iter = 4000, warmup = 2000, cores = 4, silent = 2, refresh = 0, control = list(adapt_delta = 0.995, max_treedepth = 15))
m10_beta <- brm(formula = Y_beta ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9, data=df, family=Beta(), prior=set_prior("normal(0,5)", class="b"), chains=4, iter=4000, warmup=2000, cores=4, silent=2, refresh=0)
m11_icar <- brm(formula = Y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + car(A, gr=spatial_id, type="bym2"), data = df, data2 = list(A=A), family = gaussian(), prior = c(set_prior("inv_gamma(0.5, 0.005)", class = "sigma"), set_prior("inv_gamma(0.5, 0.005)", class = "sdcar")), chains = 4, iter = 2000, warmup = 1000, cores = 4, silent = 2, refresh = 0)


# ==============================================================================
# 3.b EXTRACTION OF MCMC CONVERGENCE DIAGNOSTICS (Reviewer Real-Time Amendment)
# ==============================================================================
cat("--- Processing Step 3.b: Extracting Parameter-Wide MCMC Diagnostics ---\n")

extract_convergence_bounds <- function(brms_model, model_name, model_id) {
  # Safe extraction using the underlying rstan fit object engine
  stan_summary <- rstan::summary(brms_model$fit)$summary
  
  # Isolate convergence metric dimensions safely removing any structural NA strings
  rhats <- stan_summary[, "Rhat"]
  neffs <- stan_summary[, "n_eff"]
  rhats <- rhats[!is.na(rhats)]
  neffs <- neffs[!is.na(neffs)]
  
  data.frame(
    ID = model_id,
    Model = model_name,
    Min_Rhat = round(min(rhats), 3),
    Max_Rhat = round(max(rhats), 3),
    Min_ESS  = round(min(neffs), 0),
    Max_ESS  = round(max(neffs), 0)
  )
}

# Map over the active brms fit tracking matrix
diag_m5  <- extract_convergence_bounds(m5_brms, "Bayes LM (Gaussian)", "M5")
diag_m6  <- extract_convergence_bounds(m6_bridge, "Bayesian ridge", "M6")
diag_m7  <- extract_convergence_bounds(m7_blasso, "Bayesian lasso", "M7")
diag_m8  <- extract_convergence_bounds(m8_horseshoe, "Horseshoe", "M8")
diag_m10 <- extract_convergence_bounds(m10_beta, "Beta regression", "M10")
diag_m11 <- extract_convergence_bounds(m11_icar, "Spatial ICAR", "M11")

tab_convergence <- rbind(diag_m5, diag_m6, diag_m7, diag_m8, diag_m10, diag_m11)

# Write to the destination tracking directory
write.csv(tab_convergence, paste0(mainDir, "/1. Findings/TableS8_Convergence_Diagnostics.csv"), row.names = FALSE)
cat("MCMC Convergence metrics calculated and archived cleanly.\n")


# 4. NON-PARAMETRIC ALGORITHMIC ENGINE ESTIMATIONS
# ==============================================================================
cat("--- Processing Step 4: Running Algorithmic Cross-Validation loops ---\n")
set.seed(99)
m12_bart <- wbart(x.train=X_mat, y.train=Y, ntree=50, ndpost=10000, sparse=FALSE, printevery=100000)
bart_imp <- (m12_bart$varcount.mean / sum(m12_bart$varcount.mean)) * 100
m14_rf <- randomForest(X_mat, Y, ntree=500, mtry=3, importance=TRUE)
xgb_params <- list(eta=0.05, max_depth=3, colsample_bytree=0.8, objective="reg:squarederror")

preds_bart <- preds_gp <- preds_rf <- preds_xgb <- numeric(34)
for(i in 1:34) {
  capture.output(b_mod <- wbart(x.train=X_mat[-i,], y.train=Y[-i], x.test=X_mat[i,,drop=FALSE], ntree=50, ndpost=2000, printevery=100000))
  preds_bart[i] <- mean(b_mod$yhat.test)
  gp_mod <- gausspr(X_scaled[-i,], Y[-i], kernel="rbfdot", variance.model=FALSE)
  preds_gp[i] <- predict(gp_mod, X_scaled[i,,drop=FALSE])
  rf_mod <- randomForest(X_mat[-i,], Y[-i], ntree=500, mtry=3)
  preds_rf[i] <- predict(rf_mod, X_mat[i,,drop=FALSE])
  xgb_mod <- xgb.train(params=xgb_params, data=xgb.DMatrix(data=X_mat[-i,], label=Y[-i]), nrounds=250, verbose=0)
  preds_xgb[i] <- predict(xgb_mod, xgb.DMatrix(data=X_mat[i,,drop=FALSE]))
}

# 5. EMPIRICAL VALIDATION MATRIX COMPILATION (Table 5 Layout Mapping)
# ==============================================================================
get_metrics <- function(mod) {
  preds <- suppressWarnings(loo_predict(mod))
  est <- if("Estimate" %in% colnames(preds)) preds[, "Estimate"] else as.numeric(preds)
  return(c(RMSE=sqrt(mean((df$Y - est)^2)), MAE=mean(abs(df$Y - est))))
}
mets <- lapply(list(m5_brms, m6_bridge, m7_blasso, m8_horseshoe, m11_icar), get_metrics)
loo_ols <- (m1_ols$residuals) / (1 - hatvalues(m1_ols))

# Extracting LOO predictions directly from cv.glmnet objects to calculate MAE
preds_ridge <- cv_ridge$fit.preval[, cv_ridge$lambda == cv_ridge$lambda.min]
preds_lasso <- cv_lasso$fit.preval[, cv_lasso$lambda == cv_lasso$lambda.min]
preds_enet  <- cv_enet$fit.preval[, cv_enet$lambda == cv_enet$lambda.min]

tab_empirical <- data.frame(
  ID = paste0("M", c(1:8, 10:15)),
  Model = c("OLS", "Ridge", "LASSO", "Elastic net", "Bayes LM (Gaussian)", "Bayesian ridge", "Bayesian LASSO", "Horseshoe", "Beta regression", "Spatial ICAR", "BART", "Gaussian process", "Random forest", "XGBoost"),
  LOO_RMSE = c(sqrt(mean(loo_ols^2)), sqrt(min(cv_ridge$cvm)), sqrt(min(cv_lasso$cvm)), sqrt(min(cv_enet$cvm)), mets[[1]]["RMSE"], mets[[2]]["RMSE"], mets[[3]]["RMSE"], mets[[4]]["RMSE"], NA, mets[[5]]["RMSE"], sqrt(mean((Y - preds_bart)^2)), sqrt(mean((Y - preds_gp)^2)), sqrt(mean((Y - preds_rf)^2)), sqrt(mean((Y - preds_xgb)^2))),
  LOO_MAE = c(mean(abs(loo_ols)), mean(abs(Y - preds_ridge)), mean(abs(Y - preds_lasso)), mean(abs(Y - preds_enet)), mets[[1]]["MAE"], mets[[2]]["MAE"], mets[[3]]["MAE"], mets[[4]]["MAE"], NA, mets[[5]]["MAE"], mean(abs(Y - preds_bart)), mean(abs(Y - preds_gp)), mean(abs(Y - preds_rf)), mean(abs(Y - preds_xgb)))
)
tab_empirical$LOO_RMSE[9] <- sqrt(mean((df$Y - suppressWarnings(loo_predict(m10_beta))[,1]*100)^2))
tab_empirical$LOO_MAE[9]  <- mean(abs(df$Y - suppressWarnings(loo_predict(m10_beta))[,1]*100))

# Renamed to Table5 to strictly adhere to the condensed manuscript tracking index
write.csv(tab_empirical, paste0(mainDir,"/1. Findings/Table5_Empirical_Comparison.csv"), row.names = FALSE)

# 6. PRIOR SENSITIVITY PIPELINES & PLOT GENERATION
# ==============================================================================
cat("--- Processing Step 6: Running Prior Sensitivity Analysis ---\n")
scales <- c(1, 10, 25, 100, 1000)
sens_results <- data.frame(Prior_Scale = scales, Mean = NA, P2.5 = NA, P97.5 = NA)
for(i in 1:length(scales)) {
  tmp_mod <- brm(formula = brms_form, data = df, prior = set_prior(paste0("normal(0,", sqrt(scales[i]), ")"), class="b"), chains = 2, iter = 2000, warmup = 1000, cores = 2, silent = 2, refresh = 0) 
  tmp_fix <- summary(tmp_mod)$fixed
  sens_results$Mean[i] <- tmp_fix["X9", "Estimate"]
  sens_results$P2.5[i]  <- tmp_fix["X9", "l-95% CI"]
  sens_results$P97.5[i] <- tmp_fix["X9", "u-95% CI"]
}

# Build the Sensitivity Plot object (p5_sens)
p5_sens <- ggplot(sens_results, aes(x = factor(Prior_Scale), y = Mean)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "#e74c3c", alpha = 0.8) +
  geom_point(size = 3, color = "#2980b9") +
  geom_errorbar(aes(ymin = P2.5, ymax = P97.5), width = 0.2, color = "#2980b9", linewidth = 0.8) +
  theme_classic(base_size = 14) %+replace% theme(
    plot.title = element_text(face = "bold", size = rel(1.1), margin = ggplot2::margin(b = 6)),
    axis.title = element_text(face = "bold"),
    panel.grid.major.y = element_line(color = "grey93", linetype = "dotted")
  ) +
  labs(x = "Prior Scale (Variance)", y = "Posterior Estimate for ICT Skills (X9)")


# 7. MANUSCRIPT PLOT GENERATION CODES
# ==============================================================================
cat("--- Processing Step 7: Exporting Manuscript Figures ---\n")
theme_pub <- function() {
  theme_classic(base_size = 14) %+replace% theme(
    plot.title = element_text(face = "bold", size = rel(1.1), margin = ggplot2::margin(b = 6)),
    legend.position = "bottom", axis.title = element_text(face = "bold"),
    panel.grid.major.x = element_line(color = "grey93", linetype = "dotted"),
    panel.grid.major.y = element_line(color = "grey93", linetype = "dotted")
  )
}

# --- Standard Manuscript Figures ---

# Figure 2: Predictor Correlation Heatmap Matrix
ggsave(paste0(mainDir, "/1. Findings/Figure2_Corr_Heatmap.pdf"), plot = ggcorrplot(cor(X_mat), type="lower", lab=TRUE, colors = c("#e74c3c", "white", "#3498db"), ggtheme=theme_pub()), width = 8, height = 7)

# Figure 3: Spatial Distribution Maps
world_states <- ne_download(scale = 50, type = "states", category = "cultural", returnclass = "sf")
indo_merged <- world_states %>% filter(admin == "Indonesia") %>% left_join(data.frame(name = c("Aceh", "Sumatera Utara", "Sumatera Barat", "Riau", "Jambi", "Sumatera Selatan", "Bengkulu", "Lampung", "Kepulauan Bangka Belitung", "Kepulauan Riau", "Jakarta Raya", "Jawa Barat", "Jawa Tengah", "Yogyakarta", "Jawa Timur", "Banten", "Bali", "Nusa Tenggara Barat", "Nusa Tenggara Timur", "Kalimantan Barat", "Kalimantan Tengah", "Kalimantan Selatan", "Kalimantan Timur", "Kalimantan Utara", "Sulawesi Utara", "Sulawesi Tengah", "Sulawesi Selatan", "Sulawesi Tenggara", "Gorontalo", "Sulawesi Barat", "Maluku", "Maluku Utara", "Papua Barat", "Papua"), Y = df$Y), by = "name")
p2_map <- ggplot(data = indo_merged) + geom_sf(aes(fill = Y), color = "white", linewidth = 0.1) + scale_fill_gradientn(colours = c("#2980b9", "#f1c40f", "#e74c3c"), name = "Poverty %") + theme_pub() + theme(panel.background = element_rect(fill = "#eef4f7"))
ggsave(paste0(mainDir, "/1. Findings/Figure3_Realistic_Choropleth.pdf"), plot = p2_map, width = 12, height = 6)

# Figure 4: Variable Relative Contribution Profiles (BART and Random Forest)
imp_df <- data.frame(Predictor = rep(colnames(X_mat), 2), Algorithm = rep(c("BART", "RF"), each=9), Importance = c(bart_imp, (m14_rf$importance[,"%IncMSE"]/sum(m14_rf$importance[,"%IncMSE"]))*100))
p4_imp <- ggplot(imp_df, aes(x = Importance, y = Predictor, fill = Algorithm)) + geom_col(position = "dodge") + scale_fill_manual(values = c("#2c3e50", "#e67e22")) + theme_pub()
ggsave(paste0(mainDir, "/1. Findings/Figure4_Var_Importance.pdf"), plot = p4_imp, width = 8, height = 5.5)

# Figure 5: Unified SHAP Beeswarm Configuration Engine
shp <- shapviz(xgb.train(params = xgb_params, data = xgb.DMatrix(data = X_mat, label = Y), nrounds = 250, verbose = 0), X_pred = X_mat)
ggsave(paste0(mainDir, "/1. Findings/Figure5_SHAP_XGBoost.pdf"), 
       plot = sv_importance(shp, kind = "beeswarm") + theme_pub() + theme(legend.position = "right"), 
       width = 9, height = 6)

# Figure 6: Leave-One-Out (LOO) Predictive Performance Comparison Matrix
tab_empirical <- read.csv(paste0(mainDir, "/1. Findings/Table5_Empirical_Comparison.csv"))
tab_long <- tab_empirical %>%
  drop_na(LOO_RMSE, LOO_MAE) %>%
  pivot_longer(cols = c(LOO_RMSE, LOO_MAE), names_to = "Metric", values_to = "Value") %>%
  mutate(Metric = ifelse(Metric == "LOO_RMSE", "LOO RMSE", "LOO MAE"))

model_order <- tab_empirical %>% arrange(desc(LOO_RMSE)) %>% pull(Model)
tab_long$Model <- factor(tab_long$Model, levels = model_order)

p6_empirical <- ggplot(tab_long, aes(x = Value, y = Model, fill = Metric)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.65) +
  scale_fill_manual(values = c("#2980b9", "#2c3e50")) +
  theme_pub() +
  labs(x = "Error Metric Value (Lower is Better)", y = "Evaluated Predictive Models", fill = "Performance Metric") +
  theme(legend.position = "bottom", axis.text.y = element_text(face = "plain"))

ggsave(paste0(mainDir, "/1. Findings/Figure6_Empirical_Performance.pdf"), plot = p6_empirical, width = 9, height = 6.5)


# --- Supplementary Figures Generation ---
cat("--- Generating Supplementary Figures ---\n")

# 1. Build the Horseshoe Forest Plot object (p3_forest)
m8_summary <- summary(m8_horseshoe)$fixed[-1, ] # Remove intercept
forest_df <- data.frame(
  Predictor = factor(paste0("X", 1:9), levels = rev(paste0("X", 1:9))),
  Mean = m8_summary[, "Estimate"],
  Lower = m8_summary[, "l-95% CI"],
  Upper = m8_summary[, "u-95% CI"]
)
p3_forest <- ggplot(forest_df, aes(x = Mean, y = Predictor)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "#e74c3c", alpha = 0.8) +
  geom_point(size = 3, color = "#2c3e50") +
  geom_errorbarh(aes(xmin = Lower, xmax = Upper), height = 0.2, color = "#2c3e50", linewidth = 0.8) +
  theme_pub() +
  labs(x = "Posterior Mean with 95% Credible Interval", y = "Predictor")

# 2. Export Supplementary Figures
ggsave(paste0(mainDir, "/1. Findings/Supp_Horseshoe_Forest.pdf"), plot = p3_forest, width = 8, height = 5)
ggsave(paste0(mainDir, "/1. Findings/Supp_PPC_Horseshoe.pdf"), plot = pp_check(m8_horseshoe, ndraws = 100) + theme_pub(), width = 8, height = 5)
ggsave(paste0(mainDir, "/1. Findings/Supp_Prior_Sensitivity.pdf"), plot = p5_sens, width = 8, height = 5.5)

# ==============================================================================
# 8. EXPANDED MONTE CARLO STRESS-TESTING ASYMPTOTIC SIMULATION GRID
# ==============================================================================
cat("\n--- Starting Step 8: Multi-Scenario Asymptotic Simulations ---\n")

# PRODUCTION BENCHMARKS (Set n_reps = 500 for final manuscript output generation)
n_reps <- 500
n_scenarios <- c(35, 75, 150, 500)
p_scenarios <- c(9, 30)
rho_scenarios <- c(0.20, 0.60, 0.85)
signal_scenarios <- c("S1", "S2", "S3")
models_list <- c("OLS", "Ridge", "LASSO", "ENet", "BART", "RF", "XGB")

sim_results_master <- data.frame()
total_iters <- length(signal_scenarios) * length(p_scenarios) * length(rho_scenarios) * length(n_scenarios)

# Setup text progress tracking metric bar
pb <- txtProgressBar(min = 0, max = total_iters, style = 3, width = 50)
iter_count <- 0

for(sig in signal_scenarios) {
  for(p in p_scenarios) {
    beta_true <- rep(0, p)
    beta_true[1] <- 1.5; beta_true[4] <- -1.0; beta_true[7] <- 0.5; beta_true[8] <- 2.0
    
    for(rho in rho_scenarios) {
      Sigma_ex <- matrix(rho, nrow = p, ncol = p)
      diag(Sigma_ex) <- 1
      
      for(n in n_scenarios) {
        iter_count <- iter_count + 1
        cell_losses <- matrix(NA, nrow = n_reps, ncol = length(models_list))
        colnames(cell_losses) <- models_list
        oracle_losses <- numeric(n_reps)
        
        for(r in 1:n_reps) {
          X_tr <- mvrnorm(n, mu = rep(0, p), Sigma = Sigma_ex)
          X_te <- mvrnorm(10000, mu = rep(0, p), Sigma = Sigma_ex)
          
          if(sig == "S1") {
            y_tr <- as.numeric(X_tr %*% beta_true) + rnorm(n, 0, sqrt(2.25))
            y_te <- as.numeric(X_te %*% beta_true) + rnorm(10000, 0, sqrt(2.25))
            oracle_losses[r] <- sqrt(mean((y_te - as.numeric(X_te %*% beta_true))^2))
          } else if(sig == "S2") {
            y_tr <- as.numeric(X_tr %*% beta_true) + 0.5*(X_tr[,1]*X_tr[,2]) + 0.5*(X_tr[,3]^2 - 1) + rnorm(n, 0, sqrt(2.25))
            y_te <- as.numeric(X_te %*% beta_true) + 0.5*(X_te[,1]*X_te[,2]) + 0.5*(X_te[,3]^2 - 1) + rnorm(10000, 0, sqrt(2.25))
            oracle_losses[r] <- sqrt(mean((y_te - (as.numeric(X_te %*% beta_true) + 0.5*(X_te[,1]*X_te[,2]) + 0.5*(X_tr[,3]^2 - 1)))^2))
          } else if(sig == "S3") {
            sig_tr <- sqrt(exp(0.5 * X_tr[,1]))
            sig_te <- sqrt(exp(0.5 * X_te[,1]))
            y_tr <- as.numeric(X_tr %*% beta_true) + sig_tr * rt(n, df = 5)
            y_te <- as.numeric(X_te %*% beta_true) + sig_te * rt(10000, df = 5)
            oracle_losses[r] <- sqrt(mean((y_te - as.numeric(X_te %*% beta_true))^2))
          }
          
          m1_sim <- suppressWarnings(lm(y ~ ., data = data.frame(y = y_tr, X_tr)))
          cell_losses[r, "OLS"] <- sqrt(mean((y_te - predict(m1_sim, data.frame(X_te)))^2))
          
          cv_idx <- min(5, n)
          m2_sim <- cv.glmnet(X_tr, y_tr, alpha = 0, nfolds = cv_idx)
          cell_losses[r, "Ridge"] <- sqrt(mean((y_te - predict(m2_sim, X_te, s = "lambda.min"))^2))
          
          m3_sim <- cv.glmnet(X_tr, y_tr, alpha = 1, nfolds = cv_idx)
          cell_losses[r, "LASSO"] <- sqrt(mean((y_te - predict(m3_sim, X_te, s = "lambda.min"))^2))
          
          m4_sim <- cv.glmnet(X_tr, y_tr, alpha = 0.5, nfolds = cv_idx)
          cell_losses[r, "ENet"] <- sqrt(mean((y_te - predict(m4_sim, X_te, s = "lambda.min"))^2))
          
          capture.output(m12_sim <- wbart(x.train = X_tr, y.train = y_tr, x.test = X_te, ntree = 20, ndpost = 300, printevery = 100000))
          cell_losses[r, "BART"] <- sqrt(mean((y_te - colMeans(m12_sim$yhat.test))^2))
          
          m14_sim <- randomForest(X_tr, y_tr, ntree = 100, mtry = max(1, floor(p/3)))
          cell_losses[r, "RF"] <- sqrt(mean((y_te - predict(m14_sim, X_te))^2))
          
          m15_sim <- xgb.train(params = list(eta = 0.1, max_depth = 3, objective = "reg:squarederror"), data = xgb.DMatrix(data = X_tr, label = y_tr), nrounds = 100, verbose = 0)
          cell_losses[r, "XGB"] <- sqrt(mean((y_te - predict(m15_sim, xgb.DMatrix(data = X_te)))^2))
        }
        
        best_in_rep <- apply(cell_losses, 1, which.min)
        for(m in models_list) {
          rep_losses <- cell_losses[, m]
          sim_results_master <- rbind(sim_results_master, data.frame(
            Scenario = sig, N = n, P = p, Rho = rho, Model = m,
            Mean_RMSE = mean(rep_losses, na.rm=TRUE), SD_RMSE = sd(rep_losses, na.rm=TRUE),
            QRMSE_90 = as.numeric(quantile(rep_losses, 0.90, na.rm=TRUE)),
            ERisk = mean(rep_losses - oracle_losses, na.rm=TRUE), TopOne = sum(models_list[best_in_rep] == m) / n_reps
          ))
        }
        setTxtProgressBar(pb, iter_count)
      }
    }
  }
}
close(pb)

# Export final metrics matrices (Table 1 Main Simulation Matrix mapping)
write.csv(sim_results_master, paste0(mainDir, "/2. Simulation/Table1_Simulation_Main_Results.csv"), row.names = FALSE)

# Figure 1: Asymptotic Performance Convergence Line Charts
sim_plot_df <- sim_results_master %>% filter(P == 9, Rho == 0.85) %>%
  mutate(Model = factor(Model, levels = c("OLS", "Ridge", "LASSO", "ENet", "BART", "RF", "XGB")))

p_sim_curves <- ggplot(sim_plot_df, aes(x = factor(N), y = Mean_RMSE, group = Model, color = Model)) +
  geom_line(linewidth = 1) + geom_point(size = 3) + facet_wrap(~Scenario, scales = "free_y") +
  scale_color_brewer(palette = "Dark2") + theme_pub() + 
  labs(x = "Sample Size (n)", y = "Mean Out-of-Sample RMSE")
ggsave(paste0(mainDir, "/2. Simulation/Figure1_Simulation_Risk_Profiles.pdf"), plot = p_sim_curves, width = 11, height = 5.5)

cat("Done. Journal compilation cycle terminates cleanly.\n")
