# api.R

library(tidyverse)
library(tidymodels)
library(plumber)

tidymodels_prefer()

# 1. Load and prep data 
data <- read_csv("data/diabetes_binary_health_indicators_BRFSS2015.csv", show_col_types = FALSE) |>
  janitor::clean_names() |>
  mutate(
    diabetes_binary = factor(diabetes_binary, levels = c(0, 1),
                             labels = c("No_Diabetes", "Diabetes")),
    high_bp  = factor(high_bp,  levels = c(0, 1), labels = c("No", "Yes")),
    high_chol = factor(high_chol, levels = c(0, 1), labels = c("No", "Yes")),
    chol_check = factor(chol_check, level = c(0,1), labels = c("No", "Yes")),
    smoker = factor(smoker, levels = c(0, 1), labels = c("No", "Yes")),
    stroke = factor(stroke, levels = c(0, 1), labels = c("No", "Yes")),
    heart_diseaseor_attack = factor(heart_diseaseor_attack, levels = c(0, 1), labels = c("No",     "Yes")),
    phys_activity = factor(phys_activity, levels = c(0, 1), labels = c("No", "Yes")),
    fruits = factor(fruits, levels = c(0, 1), labels = c("No", "Yes")),
    veggies = factor(veggies, levels = c(0, 1), labels = c("No", "Yes")),
    hvy_alcohol_consump = factor(hvy_alcohol_consump, levels = c(0, 1), labels = c("No", "Yes"  )),
    any_healthcare = factor(any_healthcare, levels = c(0, 1), labels = c("No", "Yes")),
    no_docbc_cost = factor(no_docbc_cost, levels = c(0, 1), labels = c("No", "Yes")),
    gen_hlth <- factor(
      gen_hlth,
      levels = 1:5,
      labels = c("Excellent", "Very Good", "Good", "Fair", "Poor")
    ),
    diff_walk = factor(diff_walk, levels = c(0, 1), labels = c("No", "Yes")),
    sex = factor(sex, levels = c(0, 1), labels = c("Female", "Male")),
    education <- factor(
      education,
      levels = 1:6,
      labels = c(
        "K-8",
        "Grades 1-8",
        "Grades 9-11",
        "High School / GED",
        "Some College",
        "College Graduate"
      )
    ),
    income <- factor(
      income,
      levels = 1:8,
      labels = c(
        "<10k",
        "10-15k",
        "15-20k",
        "20-25k",
        "25-35k",
        "35-50k",
        "50-75k",
        "75k+"
      )
    ),
    age = factor(
      age,
      levels = 1:13,
      labels = c(
        "18–24", "25–29", "30–34", "35–39",
        "40–44", "45–49", "50–54", "55–59",
        "60–64", "65–69", "70–74", "75–79",
        "80+"
      )
    )
  )
# Choose predictors 
predictors <- c("bmi",
                "high_bp",
                "high_chol",
                "phys_activity",
                "age",
                "smoker",
                "stroke")

recipe_full <- recipe(
  diabetes_binary ~ bmi + high_bp + high_chol + phys_activity + age + smoker + stroke,
  data = data
) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors())

# 2. Recreate best model spec 
rf_spec_final <- rand_forest(
  mode = "classification",
  mtry = 2,         # from best_rf
  trees = 500,
  min_n = 27        # from best_rf
) |>
  set_engine("ranger", importance = "impurity")

rf_workflow_final <- workflow() |>
  add_model(rf_spec_final) |>
  add_recipe(recipe_full)

model_fit <- fit(rf_workflow_final, data = data)

# 3. Helper: default values for predictors 

# Numeric default
defaults <- data |>
  summarise(
    bmi = mean(bmi, na.rm = TRUE)
  ) |>
  as.list()

# Factor defaults 
default_high_bp      <- data |> count(high_bp)      |> slice_max(n, n = 1, with_ties = FALSE) |> pull(high_bp)
default_high_chol    <- data |> count(high_chol)    |> slice_max(n, n = 1, with_ties = FALSE) |> pull(high_chol)
default_phys_activity<- data |> count(phys_activity)|> slice_max(n, n = 1, with_ties = FALSE) |> pull(phys_activity)
default_age          <- data |> count(age)          |> slice_max(n, n = 1, with_ties = FALSE) |> pull(age)
default_smoker       <- data |> count(smoker)       |> slice_max(n, n = 1, with_ties = FALSE) |> pull(smoker)
default_stroke       <- data |> count(stroke)       |> slice_max(n, n = 1, with_ties = FALSE) |> pull(stroke)


# 4. plumber endpoints ----

#* @apiTitle Diabetes Prediction API

#* Make a prediction
#* @param high_bp Whether high BP: "No" or "Yes"
#* @param high_chol Whether high cholesterol: "No" or "Yes"
#* @param phys_activity Physical activity: "No" or "Yes"
#* @param bmi Body mass index (numeric)
#* @param smoker "No" or "Yes"
#* @param stroke "No" or "Yes"
#* @param age Age category, e.g. "50–54"
#* @get /pred
function(high_bp       = as.character(default_high_bp),
         high_chol     = as.character(default_high_chol),
         phys_activity = as.character(default_phys_activity),
         bmi           = defaults$bmi,
         smoker        = as.character(default_smoker),
         stroke        = as.character(default_stroke),
         age           = as.character(default_age)) {
  
  new_df <- tibble(
    high_bp       = factor(high_bp,       levels = levels(data$high_bp)),
    high_chol     = factor(high_chol,     levels = levels(data$high_chol)),
    phys_activity = factor(phys_activity, levels = levels(data$phys_activity)),
    bmi           = as.numeric(bmi),
    smoker        = factor(smoker,        levels = levels(data$smoker)),
    stroke        = factor(stroke,        levels = levels(data$stroke)),
    age           = factor(age,           levels = levels(data$age))
  )
  
  pred_prob <- predict(model_fit, new_df, type = "prob")
  
  list(
    input = new_df,
    prediction = pred_prob
  )
}


#* Info endpoint
#* @get /info
function() {
  list(
    name = "Derek Chao",
    github_pages_url = "https://chaowl2.github.io/ST558---Project3/EDA.html"
  )
}

#* Confusion matrix plot
#* @serializer png
#* @get /confusion
function() {
  library(ggplot2)
  library(yardstick)
  
  # Predictions (class labels)
  preds <- predict(model_fit, data, type = "class") |>
    bind_cols(data |> select(diabetes_binary))
  
  # Confusion matrix object
  cm <- conf_mat(preds, truth = diabetes_binary, estimate = .pred_class)
  
  # Return heatmap
  autoplot(cm, type = "heatmap")
}
