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
    chol_check = factor(chol_check, levels = c(0,1), labels = c("No", "Yes")),
    smoker = factor(smoker, levels = c(0, 1), labels = c("No", "Yes")),
    stroke = factor(stroke, levels = c(0, 1), labels = c("No", "Yes")),
    heart_diseaseor_attack = factor(heart_diseaseor_attack, levels = c(0, 1), labels = c("No",     "Yes")),
    phys_activity = factor(phys_activity, levels = c(0, 1), labels = c("No", "Yes")),
    fruits = factor(fruits, levels = c(0, 1), labels = c("No", "Yes")),
    veggies = factor(veggies, levels = c(0, 1), labels = c("No", "Yes")),
    hvy_alcohol_consump = factor(hvy_alcohol_consump, levels = c(0, 1), labels = c("No", "Yes"  )),
    any_healthcare = factor(any_healthcare, levels = c(0, 1), labels = c("No", "Yes")),
    no_docbc_cost = factor(no_docbc_cost, levels = c(0, 1), labels = c("No", "Yes")),
    gen_hlth = factor(
      gen_hlth,
      levels = 1:5,
      labels = c("Excellent", "Very Good", "Good", "Fair", "Poor")
    ),
    diff_walk = factor(diff_walk, levels = c(0, 1), labels = c("No", "Yes")),
    sex = factor(sex, levels = c(0, 1), labels = c("Female", "Male")),
    education = factor(
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
    income = factor(
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


recipe_full <- recipe(diabetes_binary ~ ., data = data) |>
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

# Numeric defaults (means)
defaults <- data |>
  summarise(
    bmi       = mean(bmi,       na.rm = TRUE),
    ment_hlth = mean(ment_hlth, na.rm = TRUE),
    phys_hlth = mean(phys_hlth, na.rm = TRUE)
  ) |>
  as.list()

# Factor defaults: most prevalent level for each factor
default_high_bp            <- data |> count(high_bp)            |> slice_max(n, n = 1, with_ties = FALSE) |> pull(high_bp)
default_high_chol          <- data |> count(high_chol)          |> slice_max(n, n = 1, with_ties = FALSE) |> pull(high_chol)
default_chol_check         <- data |> count(chol_check)         |> slice_max(n, n = 1, with_ties = FALSE) |> pull(chol_check)
default_smoker             <- data |> count(smoker)             |> slice_max(n, n = 1, with_ties = FALSE) |> pull(smoker)
default_stroke             <- data |> count(stroke)             |> slice_max(n, n = 1, with_ties = FALSE) |> pull(stroke)
default_heart_dz           <- data |> count(heart_diseaseor_attack) |> slice_max(n, n = 1, with_ties = FALSE) |> pull(heart_diseaseor_attack)
default_phys_activity      <- data |> count(phys_activity)      |> slice_max(n, n = 1, with_ties = FALSE) |> pull(phys_activity)
default_fruits             <- data |> count(fruits)             |> slice_max(n, n = 1, with_ties = FALSE) |> pull(fruits)
default_veggies            <- data |> count(veggies)            |> slice_max(n, n = 1, with_ties = FALSE) |> pull(veggies)
default_hvy_alcohol        <- data |> count(hvy_alcohol_consump)|> slice_max(n, n = 1, with_ties = FALSE) |> pull(hvy_alcohol_consump)
default_any_healthcare     <- data |> count(any_healthcare)     |> slice_max(n, n = 1, with_ties = FALSE) |> pull(any_healthcare)
default_no_docbc_cost      <- data |> count(no_docbc_cost)      |> slice_max(n, n = 1, with_ties = FALSE) |> pull(no_docbc_cost)
default_gen_hlth           <- data |> count(gen_hlth)           |> slice_max(n, n = 1, with_ties = FALSE) |> pull(gen_hlth)
default_diff_walk          <- data |> count(diff_walk)          |> slice_max(n, n = 1, with_ties = FALSE) |> pull(diff_walk)
default_sex                <- data |> count(sex)                |> slice_max(n, n = 1, with_ties = FALSE) |> pull(sex)
default_education          <- data |> count(education)          |> slice_max(n, n = 1, with_ties = FALSE) |> pull(education)
default_income             <- data |> count(income)             |> slice_max(n, n = 1, with_ties = FALSE) |> pull(income)
default_age                <- data |> count(age)                |> slice_max(n, n = 1, with_ties = FALSE) |> pull(age)



# 4. plumber endpoints -----------------------------------------------------

#* @apiTitle Diabetes Prediction API

#* Make a prediction with ALL predictors
#* @param bmi Body mass index (numeric)
#* @param ment_hlth # days of poor mental health (numeric)
#* @param phys_hlth # days of poor physical health (numeric)
#* @param high_bp High blood pressure: "No" or "Yes"
#* @param high_chol High cholesterol: "No" or "Yes"
#* @param chol_check Cholesterol check in past 5 years: "No" or "Yes"
#* @param smoker Ever smoked 100+ cigarettes: "No" or "Yes"
#* @param stroke Ever had stroke: "No" or "Yes"
#* @param heart_diseaseor_attack Coronary heart disease / MI: "No" or "Yes"
#* @param phys_activity Recent physical activity: "No" or "Yes"
#* @param fruits Eats fruit 1+ times/day: "No" or "Yes"
#* @param veggies Eats veggies 1+ times/day: "No" or "Yes"
#* @param hvy_alcohol_consump Heavy alcohol consumption: "No" or "Yes"
#* @param any_healthcare Has any health care coverage: "No" or "Yes"
#* @param no_docbc_cost Could not see doctor due to cost: "No" or "Yes"
#* @param gen_hlth Self-rated general health: "Excellent", "Very Good", "Good", "Fair", "Poor"
#* @param diff_walk Serious difficulty walking: "No" or "Yes"
#* @param sex "Female" or "Male"
#* @param education Education category
#* @param income Income category
#* @param age Age category (e.g. "50–54")
#* @get /pred
function(
    bmi                = defaults$bmi,
    ment_hlth          = defaults$ment_hlth,
    phys_hlth          = defaults$phys_hlth,
    high_bp            = as.character(default_high_bp),
    high_chol          = as.character(default_high_chol),
    chol_check         = as.character(default_chol_check),
    smoker             = as.character(default_smoker),
    stroke             = as.character(default_stroke),
    heart_diseaseor_attack = as.character(default_heart_dz),
    phys_activity      = as.character(default_phys_activity),
    fruits             = as.character(default_fruits),
    veggies            = as.character(default_veggies),
    hvy_alcohol_consump= as.character(default_hvy_alcohol),
    any_healthcare     = as.character(default_any_healthcare),
    no_docbc_cost      = as.character(default_no_docbc_cost),
    gen_hlth           = as.character(default_gen_hlth),
    diff_walk          = as.character(default_diff_walk),
    sex                = as.character(default_sex),
    education          = as.character(default_education),
    income             = as.character(default_income),
    age                = as.character(default_age)
) {
  
  new_df <- tibble(
    bmi       = as.numeric(bmi),
    ment_hlth = as.numeric(ment_hlth),
    phys_hlth = as.numeric(phys_hlth),
    
    high_bp            = factor(high_bp,            levels = levels(data$high_bp)),
    high_chol          = factor(high_chol,          levels = levels(data$high_chol)),
    chol_check         = factor(chol_check,         levels = levels(data$chol_check)),
    smoker             = factor(smoker,             levels = levels(data$smoker)),
    stroke             = factor(stroke,             levels = levels(data$stroke)),
    heart_diseaseor_attack = factor(heart_diseaseor_attack,
                                    levels = levels(data$heart_diseaseor_attack)),
    phys_activity      = factor(phys_activity,      levels = levels(data$phys_activity)),
    fruits             = factor(fruits,             levels = levels(data$fruits)),
    veggies            = factor(veggies,            levels = levels(data$veggies)),
    hvy_alcohol_consump= factor(hvy_alcohol_consump,levels = levels(data$hvy_alcohol_consump)),
    any_healthcare     = factor(any_healthcare,     levels = levels(data$any_healthcare)),
    no_docbc_cost      = factor(no_docbc_cost,      levels = levels(data$no_docbc_cost)),
    gen_hlth           = factor(gen_hlth,           levels = levels(data$gen_hlth)),
    diff_walk          = factor(diff_walk,          levels = levels(data$diff_walk)),
    sex                = factor(sex,                levels = levels(data$sex)),
    education          = factor(education,          levels = levels(data$education)),
    income             = factor(income,             levels = levels(data$income)),
    age                = factor(age,                levels = levels(data$age))
  )
  
  pred_prob <- predict(model_fit, new_df, type = "prob")
  
  list(
    input      = new_df,
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
  
  # Create heatmap plot
  p <- autoplot(cm, type = "heatmap")
  
  # Draw the plot to the PNG device
  print(p)
  
  invisible(NULL)
}
