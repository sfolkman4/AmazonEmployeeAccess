
library(tidymodels)
library(embed)
library(ggplot2)
library(recipes)
library(vroom)
library(themis)
library(discrim)
#REGRESSION MODEL

train_data <- vroom("C:/Users/sfolk/Desktop/STAT348/AmazonEmployeeAccess/train.csv")
test_data <- vroom("C:/Users/sfolk/Desktop/STAT348/AmazonEmployeeAccess/test.csv")

train_data$ACTION <- as.factor(train_data$ACTION)

my_recipe <- recipe(ACTION ~ ., data = train_data) |> 
  step_mutate_at(all_numeric_predictors(), fn = factor) |> 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) |>
  step_normalize(all_numeric_predictors())

prep <- prep(my_recipe)
baked <- bake(prep, new_data = train_data)


logRegModel <- logistic_reg() |> 
  set_engine("glm")

log_wf <- workflow() |> 
  add_recipe(my_recipe) |> 
  add_model(logRegModel) |> 
  fit(data = train_data)



amazon_predictions <- predict(log_wf,
                              new_data=test_data,
                              type="class")


predictions_df <- data.frame(id = test_data$id, Action = as.numeric(amazon_predictions$.pred_class))

predictions_df$Action <- pmax(0, predictions_df$Action)

predictions_df$id <- as.character(predictions_df$id)

vroom_write(x = predictions_df, file = "C:/Users/sfolk/Desktop/STAT348/AmazonEmployeeAccess/RegPredictions.csv", delim = ",")




#PENALIZED REGRESSION

logRegModel <- logistic_reg(mixture = tune(), penalty = tune()) |>
  set_engine("glmnet")

penlog_wf <- workflow() |>
  add_recipe(my_recipe) |>
  add_model(logRegModel)


tuning_grid <- grid_regular(penalty(), mixture(), levels = 5)
folds <- vfold_cv(train_data, v = 5)

CV_results <- penlog_wf |>
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc) 
  )

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

final_wf <-
  penlog_wf |> 
  finalize_workflow(bestTune) |> 
  fit(data=train_data)

new_predictions <- predict(final_wf,
                              new_data=test_data,
                              type="class")


predictions_df <- data.frame(id = test_data$id, Action = as.numeric(new_predictions$.pred_class))


predictions_df$Action <- pmax(0, predictions_df$Action)

predictions_df$id <- as.character(predictions_df$id)

vroom_write(x = predictions_df, file = "C:/Users/sfolk/Desktop/STAT348/AmazonEmployeeAccess/PenReg_Predictions.csv", delim = ",")




#KNN


# Create KNN model
knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_engine("kknn") %>%
  set_mode("classification")

# Create workflow
knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn_model)

# Create tuning grid
tuning_grid <- grid_regular(neighbors(), levels = 10)

# Create cross-validation folds
folds <- vfold_cv(train_data, v = 5)

# Perform cross-validation
cv_results <- knn_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc)
  )

# Select best tuning parameters
best_tune <- cv_results %>%
  select_best(metric = "roc_auc")

# Finalize workflow with best tuning parameters
final_wf <- knn_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train_data)

# Make predictions on test data
new_predictions <- predict(final_wf,
                           new_data = test_data,
                           type = "class")

# Create predictions dataframe
predictions_df <- data.frame(id = test_data$id, Action = as.numeric(as.character(new_predictions$.pred_class)))

# Ensure Action is 0 or 1
predictions_df$Action <- pmax(0, predictions_df$Action)

# Convert id to character
predictions_df$id <- as.character(predictions_df$id)

# Write predictions to CSV
vroom_write(x = predictions_df, file = "C:/Users/sfolk/Desktop/STAT348/AmazonEmployeeAccess/KNN_Predictions.csv", delim = ",")






#Random Forest


rf_model <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees = 500) %>%
  set_engine("ranger") %>%
  set_mode("classification")


rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_model)


tuning_grid <- grid_regular(
  mtry(range = c(1, 9)),  
  min_n(),                
  levels = 5               
)


folds <- vfold_cv(train_data, v = 10)


cv_results <- rf_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc)
  )

  best_tune <- cv_results %>%
  select_best(metric = "roc_auc")


final_wf <- rf_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train_data)


new_predictions <- predict(final_wf,
                           new_data = test_data,
                           type = "class")

predictions_df <- data.frame(id = test_data$id, Action = as.numeric(as.character(new_predictions$.pred_class)))


predictions_df$Action <- pmax(0, predictions_df$Action)

print(names(predictions_df))


predictions_df$id <- as.character(predictions_df$id)

vroom_write(x = predictions_df, file = "C:/Users/sfolk/Desktop/STAT348/AmazonEmployeeAccess/RF_Predictions.csv", delim = ",")







#KNN

# Load data
train_data <- vroom("/home/slf77/STAT348/AmazonEmployeeAccess/train.csv")
test_data <- vroom("/home/slf77/STAT348/AmazonEmployeeAccess/test.csv")

# Convert ACTION to factor
train_data$ACTION <- as.factor(train_data$ACTION)

# Create recipe
knn_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(ACTION)|> 
  step_pca(all_predictors(), threshold = 0.8)

# Create KNN model
knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_engine("kknn") %>%
  set_mode("classification")

# Create workflow
knn_wf <- workflow() %>%
  add_recipe(knn_recipe) %>%
  add_model(knn_model)

# Create tuning grid
tuning_grid <- grid_regular(neighbors(), levels = 10)

# Create cross-validation folds
folds <- vfold_cv(train_data, v = 5)

# Perform cross-validation
cv_results <- knn_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc)
  )

# Select best tuning parameters
best_tune <- cv_results %>%
  select_best(metric = "roc_auc")

print(best_tune)
show_best(cv_results, metric = "roc_auc")

# Finalize workflow with best tuning parameters
final_wf <- knn_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train_data)

# Make predictions on test data
new_predictions <- predict(final_wf,
                           new_data = test_data,
                           type = "class")

# Create predictions dataframe
predictions_df <- data.frame(id = test_data$id, Action = as.numeric(as.character(new_predictions$.pred_class)))

# Ensure Action is 0 or 1
predictions_df$Action <- pmax(0, predictions_df$Action)

print(names(predictions_df))

# Convert id to character
predictions_df$id <- as.character(predictions_df$id)

# Write predictions to CSV
vroom_write(x = predictions_df, file = "/home/slf77/STAT348/AmazonEmployeeAccess/KNN_Predictions.csv", delim = ",")






#Naive Bayes
library(naivebayes)


train_data <- vroom("/home/slf77/STAT348/AmazonEmployeeAccess/train.csv")
test_data <- vroom("/home/slf77/STAT348/AmazonEmployeeAccess/test.csv")

train_data$ACTION <- as.factor(train_data$ACTION)


nb_recipe <- recipe(ACTION~., data=train_data) |> 
  step_mutate_at(all_numeric_predictors(), fn=factor) |> 
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) |> 
  step_normalize(all_numeric_predictors())|> 
  step_pca(all_predictors(), threshold = 0.8)


nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) |> 
  set_mode("classification") |> 
  set_engine("naivebayes")


nb_wf <- workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(nb_model)


tuning_grid <- grid_regular(Laplace(), smoothness(), levels = 5)


folds <- vfold_cv(train_data, v = 5)


cv_results <- nb_wf %>%
  tune_grid(
    resamples = folds,
    grid = tuning_grid,
    metrics = metric_set(roc_auc)
  )

best_tune <- cv_results %>%
  select_best(metric = "roc_auc")


final_wf <- nb_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train_data)


new_predictions <- predict(final_wf,
                           new_data = test_data,
                           type = "probability")

predictions_df <- data.frame(id = test_data$id, Action = as.numeric(as.character(new_predictions$.pred_class)))


predictions_df$Action <- pmax(0, predictions_df$Action)

print(names(predictions_df))


predictions_df$id <- as.character(predictions_df$id)

vroom_write(x = predictions_df, file = "/home/slf77/STAT348/AmazonEmployeeAccess/NB_Predictions.csv", delim = ",")
