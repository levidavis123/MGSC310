library(tidyverse)
library(caret)
library(rsample) 
library(yardstick)

football <- read_csv("data/soccer.csv")
dim(football)

football_selected <- football %>%
  select(Goals, `Shots on target`, Appearances, Shots, Age, Position, `Big chances created`, `Big chances missed`, Offsides) %>%
  drop_na()

dim(football_selected)

set.seed(310)

football_split <- initial_split(football_selected, prop = 0.75)
football_train <- training(football_split)
football_test <- testing(football_split)

football_train %>% glimpse()

dim(football_train)
dim(football_test)

model1 <- lm(Goals ~ `Shots on target` + Appearances + Shots + Age + Position + `Big chances created` + `Big chances missed` + Offsides, football_train)

# In-sample prediction (training)
y_hat_train <- predict(model1, football_train)

# Out-of-sample prediction (test)
y_hat_test <- predict(model1, newdata = football_test)

rmse_train <- RMSE(y_hat_train, football_train$Goals)
rmse_test <- RMSE(y_hat_test, football_test$Goals)

print(paste("In-sample RMSE:", rmse_train))
print(paste("Out-of-sample RMSE:", rmse_test))