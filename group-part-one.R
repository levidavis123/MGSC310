An Analysis of Football Players
MGSC 310 Final Project
Yoni Kazovsky, Levi Davis, Art Song, Adithya Mahesh

# load all your libraries here
library(tidyverse)
library(caret)
library(rsample) 
library(yardstick)
library(ggplot2)
library(randomForest)
library(randomForestExplainer)
library(ggplot2)
library(tidyverse)
library(cluster)
library(factoextra)
library(GGally)
# note, do not run install.packages() inside a code chunk. install them in the console outside of a code chunk. 
Data Exploration & Cleaning
First we took a look at the data set then created a subset called football_selected that only has the varibales we want in it. Finally, we changed the name of the shots on target variable because it was not working in the random forest model due to its strange naming convention usin the apostraphes.


football <- read_csv("datasets/soccer.csv")

#football %>% glimpse()

football <- football %>%
  mutate(Goals_per_Appearance = Goals / Appearances,
         Conversion_rate = Goals / Shots)
#football %>% glimpse()

football_selected <- football %>%
  select(Conversion_rate, Goals_per_Appearance, Goals, `Shots on target`, Appearances, Shots, Age, Position, Offsides) %>%
  drop_na()
football_selected = football_selected %>%
  rename(Shots_on_target = `Shots on target`)
  
football_selected %>% glimpse()
Rows: 264
Columns: 9
$ Conversion_rate      <dbl> 0.16097561, 0.06521739, 0.07142857, 0.00000000, 0.05555556, 0.00000000, 0…
$ Goals_per_Appearance <dbl> 0.17934783, 0.04761905, 0.01851852, 0.00000000, 0.03030303, 0.00000000, 0…
$ Goals                <dbl> 33, 3, 1, 0, 1, 0, 8, 1, 0, 39, 55, 5, 1, 4, 3, 37, 1, 3, 3, 0, 9, 0, 3, …
$ Shots_on_target      <dbl> 92, 12, 4, 5, 5, 9, 41, 4, 3, 93, 105, 18, 4, 8, 5, 145, 3, 13, 20, 0, 32…
$ Appearances          <dbl> 184, 63, 54, 47, 33, 57, 132, 28, 26, 99, 87, 33, 20, 23, 14, 236, 18, 37…
$ Shots                <dbl> 205, 46, 14, 44, 18, 27, 144, 15, 14, 204, 222, 51, 13, 20, 12, 393, 20, …
$ Age                  <dbl> 31, 24, 23, 28, 21, 21, 27, 19, 24, 29, 31, 25, 20, 21, 19, 32, 25, 22, 2…
$ Position             <chr> "Midfielder", "Midfielder", "Midfielder", "Midfielder", "Midfielder", "Mi…
$ Offsides             <dbl> 83, 0, 1, 2, 0, 0, 2, 13, 0, 62, 55, 4, 4, 5, 2, 33, 1, 0, 0, 1, 6, 0, 4,…
Data Spliting for Regression
We used a 75 to 25 split which resulted in a training set of about 200 observations


set.seed(310)

football_split <- initial_split(football_selected, prop = 0.75)
football_train <- training(football_split)
football_test <- testing(football_split)

football_train %>% glimpse()
Rows: 198
Columns: 9
$ Conversion_rate      <dbl> 0.00000000, 0.17371938, 0.00000000, 0.13333333, 0.03921569, 0.13669065, 0…
$ Goals_per_Appearance <dbl> 0.00000000, 0.63414634, 0.00000000, 0.18181818, 0.05000000, 0.18095238, 0…
$ Goals                <dbl> 0, 78, 0, 6, 2, 19, 7, 12, 20, 5, 0, 25, 2, 3, 75, 86, 0, 37, 1, 42, 2, 1…
$ Shots_on_target      <dbl> 3, 201, 6, 14, 15, 51, 17, 38, 61, 11, 2, 86, 5, 4, 267, 190, 0, 117, 2, …
$ Appearances          <dbl> 26, 123, 35, 33, 40, 105, 34, 93, 122, 36, 15, 181, 29, 25, 346, 196, 12,…
$ Shots                <dbl> 14, 449, 28, 45, 51, 139, 38, 97, 195, 44, 6, 224, 21, 13, 616, 445, 1, 3…
$ Age                  <dbl> 24, 28, 28, 25, 24, 26, 30, 29, 27, 26, 24, 29, 26, 26, 31, 28, 29, 30, 1…
$ Position             <chr> "Midfielder", "Forward", "Midfielder", "Forward", "Forward", "Midfielder"…
$ Offsides             <dbl> 0, 65, 1, 4, 22, 4, 9, 22, 20, 7, 2, 15, 1, 6, 163, 117, 0, 60, 0, 59, 5,…
dim(football_train)
[1] 198   9
dim(football_test)
[1] 66  9
Model & Model Performance
Here are the results of our model and its performance


model1 <- lm(Goals ~ Goals_per_Appearance + Conversion_rate + Shots_on_target + Appearances + Shots + Age + Position + Offsides, football_train)

# In-sample prediction (training)
y_hat_train <- predict(model1, football_train)

# Out-of-sample prediction (test)
y_hat_test <- predict(model1, newdata = football_test)

rmse_train <- RMSE(y_hat_train, football_train$Goals)
rmse_test <- RMSE(y_hat_test, football_test$Goals)

print(paste("In-sample RMSE:", rmse_train))
[1] "In-sample RMSE: 3.75657362034032"
print(paste("Out-of-sample RMSE:", rmse_test))
[1] "Out-of-sample RMSE: 7.30030273761757"
summary(model1)

Call:
lm(formula = Goals ~ Goals_per_Appearance + Conversion_rate + 
    Shots_on_target + Appearances + Shots + Age + Position + 
    Offsides, data = football_train)

Residuals:
     Min       1Q   Median       3Q      Max 
-16.9431  -1.1588   0.4106   1.3316  17.4625 

Coefficients:
                      Estimate Std. Error t value             Pr(>|t|)    
(Intercept)          -1.427328   3.307413  -0.432             0.666560    
Goals_per_Appearance 18.119887   3.740498   4.844           0.00000265 ***
Conversion_rate      -7.307742   3.530588  -2.070             0.039835 *  
Shots_on_target       0.399460   0.037767  10.577 < 0.0000000000000002 ***
Appearances           0.012607   0.008975   1.405             0.161767    
Shots                -0.050866   0.015774  -3.225             0.001487 ** 
Age                   0.025943   0.089519   0.290             0.772289    
PositionForward      -0.349670   2.806679  -0.125             0.900985    
PositionMidfielder   -0.698716   2.779626  -0.251             0.801802    
Offsides              0.062536   0.016982   3.682             0.000302 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

Residual standard error: 3.855 on 188 degrees of freedom
Multiple R-squared:  0.9566,    Adjusted R-squared:  0.9545 
F-statistic:   460 on 9 and 188 DF,  p-value: < 0.00000000000000022
Data Visualization
As we can see the relationship between conversion rate and goals shows a lot of heteroscedasticity. In general a higher conversion rate results in more goals, however, as we increase in conversion ratre that relationship weakens hence the cone shape (heteroscedasticity).


ggplot(football_train, aes(x = Conversion_rate, y = Goals_per_Appearance, color = Position)) +
  geom_point() +
  labs(x = "Conversion Rate", y = "Goals") +
  ggtitle("Relationship between Conv Rate and Goals")


Random Forest Model
Here we created our random forest model.


bag_mod <- randomForest(Goals ~ .,
                        football_selected,
                        ntree=50,
                        mtry=5,
                        importance=TRUE)
print(bag_mod)

Call:
 randomForest(formula = Goals ~ ., data = football_selected, ntree = 50,      mtry = 5, importance = TRUE) 
               Type of random forest: regression
                     Number of trees: 50
No. of variables tried at each split: 5

          Mean of squared residuals: 27.75886
                    % Var explained: 94.59
plot(bag_mod)



importance(bag_mod)
                      %IncMSE IncNodePurity
Conversion_rate      6.108641    3117.77650
Goals_per_Appearance 4.837681    6521.99257
Shots_on_target      9.931036   79000.64838
Appearances          3.338903     852.45817
Shots                6.216053   43108.23943
Age                  1.152108     187.36552
Position             0.144325      52.66348
Offsides             2.301087    7704.55352
varImpPlot(bag_mod)


Random Forest Visualization
Here is the min-depth distribution and multi way importance plot for our random forest model



# plot min depth distribution
plot_min_depth_distribution(bag_mod)


# plot variable two-way importance measure
plot_multi_way_importance(bag_mod)


Bonus! Clustering Algorithm
Here we assign clusters and add cluster as a column in or football dataframe.


football_numeric <- football %>%
  select(Goals, `Shots on target`, Appearances, Conversion_rate)

football_numeric <- football_numeric %>%
  mutate(across(everything(), ~replace_na(., mean(., na.rm = TRUE))))

football_scaled <- scale(football_numeric)

kmeans_result <- kmeans(football_scaled, centers = 4, nstart = 25)

football$cluster <- factor(kmeans_result$cluster, labels = c("Goal Scorers", "Unkownss", "Seasoned Stars", "Average Players"))

football %>% glimpse()
Rows: 571
Columns: 62
$ Name                     <chr> "Bernd Leno", "Matt Macey", "Rúnar Alex Rúnarsson", "Héctor Bellerín"…
$ `Jersey Number`          <dbl> 1, 33, 13, 2, 3, 4, 5, 16, 20, 21, 23, 31, 6, 10, 11, 15, 25, 28, 29,…
$ Club                     <chr> "Arsenal", "Arsenal", "Arsenal", "Arsenal", "Arsenal", "Arsenal", "Ar…
$ Position                 <chr> "Goalkeeper", "Goalkeeper", "Goalkeeper", "Defender", "Defender", "De…
$ Nationality              <chr> "Germany", "England", "Iceland", "Spain", "Scotland", "France", "Gree…
$ Age                      <dbl> 28, 26, 25, 25, 23, 19, 32, 25, 28, 25, 33, 27, 22, 31, 24, 23, 28, 2…
$ Appearances              <dbl> 64, 0, 0, 160, 16, 0, 44, 41, 99, 139, 194, 78, 2, 184, 63, 54, 47, 3…
$ Wins                     <dbl> 28, 0, 0, 90, 7, 0, 21, 25, 52, 47, 113, 40, 2, 100, 28, 26, 29, 12, …
$ Losses                   <dbl> 16, 0, 0, 37, 5, 0, 11, 9, 26, 57, 38, 22, 0, 39, 15, 16, 10, 12, 17,…
$ Goals                    <dbl> 0, 0, 0, 7, 1, 0, 3, 0, 7, 6, 13, 2, 1, 33, 3, 1, 0, 1, 0, 0, 8, 1, 0…
$ `Goals per match`        <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, 0.18, 0.05, 0.02,…
$ `Headed goals`           <dbl> NA, NA, NA, 0, 0, 0, 1, 0, 6, 1, 6, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, …
$ `Goals with right foot`  <dbl> NA, NA, NA, 4, 0, 0, 2, 0, 1, 4, 6, 0, 0, 4, 3, 1, 0, 1, 0, 0, 0, 0, …
$ `Goals with left foot`   <dbl> NA, NA, NA, 3, 1, 0, 0, 0, 0, 1, 1, 2, 0, 25, 0, 0, 0, 0, 0, 0, 8, 1,…
$ `Penalties scored`       <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, 0, 0, 0, 0, 0, 0,…
$ `Freekicks scored`       <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, 1, 0, 0, 0, 0, 0,…
$ Shots                    <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, 205, 46, 14, 44, …
$ `Shots on target`        <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, 92, 12, 4, 5, 5, …
$ `Shooting accuracy %`    <chr> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, "45%", "26%", "29…
$ `Hit woodwork`           <dbl> NA, NA, NA, 3, 0, 0, 1, 0, 2, 1, 5, 1, 0, 7, 1, 1, 0, 0, 1, 0, 2, 1, …
$ `Big chances missed`     <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, 26, 1, 1, 2, 2, 1…
$ `Clean sheets`           <dbl> 14, 0, 0, 53, 2, 0, 9, 10, 26, 28, 64, 17, 1, NA, NA, NA, NA, NA, NA,…
$ `Goals conceded`         <dbl> 82, 0, 0, 166, 16, 0, 58, 45, 117, 170, 173, 86, 1, NA, NA, NA, NA, N…
$ Tackles                  <dbl> NA, NA, NA, 214, 21, 0, 67, 50, 197, 257, 240, 112, 2, 161, 96, 101, …
$ `Tackle success %`       <chr> NA, NA, NA, "78%", "81%", "0%", "61%", "70%", "73%", "70%", "74%", "6…
$ `Last man tackles`       <dbl> NA, NA, NA, 1, 0, 0, 1, 1, 2, 1, 4, 0, 0, NA, NA, NA, NA, NA, NA, NA,…
$ `Blocked shots`          <dbl> NA, NA, NA, 32, 1, 0, 4, 3, 14, 22, 59, 11, 0, 58, 10, 3, 22, 4, 11, …
$ Interceptions            <dbl> NA, NA, NA, 208, 12, 0, 43, 54, 195, 247, 344, 72, 2, 75, 75, 62, 41,…
$ Clearances               <dbl> NA, NA, NA, 304, 32, 0, 175, 170, 531, 537, 832, 111, 8, 37, 53, 66, …
$ `Headed Clearance`       <dbl> NA, NA, NA, 143, 12, 0, 82, 92, 345, 301, 443, 68, 7, 13, 22, 29, 15,…
$ `Clearances off line`    <dbl> NA, NA, NA, 3, 0, 0, 1, 0, 2, 1, 2, 0, 1, NA, NA, NA, NA, NA, NA, NA,…
$ Recoveries               <dbl> NA, NA, NA, 732, 63, 0, 173, 192, 455, 735, 1042, 365, 6, 778, 327, 2…
$ `Duels won`              <dbl> NA, NA, NA, 611, 55, 0, 227, 176, 670, 703, 846, 284, 8, 625, 276, 21…
$ `Duels lost`             <dbl> NA, NA, NA, 709, 38, 0, 160, 142, 416, 619, 680, 264, 7, 744, 212, 20…
$ `Successful 50/50s`      <dbl> NA, NA, NA, 196, 12, 0, 11, 19, 26, 93, 80, 41, 0, 248, 30, 58, 11, 2…
$ `Aerial battles won`     <dbl> NA, NA, NA, 161, 10, 0, 115, 96, 413, 301, 344, 97, 4, 28, 32, 29, 18…
$ `Aerial battles lost`    <dbl> NA, NA, NA, 215, 16, 0, 73, 64, 218, 240, 256, 80, 3, 83, 47, 38, 34,…
$ `Own goals`              <dbl> 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, NA, NA, NA, NA, NA, NA, NA, NA…
$ `Errors leading to goal` <dbl> 7, 0, 0, 1, 0, 0, 1, 0, 3, 1, 4, 1, 0, 2, 0, 0, 0, 0, 1, 0, 7, NA, 0,…
$ Assists                  <dbl> 0, 0, 0, 18, 1, 0, 2, 0, 4, 3, 7, 11, 0, 54, 3, 3, 3, 1, 1, 0, 13, 5,…
$ Passes                   <dbl> 1783, 0, 0, 7125, 519, 0, 2416, 2200, 5907, 5935, 10165, 3201, 197, 1…
$ `Passes per match`       <dbl> 27.86, 0.00, 0.00, 44.53, 32.44, 0.00, 54.91, 53.66, 59.67, 42.70, 52…
$ `Big chances created`    <dbl> NA, NA, NA, 28, 1, 0, 4, 0, 5, 9, 22, 18, 0, 65, 3, 4, 2, 1, 4, 0, 12…
$ Crosses                  <dbl> NA, NA, NA, 389, 45, 0, 1, 3, 45, 210, 67, 123, 0, 834, 40, 57, 30, 9…
$ `Cross accuracy %`       <chr> NA, NA, NA, "16%", "18%", "0%", "1%", "67%", "20%", "18%", "28%", "20…
$ `Through balls`          <dbl> NA, NA, NA, 31, 0, 0, 2, 1, 13, 4, 79, 3, 0, 121, 4, 2, 4, 3, 16, 0, …
$ `Accurate long balls`    <dbl> 234, 0, 0, 144, 22, 0, 172, 116, 273, 304, 980, 67, 6, 276, 94, 74, 8…
$ Saves                    <dbl> 222, 0, 0, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA…
$ `Penalties saved`        <dbl> 1, 0, 0, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, …
$ Punches                  <dbl> 34, 0, 0, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
$ `High Claims`            <dbl> 26, 0, 0, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
$ Catches                  <dbl> 17, 0, 0, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
$ `Sweeper clearances`     <dbl> 28, 0, 0, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,…
$ `Throw outs`             <dbl> 375, 0, 0, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA…
$ `Goal Kicks`             <dbl> 489, 0, 0, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA…
$ `Yellow cards`           <dbl> 2, 0, 0, 23, 2, 0, 18, 8, 28, 28, 40, 12, 0, 13, 14, 6, 7, 2, 15, 0, …
$ `Red cards`              <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 0, 1, 2, 1, 0, 0, 0, 2, 0, 0, …
$ Fouls                    <dbl> 0, 0, 0, 125, 9, 0, 56, 32, 117, 137, 212, 74, 3, 95, 62, 35, 24, 13,…
$ Offsides                 <dbl> NA, NA, NA, 8, 0, 0, 1, 0, 7, 2, 5, 16, 0, 83, 0, 1, 2, 0, 0, 0, 2, 1…
$ Goals_per_Appearance     <dbl> 0.00000000, NaN, NaN, 0.04375000, 0.06250000, NaN, 0.06818182, 0.0000…
$ Conversion_rate          <dbl> NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, NA, 0.16097561, 0.065…
$ cluster                  <fct> Average Players, Average Players, Average Players, Goal Scorers, Aver…
avg_goals_per_cluster <- football %>%
  group_by(cluster) %>%
  summarise(avg_goals_per_appearance = mean(Goals_per_Appearance))

avg_goals_per_cluster %>% glimpse()
Rows: 4
Columns: 2
$ cluster                  <fct> Goal Scorers, Unkownss, Seasoned Stars, Average Players
$ avg_goals_per_appearance <dbl> 0.09119197, 0.73333333, 0.32950881, NaN
Cluster Visualization
As we can see we have 4 pretty distinct clusters characterized by certain performance metrics.


ggplot(football, aes(x = Appearances, y = Goals_per_Appearance, color = cluster)) +
  geom_point() +
  labs(title = "Appearances vs Goals per", x = "Appearances", y = "Goals per Appearance")


Clustering Performance
As we can see our silhouette scores suggest that some of our clusters are pretty unique from eachother while others aren’t as well distinguishable.


sil_scores <- silhouette(kmeans_result$cluster, dist(football_scaled))
summary(sil_scores)
Silhouette of 571 units in 4 clusters from silhouette.default(x = kmeans_result$cluster, dist = dist(football_scaled)) :
 Cluster sizes and average silhouette widths:
      124         3        30       414 
0.3198255 0.9943953 0.2576557 0.5697269 
Individual silhouette widths:
    Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
-0.08191  0.39780  0.53033  0.50129  0.65301  0.99490 
