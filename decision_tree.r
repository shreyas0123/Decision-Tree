################### problem1 ################################3
#load the dataset
company_data <- read.csv("C:\\Users\\DELL\\Downloads\\Decision Tree\\Company_Data.csv",stringsAsFactors = TRUE)
#converting continous data of target to categorical data
company_data$Sales <- cut(company_data$Sales,breaks = c(-Inf, 5.4, 11,Inf),labels = c("low","middle","high"))

##Exploring and preparing the data ----
str(company_data)
summary(company_data$Sales)

# look at the class variable
table(company_data$Sales)

#splitting train and test data
set.seed(0)
split <- sample.split(company_data$Sales, SplitRatio = 0.8)
company_data_train <- subset(company_data, split == TRUE)
company_data_test <- subset(company_data, split == FALSE)

# check the proportion of class variable for train and test data
prop.table(table(company_data$Sales))
prop.table(table(company_data_train$Sales))
prop.table(table(company_data_test$Sales))

# Step 3: Training a model on the data
install.packages("C50")
library(C50)

company_data_model <- C5.0(company_data_train[, -1], company_data_train$Sales)

windows()
plot(company_data_model) 

# Display detailed information about the tree
summary(company_data_model)

# Step 4: Evaluating model performance
# Test data accuracy
test_res <- predict(company_data_model, company_data_test)
test_acc <- mean(company_data_test$Sales == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(company_data_test$Sales, test_res, dnn = c('actual default', 'predicted default'))

# On Training Dataset
train_res <- predict(company_data_model, company_data_train)
train_acc <- mean(company_data_train$Sales == train_res)
train_acc

table(company_data_train$Sales, train_res)
#overfitting is occured due to training accuracy is high and testing accuracy is low

#prune the decision tree

library(rpart)
model <- rpart(company_data_train$Sales ~ .,data = company_data_train,method = "class",control = rpart.control(cp = 0, maxdepth = 3))

# Plot Decision Tree
install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(model, box.palette = "auto", digits = -3)

# Evaluating model performance
# Test data accuracy
test_res <- predict(model, company_data_test,type = "class")
test_acc <- mean(company_data_test$Sales == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(company_data_test$Sales, test_res, dnn = c('actual default', 'predicted default'))

# On Training Dataset
train_res <- predict(model, company_data_train,type = "class")
train_acc <- mean(company_data_train$Sales == train_res)
train_acc

table(company_data_train$Sales, train_res)
# Now its a right fit model because train accuracy and test accuracy almost similar

################### Random forest ######################################################
##Exploring and preparing the data ----
# install.packages("randomForest")
install.packages("randomForest")
library(randomForest)

rf <- randomForest(company_data_train$Sales ~ ., data = company_data_train,ntree = 50,maxnodes = 15,method  = "class")

#on test data
test_res <- predict(rf, company_data_test,type = "class")
test_acc <- mean(company_data_test$Sales == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(company_data_test$Sales, test_res, dnn = c('actual default', 'predicted default'))

# On Training Dataset
train_res <- predict(rf, company_data_train,type = "class")
train_acc <- mean(company_data_train$Sales == train_res)
train_acc

table(company_data_train$Sales, train_res)

############################### problem2 ##########################################################################
#load the dataset
diabetes <- read.csv("C://Users//DELL//Downloads//Decision Tree//Diabetes.csv",stringsAsFactors = TRUE)
##Exploring and preparing the data ----
str(diabetes)

# look at the class variable
table(diabetes$Class.variable)

#prepare tarining and testing data
library(caTools)
set.seed(0)
split <- sample.split(diabetes$Class.variable, SplitRatio = 0.8)
diabetes_train <- subset(diabetes, split == TRUE)
diabetes_test <- subset(diabetes, split == FALSE)

# check the proportion of class variable
prop.table(table(diabetes$Class.variable))
prop.table(table(diabetes_train$Class.variable))
prop.table(table(diabetes_test$Class.variable))

# Step 3: Training a model on the data
install.packages("C50")
library(C50)

diabetes_model <- C5.0(diabetes_train[, -9], diabetes_train$Class.variable)

windows()
plot(diabetes_model) 

# Display detailed information about the tree
summary(diabetes_model)

# Step 4: Evaluating model performance
# Test data accuracy
test_res <- predict(diabetes_model, diabetes_test)
test_acc <- mean(diabetes_test$Class.variable == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(diabetes_test$Class.variable, test_res, dnn = c('actual default', 'predicted default'))

# On Training Dataset
train_res <- predict(diabetes_model, diabetes_train)
train_acc <- mean(diabetes_train$Class.variable == train_res)
train_acc

table(diabetes_train$Class.variable, train_res)
# overfitting issues since testing accuracy is low and traning accuracy is high

#pruning the model
library(rpart)
model <- rpart(diabetes_train$Class.variable ~ ., data = diabetes_train,method = "class",control = rpart.control(cp = 0, maxdepth = 3))

# Plot Decision Tree
library(rpart.plot)
rpart.plot(model, box.palette = "auto", digits = 4)

# Measure the RMSE on Test data
test_pred <- predict(model, newdata = diabetes_test, type = "class")

# RMSE
accuracy1 <- sqrt(mean(diabetes_test$Class.variable == test_pred)^2)
accuracy1

# Measure the RMSE on Train data
train_pred <- predict(model, newdata = diabetes_train, type = "class")

# RMSE
accuracy_train <- sqrt(mean(diabetes_train$Class.variable == train_pred)^2)
accuracy_train

#Training and Testing accuracy is almost similar hence it is right fit model

######################### Random Forest ###############################################
# install.packages("randomForest")
library(randomForest)

rf <- randomForest(diabetes_train$Class.variable ~ .,ntree = 5, maxnodes = 4  ,data = diabetes_train)

test_rf_pred <- predict(rf, diabetes_test)

rmse_rf <- sqrt(mean(diabetes_test$Class.variable == test_rf_pred)^2)
rmse_rf

# Prediction for trained data result
train_rf_pred <- predict(rf, diabetes_train)

# RMSE on Train Data
train_rmse_rf <- sqrt(mean(diabetes_train$Class.variable == train_rf_pred)^2)
train_rmse_rf

#Training and testing accuracy is almost simlilar hence it is a right fit model

############################### problem3 #####################################
##Exploring and preparing the data ----
str(fraud_check)
summary(company_data$Sales)

# look at the class variable
table(fraud_check$Taxable.Income)

#splitting train and test data
set.seed(0)
split <- sample.split(fraud_check$Taxable.Income, SplitRatio = 0.8)
fraud_check_train <- subset(fraud_check, split == TRUE)
fraud_check_test <- subset(fraud_check, split == FALSE)

# check the proportion of class variable for train and test data
prop.table(table(fraud_check$Taxable.Income))
prop.table(table(fraud_check_train$Taxable.Income))
prop.table(table(fraud_check_test$Taxable.Income))

# Step 3: Training a model on the data
install.packages("C50")
library(C50)

fraud_check_model <- C5.0(fraud_check_train[, -3], fraud_check_train$Taxable.Income)

windows()
plot(fraud_check_model) 

# Display detailed information about the tree
summary(fraud_check_model)

# Step 4: Evaluating model performance
# Test data accuracy
test_res <- predict(fraud_check_model, fraud_check_test)
test_acc <- mean(fraud_check_test$Taxable.Income == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(fraud_check_test$Taxable.Income, test_res, dnn = c('actual default', 'predicted default'))

# On Training Dataset
train_res <- predict(fraud_check_model, fraud_check_train)
train_acc <- mean(fraud_check_train$Taxable.Income == train_res)
train_acc

table(fraud_check_train$Taxable.Income, train_res)
#overfitting is occured 
#prune the decision tree

library(rpart)
model <- rpart(fraud_check_train$Taxable.Income ~ .,data = fraud_check_train,method = "class",control = rpart.control(cp = 0, maxdepth = 3))

# Plot Decision Tree
install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(model, box.palette = "auto", digits = 2)

# Evaluating model performance
# Test data accuracy
test_res <- predict(model, fraud_check_test,type = "class")
test_acc <- mean(fraud_check_test$Taxable.Income == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(fraud_check_test$Taxable.Income, test_res, dnn = c('actual default', 'predicted default'))

# On Training Dataset
train_res <- predict(model, fraud_check_train,type = "class")
train_acc <- mean(fraud_check_train$Taxable.Income == train_res)
train_acc

table(fraud_check_train$Taxable.Income, train_res)
# Now its a right fit model because train accuracy and test accuracy almost similar

################### Random forest ######################################################
##Exploring and preparing the data ----
# install.packages("randomForest")
install.packages("randomForest")
library(randomForest)

rf <- randomForest(c$Taxable.Income ~ ., data = fraud_check_train,ntree = 50,maxnodes = 15,method  = "class")

#on test data
test_res <- predict(rf, fraud_check_test,type = "class")
test_acc <- mean(fraud_check_test$Taxable.Income == test_res)
test_acc

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(fraud_check_test$Taxable.Income, test_res, dnn = c('actual default', 'predicted default'))

# On Training Dataset
train_res <- predict(rf, fraud_check_train,type = "class")
train_acc <- mean(fraud_check_train$Taxable.Income == train_res)
train_acc

table(fraud_check_train$Taxable.Income, train_res)
#now model is right fit .

########################### problem4 #################################################
#load the dataset 
HR_data <- read.csv("C:\\Users\\DELL\\Downloads\\Decision Tree\\HR_DT.csv",stringsAsFactors = TRUE)

# look at the class variable
table(HR_data$monthly.income.of.employee)
str(HR_data)

summary(HR_data$monthly.income.of.employee)

#splitting train and test data
library(caTools)
set.seed(0)
split <- sample.split(HR_data$monthly.income.of.employee, SplitRatio = 0.8)
HR_data_train <- subset(HR_data, split == TRUE)
HR_data_test <- subset(HR_data, split == FALSE)

library(rpart)
model <- rpart(HR_data_train$monthly.income.of.employee ~ ., data = HR_data_train,method = "class",
               control = rpart.control(cp = 0, maxdepth = 5))

# Plot Decision Tree
library(rpart.plot)
rpart.plot(model, box.palette = "auto", digits = 3)


# Measure the RMSE on Test data
test_pred <- predict(model, newdata = HR_data_test, type = "vector")

# RMSE
accuracy1 <- sqrt(mean(HR_data_test$monthly.income.of.employee - test_pred)^2)
accuracy1

# Measure the RMSE on Train data
train_pred <- predict(model, newdata = HR_data_train, type = "vector")

# RMSE
accuracy_train <- sqrt(mean(HR_data_train$monthly.income.of.employee - train_pred)^2)
accuracy_train

# Prune the Decision Tree

# Grow the full tree
fullmodel <- rpart(HR_data_train$monthly.income.of.employee ~ ., data = HR_data_train,
                   control = rpart.control(cp = 0))

rpart.plot(fullmodel, box.palette = "auto", digits = 5)

# Examine the complexity plot
# Tunning parameter check the value of cp which is giving us minimum cross validation error (xerror)
printcp(fullmodel)   
plotcp(model)

mincp <- model$cptable[which.min(model$cptable[, "xerror"]), "CP"]

# Prune the model based on the optimal cp value
model_pruned_1 <- prune(fullmodel, cp = mincp)
rpart.plot(model_pruned_1, box.palette = "auto", digits = 5)

model_pruned_2 <- prune(fullmodel, cp = 0.02)
rpart.plot(model_pruned_2, box.palette = "auto", digits = 5)


# Measure the RMSE using Full tree
test_pred_fultree <- predict(fullmodel, newdata = HR_data_test, type = "vector")
# RMSE
accuracy_f <- sqrt(mean(HR_data_test$monthly.income.of.employee - test_pred_fultree)^2)
accuracy_f

# Measure the RMSE using Prune tree - model1
test_pred_prune1 <- predict(model_pruned_1, newdata = HR_data_test, type = "vector")
# RMSE
accuracy_prune1 <- sqrt(mean(HR_data_test$monthly.income.of.employee - test_pred_prune1)^2)
accuracy_prune1

# Measure the RMSE using Prune tree - model2
test_pred_prune2 <- predict(model_pruned_2, newdata = HR_data_test, type = "vector")
# RMSE
accuracy_prune2 <- sqrt(mean(HR_data_test$monthly.income.of.employee - test_pred_prune2)^2)
accuracy_prune2

# Prediction for trained data result
train_pred_fultree <- predict(fullmodel, HR_data_train, type = 'vector')

# RMSE on Train Data
train_accuracy_fultree <- sqrt(mean(HR_data_train$monthly.income.of.employee - train_pred_fultree)^2)
train_accuracy_fultree


# Prediction for trained data result
train_pred_prune1 <- predict(model_pruned_1, HR_data_train, type = 'vector')

# RMSE on Train Data
train_accuracy_fultree2 <- sqrt(mean(HR_data_train$monthly.income.of.employee - train_pred_prune1)^2)
train_accuracy_fultree2

# Prediction for trained data result
train_pred_prune2 <- predict(model_pruned_2, HR_data_train, type = 'vector')

# RMSE on Train Data
train_accuracy_fultree2 <- sqrt(mean(HR_data_train$monthly.income.of.employee - train_pred_prune2)^2)
train_accuracy_fultree2

#RMSE value for both test and accuracy is almost right hence its a right fit model

################# Random Forest #####################################################
# install.packages("randomForest")
library(randomForest)

rf <- randomForest(monthly.income.of.employee ~ .,maxnodes = 4,ntree = 3, data = HR_data_train)

test_rf_pred <- predict(rf, HR_data_test)

rmse_rf <- sqrt(mean(HR_data_test$monthly.income.of.employee - test_rf_pred)^2)
rmse_rf

# Prediction for trained data result
train_rf_pred <- predict(rf, HR_data_train)

# RMSE on Train Data
train_rmse_rf <- sqrt(mean(HR_data_train$monthly.income.of.employee - train_rf_pred)^2)
train_rmse_rf

#now checking that the candidate claim is genuine or fake
#candidate claims are stored in a dataframe
cols <- c("Position.of.the.employee", "no.of.Years.of.Experience.of.employee","monthly.income.of.employee")
Candidate_claim <- data.frame(a <-  "Region Manager" , b <-  5 , c <-  70000)
colnames(Candidate_claim) <- cols
#binding with test data for prediction
HR_data_test <- rbind(HR_data_test, Candidate_claim)

#predicting using model
HR_data_test$monthly.income.of.employee.pred <- predict(model, newdata = HR_data_test)

#our predicted salary is (61131.74)
HR_data_test$monthly.income.of.employee.pred[31]

#since the  predicted salary is almost similar to claimed salary it is said that candidate is genuine 






