# Advanced Regression techniques to predict house prices----
# 24/04/2019

options(scipen = 999)  #Tp remove scientific notation

# Loading packages----
library(tidyverse) # data manipulation
library(caret)  # machine learning
library(lares)  #model evaluation
library(DataExplorer)   #Data exploratory
library(tidyquant)   #For themes
library(Metrics)   #Rmse evaluation




#Reading Data----
train <- read.csv("Data/train.csv",stringsAsFactors = FALSE,na.strings = c("NA",""))
test <- read.csv("Data/test.csv",stringsAsFactors = FALSE,na.strings = c("NA",""))

attach(train)

# Data Cleaning----
# variable names
names(train)

# Looking at the structure
str(train)

# Train data set has 1460 observations with 81  variables
# Looking at the summary of data set
summary(train)

# Getting rid of variables about year as this variable are likely  not to be predictive
train <- select(train, -(starts_with('Y',ignore.case = FALSE))) # For year
train <- select(train, -(starts_with('Mo',ignore.case = FALSE))) #For month

target_saleprice <- train$SalePrice
train$SalePrice <- NULL 
Test_Id <- test$Id
train$Id <- NULL
test$Id <- NULL

# Looking at missing values per column
colSums(is.na(train))

# LotFrontage,Alley,FireplaceQu,GarageType,GarageYrBlt,PoolQC,Fence, MiscFeature and other
# some other variables have more than 50 Nas

# Removing variables with missing values more than 50----
train  <- train[,colSums(is.na(train)) < 50]


# Replacing remaining  Nas in character variables with Missing value----

for (col in colnames(train)) {
  if(class(train[,col])=="character"){
    new_col <- train[,col]
    new_col[is.na(new_col)] <- "MISSING"
    train[,col]  <- as.factor(new_col)
  }
}

colSums(is.na(train))

# Replace NAs in numeric columns with median value ----

for (col in colnames(train)) {
  if(class(train[,col])=="integer"){
    new_col <- train[,col]
    new_col[is.na(new_col)] <- median(new_col,na.rm = T)
    train[,col]  <- as.numeric(new_col)
  }
}

# Checking no more Nas
anyNA(train)

# There are no more missing values

# Exploratory data Analysis----
# Adding back target variable to train data set
train$SalePrice <- target_saleprice

#Target variable, SalePrice----
summary(train$SalePrice)

# SalePrice is positively skewed

# Visualizing its distribution.

ggplot(train,aes(SalePrice)) + 
  geom_histogram(binwidth =10000,bins = 80,color="red")+
  ggtitle("SalePrice Distribution")+
  theme_tq()

# The saleprice is positively skewed as we expected,
# a few houses having high prices
# and majority median price

# Taking log10 of SalePrice to normalize
ggplot(train,aes(SalePrice)) + 
  geom_histogram(color="red")+
  ggtitle("SalePrice Distribution")+
  scale_x_log10()+
  theme_tq()

# SalePrice is now almost normalize

# Looking at distribution of all discrete variables using DataExplorer package----
plot_bar(train)

# All these discrete variables are imbalanced with exception of KitchenQual, and we should
# remove theme as they will bring bias in our model and hence reducing accuracy
#Removing imbalanced discrete variables
imb_discrete <- c('CentralAir','Electrical','Functional','PavedDrive','SaleType','SaleCondition')

train <- select(train,-imb_discrete)

# Looking at distribution of all continuous variables at once----
plot_histogram(train)

# Some of continous variables are extremely skewed and imbalance and we remove them
imb_cont <- c('EnclosedPorch','KitchenAbvGr','MiscVal','PoolArea','ScreenPorch','X3SsnPorch')

train <- select(train, -imb_cont)

#Advanced Data Exploratory----
# Removing near zero variance variables----
train$SalePrice <- NULL
nzv <- nearZeroVar(train)

train <- train[,-nzv]

#Removing highly correlated variables----
numer <- sapply(train, is.numeric)
train_numeric <- train[,numer]
train_numeric$SalePrice <- NULL
corr <- cor(train_numeric)

# We use correlation cutoff of 0.7
highlycor <- findCorrelation(corr,cutoff = 0.7)
train <- train[,-highlycor]

#Prepocessing the data set by scaling, centering----
pre_data <- preProcess(train, method = c("center","scale"))

train <- predict(pre_data, train)

summary(train)

# Creating dummy variables of nominal variable----
train$SalePrice <- target_saleprice

dummy <- dummyVars(~., data=train)

train <- predict(dummy, newdata = train)


# Exploratory data modelling----

# Data Splitting----
train$SalePrice <- target_saleprice

set.seed(100)
index <- createDataPartition(train$SalePrice,times = 1,p=0.7,list = FALSE)
training <- train[index,]
validating <- train[-index,]


# Modelling----
#Setting train Control
set.seed(100)
control <- trainControl(method = "repeatedcv",repeats=5,number =5 )

#Setting grid



#Setting train model
# taking log10 of SalePrice
training$SalePrice <- log10(training$SalePrice)
set.seed(100)
xgb_tree <- train(SalePrice~.,data=training,method="xgbTree",metric="RMSE",
               trControl=control)

xgb_tree



#Making Predictions 
pred <- predict(xgb_tree,newdata = validating)

summary(pred)

# Taking inverse of log10 since we took log10 of target variable
pred_clean <- 10^pred

summary(pred_clean)

#Evaluating model performance----
#Since this is a regression model, we use RMSE and Adjusted R^2 to evaluate model performance
# calculating RMSE
rmse(validating$SalePrice, pred_clean)

# Calculating RMSE,Adj R^2 by means of visualization using lares package
mplot_full(validating$SalePrice, pred_clean)


# We have RMSE of 26,570 and Adjusted R^2 of 0.8888 which is reasonably good.

# Preparing test data set for submission to kaggle competition for scoring----
#Subsetting test data to include only variables in train data set
train$SalePrice <- NULL
name_col <- colnames(train)
test <- test[,name_col]


#Replacing Nas with Missing value
#Integer variables
for (col in colnames(test)) {
  if(class(test[,col])=="character"){
    new_col <- test[,col]
    new_col[is.na(new_col)] <- "MISSING"
    test[,col]  <- as.factor(new_col)
  }
}

#Integer variables

for (col in colnames(test)) {
  if(class(test[,col])=="integer"){
    new_col <- test[,col]
    new_col[is.na(new_col)] <- median(new_col,na.rm = T)
    test[,col]  <- as.numeric(new_col)
  }
}

#Checking for Nas
anyNA(test)

#Scaling and centering test data set
pre_test <- preProcess(test, method = c("center","scale"))

test <- predict(pre_test, newdata = test)

summary(test)

# Making predictions on test data for submission to kaggle----
#cleaning some factor variables with label not  present in train data set
summary(test$MSZoning)
test$MSZoning[test$MSZoning=="MISSING"] <- "RL"
test$MSZoning <- factor(test$MSZoning)

table(test$Exterior1st)
test$Exterior1st[test$Exterior1st=="MISSING"] <- "Plywood"
test$Exterior1st <- factor(test$Exterior1st)

table(test$Exterior2nd)
test$Exterior2nd[test$Exterior2nd=="MISSING"] <- "Plywood"
test$Exterior2nd <- factor(test$Exterior2nd)

table(test$KitchenQual)
test$KitchenQual[test$KitchenQual=="MISSING"] <- "Gd"
test$KitchenQual <- factor(test$KitchenQual)



#Predicting
predicted <- predict(xgb_tree,newdata = test)

summary(predicted)

# Taking inverse of predicted values
predicted_clean <- 10^predicted

summary(predicted_clean)

# Preparing a submission file to kaggle
sub <- read.csv("Data/Sample_Submission.csv", header = T)

# Appending details to sub data frame
sub$Id <- Test_Id
sub$SalePrice <- predicted_clean

#Writing a csv file for submission to kaggle.----
write.csv(sub,"Xgb_19-04-24.csv",row.names = FALSE)
