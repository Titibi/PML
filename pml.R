if(!file.exists("pml_train.csv")){
  url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
  download.file(url,"pml_train.csv")
}

if(!file.exists("pml_test.csv")){
  url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
  download.file(url,"pml_test.csv")
}

library(caret)
library(dplyr)
library(kernlab)
library(RANN)
library(parallel)
library(doParallel)

set.seed(11223344)

train <- read.csv("pml_train.csv")
## Creatin testset and trainingset from training data

ind <- createDataPartition(train$classe, p = .80, list=FALSE)
trainset <- train[ind,]
testset <- train[-ind,]

## Checking for percentage NAs in each column/vector of trainset 
## which are more than 90%

trainNAs <- sapply(trainset[,-160],function(x)(sum(is.na(x))/length(x))*100)

## Removing columns with more than 90% NAs from testset and trainset on basis of trainset findings.

trainset <- trainset[,trainNAs<90]
testset <- testset[,trainNAs<90]

## Checking for near zero variance vectors in trainset and removing them

trainset_nzv <- nearZeroVar(trainset[,-93])
trainset <- trainset[,-trainset_nzv]
testset <- testset[,-trainset_nzv]

##removing first 5 columns from training set as they are not required for this analysis
trainset <- trainset[,-(1:5)]

## Principle component analysis on trainset further shrinks the variables/features
## to 37 on training data set, and capturing 99% of data

pca <- preProcess(trainset[,-54],method = "pca",thresh = .99)
trainpca<- predict(pca,trainset[])

## Model build/train 

### I will be training using random forest, rpart and ada

cluster <- makeCluster(detectCores() - 1) 
registerDoParallel(cluster)
trainControl <- trainControl(method = "repeatedcv",number =10, repeats = 5, allowParallel = TRUE)

### Random Forest
mod_rf <- train(classe ~.,data=trainpca,method = "rf",trControl = trainControl)

### Rpart
mod_rpart <- train(classe ~., data = trainpca, method = "rpart",trControl = trainControl)

### gbm
mod_gbm <- train(classe ~., data = trainpca, method = "gbm",trControl = trainControl)

stopCluster(cluster)
registerDoSEQ()

## Checking accuracy of each model on testset

### Preprocess testset on generated pca for trainingset

testpca <- predict(pca,testset[])

### Predicting through random forest and generating confusion matrix to verify accuracy

predict_rf <- predict(mod_rf,testpca)
cm_rf <- confusionMatrix(predict_rf,testpca$classe)
cm_rf
                   
### Predicting through Recursive Partitioning and Regression Trees and generating confusion matrix to verify accuracy

predict_rpart <- predict(mod_rpart,testpca)
cm_rpart <- confusionMatrix(predict_rpart,testpca$classe)
cm_rpart
                   
### Predicting through Generalized Boosted Regression and generating confusion matrix to verify accuracy

predict_gbm <- predict(mod_gbm,testpca)
cm_gbm<- confusionMatrix(predict_gbm,testpca$classe)
cm_gbm
                   
## The generated confusion matrix reveal that out of the 3 models, random forest has the highest prediction accuracy of 98%.
## Hence this model will be used to predict on the test set of the problem

