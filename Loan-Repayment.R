library(randomForest)
library(caTools)
library(e1071)
library(ROCR)

Dataset = read.csv("loan_data.csv")
head(Dataset)
str(Dataset)

# Change some variables data type
Dataset$credit.policy <- factor(Dataset$credit.policy)
Dataset$purpose <- factor(Dataset$purpose)
Dataset$not.fully.paid <- factor(Dataset$not.fully.paid)
Dataset$pub.rec <- factor(Dataset$pub.rec)
Dataset$delinq.2yrs <- factor(Dataset$delinq.2yrs)
Dataset$inq.last.6mths <- factor(Dataset$inq.last.6mths)

#Check if there could be possibility of multicollinearity among numerical variables
CorrChe = Dataset[, 3:10]
cor(CorrChe)

# Data partition into 70% training set and 30% test set 

set.seed(123) 

PartitionPercentage = sample.split(Dataset$not.fully.paid, SplitRatio = .70)
TrainSet = subset(Dataset, PartitionPercentage == TRUE)
TestSet = subset(Dataset, PartitionPercentage == FALSE)

# Training the base Random Forest model

BaseRandomForest <- randomForest(not.fully.paid~., data = TrainSet, importance = TRUE, ntree = 20)
print(BaseRandomForest)
print(BaseRandomForest$importance)

#Model Tuning

OptimalParameters <- tuneRF(TrainSet[-14], TrainSet$not.fully.paid, ntreeTry = 20,stepFactor = 0.8, 
                            improve = 0.05, doBest = FALSE, trace = TRUE, plot = TRUE)
BestParameter <- OptimalParameters[OptimalParameters[, 2] == min(OptimalParameters[, 2]), 1]

#=========================================================================================================================
# Rerun random forest with fine-tuned Optimal Parameters  

FineTuneRandomForest <-randomForest(not.fully.paid~., data = TrainSet, mtry = BestParameter, importance = TRUE, ntree = 20)
print(FineTuneRandomForest)

# Evaluating variable importance
varImpPlot(FineTuneRandomForest)

# Predict probabilities based on the test dataset
BuildingCM = predict(FineTuneRandomForest, TestSet[,-14])
print("The confusion matrix is:")
print(table(Actual = TestSet[,14], Predicted = BuildingCM))

# Creating the ROC curve

# Predict probabilities based on the test dataset

PredictedProbabilities <- prediction(as.vector(as.numeric(BuildingCM)),TestSet$not.fully.paid)
RFPerformance <- performance(PredictedProbabilities, "tpr", "fpr")
plot(RFPerformance)
abline(a = 0, b = 1)

# Calculating and print AUC value
auc <- performance(PredictedProbabilities, measure = "auc")
auc <- auc@y.values[[1]]
print(paste("AUC for the Fine Tune Random Forest is:", auc))

#=============================================================================================================================

# Build an Support Vector Machine Model

#Base Support Vector Machine Model
BaseSVM = svm(not.fully.paid~.,data = TrainSet)
print(summary(BaseSVM))

# Evaluate the Base Support Vector Machine Model

BuildingCMBSVM <- predict(BaseSVM, TestSet[,-14])
print("The confusion matrix is:")
print(table(Actual = TestSet[,14], Predicted = BuildingCMBSVM))

# Creating the ROC curve the Base Support Vector Machine Model

# Predict probabilities based on the test dataset

PredictedProbabilitiesBSVM <- prediction(as.vector(as.numeric(BuildingCMBSVM)),TestSet$not.fully.paid)
BSVMPerformance <- performance(PredictedProbabilitiesBSVM, "tpr", "fpr")
plot(BSVMPerformance)
abline(a = 0,b = 1)

# Calculating and print AUC value the Base Support Vector Machine Model
auc <- performance(PredictedProbabilitiesBSVM, measure = "auc")
auc <- auc@y.values[[1]]
print(paste("AUC for the Base Support Vector Machine Model is:", auc))

#=======================================================================================================
# Choose the linear Kernel function
LinearKSVM <- svm(not.fully.paid~.,kernel='linear',data = TrainSet)
print(summary(LinearSVMK))

# Evaluate the Linear Kernel Support Vector Machine Model

BuildingCMLKSVM <- predict(LinearKSVM, TestSet[,-14])
print("The confusion matrix is:")
print(table(Actual = TestSet[,14], Predicted = BuildingCMLKSVM))

# Creating the ROC curve the Linear Kernel Support Vector Machine Model

# Predict probabilities Linear Kernel on the test dataset

PredictedProbabilitiesLKSVM <- prediction(as.vector(as.numeric(BuildingCMLKSVM)),TestSet$not.fully.paid)
LKSVMPerformance <- performance(PredictedProbabilitiesLKSVM, "tpr", "fpr")
plot(LKSVMPerformance)
abline(a = 0,b = 1)

# Calculating and print AUC value the Linear Kernel Support Vector Machine Model
auc <- performance(PredictedProbabilitiesLKSVM, measure = "auc")
auc <- auc@y.values[[1]]
print(paste("AUC for the Linear Kernel Support Vector Machine Model is:", auc))

#==========================================================================================================

#Model Tuning

OptimalParameters4SVM <- tune(svm, train.x = not.fully.paid~., data = TrainSet, 
                              kernel = 'radial', ranges = list(cost = c(0.5, 5),gamma = c(0.1,1)))
print(summary(OptimalParameters4SVM))

# Improved Support Vector Machine Model based on the tuned result
ImprovedSVM <- svm(not.fully.paid~.,data = TrainSet, kernel='radial', cost = 0.5, gamma = 0.1)
print(summary(ImprovedSVM))

# Evaluate the Improved Support Vector Machine Mode

BuildingCMImprSVM <- predict(ImprovedSVM, TestSet[,-14])
print("The confusion matrix is:")
print(table(Actual = TestSet[,14], Predicted = BuildingCMImprSVM))

# Creating the ROC curve for Improved Support Vector Machine Mode
PredictedProbabilitiesImprSVM <- prediction(as.vector(as.numeric(BuildingCMImprSVM)),TestSet$not.fully.paid)
ImprSVMPerformance <- performance(PredictedProbabilitiesImprSVM, "tpr", "fpr")
plot(ImprSVMPerformance)
abline(a = 0,b = 1)

# Calculate and print AUC value
auc <- performance(PredictedProbabilitiesImprSVM, measure="auc")
auc <- auc@y.values[[1]]
print(paste("AUC for the Improved Support Vector Machine Mode is:", auc))
