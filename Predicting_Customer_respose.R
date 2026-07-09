library(caret)
library(randomForest)
library(nnet)

#load and inspect data
marketing <- read.csv(file="marketing_campaign.csv", stringsAsFactors = TRUE)
summary(marketing)

#remove variables that are not useful for prediction
marketing <- marketing[, !(names(marketing) %in% c("ID", "Dt_Customer", "Z_CostContact", "Z_Revenue"))]

#make sure the response variable is a factor
marketing$Response <- as.factor(marketing$Response)

#remove missing values
marketing <- marketing[complete.cases(marketing), ]
summary(marketing)

#check class balance
table(marketing$Response)

#visualize pairs plot of numeric predictors colored by class
numeric_vars <- marketing[, sapply(marketing, is.numeric)]
pairs(numeric_vars, col=marketing$Response)

#partition the data into training and testing using hold out method
#first set the random seed for repeatability
set.seed(4567)

# Create an index variable to perform a 70/30 split
trainIndex <- createDataPartition(marketing$Response, p=.7, list=FALSE, times=1)
marketing_train <- marketing[trainIndex, ]
marketing_test <- marketing[-trainIndex, ]

#logistic Regression using glm
trControl <- trainControl(method = 'none')

glmFit <- train(Response ~ ., 
                data = marketing_train, 
                method = 'glm', 
                family = 'binomial',
                preProcess = c("center","scale"),
                trControl = trControl)

glmPredClass <- predict(glmFit, marketing_test)

#now evaluate the classifier using the confusionMatrix() function
confusionMatrix(glmPredClass, marketing_test$Response, mode="everything")

#random Forest using rf
trControl <- trainControl(method = 'none')

rfFit <- train(Response ~ ., 
               data = marketing_train, 
               method = 'rf', 
               preProcess = c("center","scale"),
               trControl = trControl)

rfPredClass <- predict(rfFit, marketing_test)

#now evaluate the classifier using the confusionMatrix() function
confusionMatrix(rfPredClass, marketing_test$Response, mode="everything")

######################################################################
#neural Network using nnet
######################################################################
# Use cross validation to optimize network parameters
trControl <- trainControl(method = 'cv', number = 10)

nnetFit <- train(Response ~ ., 
                 data = marketing_train, 
                 method = 'nnet', 
                 preProcess = c("center","scale"), 
                 trControl = trControl,
                 trace = FALSE)

# Examine the result of the cross validation
plot(nnetFit)

nnetPredClass <- predict(nnetFit, marketing_test)

# Now evaluate the classifier using the confusionMatrix() function
confusionMatrix(nnetPredClass, marketing_test$Response, mode="everything")