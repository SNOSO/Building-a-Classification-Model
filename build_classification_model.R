#load the data
library(datasets)
data("iris")

iris <- datasets::iris

# check to see if there is any missing data
sum(is.na(iris))

# skimr() expands on summary() by providing larger set of statistics
library(skimr)

#perform skim to display summary statistics
skim(iris)

# group data by Species then perform skim
iris %>%
  dplyr::group_by(Species) %>%
  skim()

############################
# quick data visualization
#
# R base plot()
############################

# panel plots
plot(iris)
plot(iris, col = "red")

# scatter plot
plot(iris$Sepal.Width, iris$Sepal.Length)

plot(iris$Sepal.Width, iris$Sepal.Length, col = "red") # makes red circles

plot(iris$Sepal.Width, iris$Sepal.Length, col = "red", 
     xlab = "Sepal width", ylab ="Sepal length")

#Histogram
hist(iris$Sepal.Width)
hist(iris$Sepal.Width, col = "red") # makes red bars
hist(iris$Sepal.Width, col = "blue",
     main = "Histogram of Sepal Width", xlab = "Sepal Width")

library(caret)

# feature plots
featurePlot(x = iris[,1:4],
            y = iris$Species,
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"),
                          y = list(relation = "free")))


###############################

# assign a seed number to a fixed number so when you rerun same model you will get the same results each time code is ran
set.seed(100)

# data splitting - want to simulate a situation where we have a dataset which we use to train the model and want to see if model will be applicable to future data
# training and testing
# perfroms stratified random split of the data set
TrainingIndex <- createDataPartition(iris$Species, p=0.8, list = FALSE)
TrainingSet <- iris[TrainingIndex,] # Training set
TestingSet <- iris[-TrainingIndex,] # Test set

# compare scatter plot of the 80/20 data subsets

plot(TestingSet, main= "Testing Set")
plot(TrainingSet, main= "Training Set")

#####################
#SVM model (polynomial kernel)

# build training model- uses training set to build the model (80% subset)
Model <- train(Species ~ ., data = TrainingSet,
               method = "svmPoly",
               na.action = na.omit, #omits any na data
               preProcess=c("scale","center"), #preprocess data according to mean centering
               trControl= trainControl(method="none"),
               tuneGrid = data.frame(degree=1,scale=1,C=1))

# build CV model - uses training model to predict class label in testing set (20% subset)
Model.cv <- train(Species ~ ., data = TrainingSet,
                  method = "svmPoly",
                  na.action = na.omit,
                  preProcess=c("scale","center"),
                  trControl= trainControl(method="cv",number=10),
                  tuneGrid = data.frame(degree=1,scale=1,C=1))

# Apply model for prediction
Model.training <-predict(Model, TrainingSet) # Apply model to make prediction on Training set
Model.testing <-predict(Model, TestingSet) # Apply model to make prediction on Testing set
Model.cv <-predict(Model.cv, TrainingSet) # Perform cross-validation

# Model performance (Displays confusion matrix and statistics)
Model.training.confusion <-confusionMatrix(Model.training, TrainingSet$Species)
Model.testing.confusion <-confusionMatrix(Model.testing, TestingSet$Species)
Model.cv.confusion <-confusionMatrix(Model.cv, TrainingSet$Species)

print(Model.training.confusion)
print(Model.testing.confusion)
print(Model.cv.confusion)

# Feature importance
Importance <- varImp(Model)
plot(Importance)
plot(Importance, col = "red")
