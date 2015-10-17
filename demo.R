# --- Clear Workspace ---
rm(list=ls())

# --- Change Working Directory ---
# setwd("")

# --- Libraries ---
library('mlbench')       # Classification Datasets
library("e1071")         # Naive-Bayes and SVM tuning
library('kernlab')       # SVM
library('randomForest')  # Random Forest
library('adabag')        # Adaboost
library("corpcor")       # Shrinkage Covariance Estimator
library("pROC")          # Compute AUC 
library('Rcpp')          # C++ functions
library('RcppArmadillo') # C++ Linear Algebra

# --- External Functions ---
source('normInputData.R')
source('regGaussianBayes.R')
sourceCpp('fastMVNDensity.cpp')       # PS: use mvnDensity instead for a R only implementation 
# sourceCpp('parallelMVNDensity.cpp') # PS: for Ubuntu users only

# --- Load Dataset ---
# -- Dataset 1: Vehicle --
# data('Vehicle')
# x <- data.matrix(Vehicle[,-19]);
# y <- Vehicle$Class;
# rm(list='Vehicle')
# -- Dataset 2: Iris --
x <- data.matrix(iris[,-5]);
y <- iris$Species;

# --- Data Column-wise normalization ---
x <- normInputData(x,type='std');

# --- Shuffle observations ---
new.order <- sample(nrow(x));
x <- x[new.order,];
y <- y[new.order];
   
# --- Split in training and test sets ---
data <- splitTrainTest(x,y,test.percent=0.25);
x.train <- data$x.tr;
y.train <- data$y.tr;
x.test  <- data$x.te;
y.test  <- data$y.te;

# --- Regularized Gaussian Bayes ---
rgb.time <- system.time({
   rgb.model <- rgbc(x.train,y.train);
});
rgb.perf <- predict(rgb.model,x.test,y.test);

# --- Naive-Bayes ---
gnb.time <- system.time({
   gnb.model <- naiveBayes(x.train,y.train);
});
gnb.resp <- predict(gnb.model,newdata=x.test);
gnb.perf <- computeClassPerformance(as.numeric(y.test),as.numeric(gnb.resp));
            
# --- RBF SVM ---
svm.time <- system.time({
   svm.sigma  <- median(do.call(c,lapply(1:100,function(i,x,scaled) sigest(x=x,scaled=scaled)[2],x=x.train,scaled=FALSE)));
   svm.params <- tune.svm(x.train, as.factor(y.train), type='C-classification', kernel='radial', gamma=svm.sigma,
                          cost=2^(-5:5), tunecontrol=tune.control(sampling='cross',cross=5))$best.param; # params: [1] sigma [2] cost
   svm.model <- ksvm(x.train, y.train, scaled=FALSE, type="C-svc", C=svm.params[1,2], kernel="rbfdot",
                     kpar=list(sigma=svm.params[1,1]), cross=0);
});
svm.resp <- predict(object=svm.model, newdata=x.test);
svm.perf <- computeClassPerformance(as.numeric(y.test),as.numeric(svm.resp));
            
# --- Random Forest ---   
rf.time <- system.time({
   rf.model <- randomForest(x=x.train, y=as.factor(y.train), xtest=x.test, ytest=as.factor(y.test), ntree=100);			
});
rf.resp <- rf.model$test$predicted;
rf.perf <- computeClassPerformance(as.numeric(y.test),as.numeric(rf.resp));
            
# --- Adaboost ---   
dt.train <- data.frame(x.train,y=y.train);
dt.test  <- data.frame(x.test,y=y.test);
ada.time <- system.time({
   ada.model <- boosting(formula=y~., data=dt.train, boos=FALSE, mfinal=30);
})
ada.resp <- as.factor( predict(object=ada.model, newdata=dt.test, type='class')$class );
ada.perf <- computeClassPerformance(as.numeric(y.test),as.numeric(ada.resp));

# --- Print Results ---
algs.names <- c('rgb','gnb','svm','rf','ada');
acc  <- c(rgb.perf$acc, gnb.perf$acc, svm.perf$acc, rf.perf$acc, ada.perf$acc);
auc  <- c(rgb.perf$auc, gnb.perf$auc, svm.perf$auc, rf.perf$auc, ada.perf$auc);
time <- c(rgb.time[3], gnb.time[3], svm.time[3], rf.time[3], ada.time[3]);
results <- data.frame(Accuracy=acc, AUC=auc, Time=time, Algorithm=as.factor(algs.names));
print(results)