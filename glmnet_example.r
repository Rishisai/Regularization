rm(list = ls())
set.seed(1000)

library(mlbench)
library(caret)

library(glmnet)
library(glmnetUtils)
library(plotmo)



data("BostonHousing")

Train_Index=sample(nrow(BostonHousing),round(nrow(BostonHousing)*.8))

TrainData=BostonHousing[Train_Index,]
TestData=BostonHousing[-Train_Index,]

###########Trying Linear Regression ################
lm_model=lm(medv~.,data=TrainData)
predicts_lm=predict(lm_model,newdata=TestData)
MSE_lm=mean(sqrt((predicts_lm-TestData$medv)^2))
print(MSE_lm)
anova(lm_model)

#Using caret
lm_model_caret<-train(medv~.,data=TrainData, method='lm')
predicts_lm_caret=predict(lm_model_caret,newdata=TestData)
MSE_lm_caret=mean(sqrt((predicts_lm_caret-TestData$medv)^2))
print(MSE_lm_caret)
varImp(lm_model_caret)


###########Trying Linear Regression ################
proc<-preProcess(BostonHousing, method = "range")
BostonHousing_scaled<-predict(proc,BostonHousing)          

BostonHousing_scaled$medv<-BostonHousing$medv          
summary(BostonHousing_scaled)
TrainData=BostonHousing_scaled[Train_Index,]
TestData=BostonHousing_scaled[-Train_Index,]

############## The performance of normal lm() would be exactly the same ####
############# Scaling would not have any impact ###################

lm_model=lm(medv~.,data=TrainData)
predicts_lm=predict(lm_model,newdata=TestData)
MSE_lm=mean(sqrt((predicts_lm-TestData$medv)^2))
print(MSE_lm)
lm_model$coefficients
sum(abs(lm_model$coefficients)) #L1 norm of coefficients

############## Let us try ridge and lasso now ####
#lets start with Ridge first

fit = glmnet(medv~.,data=TrainData, alpha = 0, nlambda =100)
plot(fit)
print(fit)

cvfit = cv.glmnet(medv~.,data=TrainData, alpha = 0, nlambda =100)
plot(cvfit)
cvfit$lambda.min
coef(cvfit, s = "lambda.min")

predicts_glm_ridge<-predict(cvfit, newdata = TestData, s = "lambda.min")
MSE_glm_ridge=mean(sqrt((predicts_glm_ridge-TestData$medv)^2))
print(MSE_glm_ridge)


##############################################################
#lets try lasso now 

cvfit = cv.glmnet(medv~.,data=TrainData, alpha = 1, nlambda =100)
plot(cvfit)
cvfit$lambda.min
coef(cvfit, s = "lambda.min")

predicts_glm_lasso<-predict(cvfit, newdata = TestData, s = "lambda.min")
MSE_glm_lasso=mean(sqrt((predicts_glm_lasso-TestData$medv)^2))
print(MSE_glm_lasso)

##############################################################
#lets elastic net 
# This also search for the best alpha value 
cvfit = cva.glmnet(medv~.,data=TrainData,  nlambda =100)
minlossplot(cvfit)

