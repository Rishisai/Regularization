library(glmnet)
library(caret)
library(psych)
library(glmnetUtils)


# load data
data(cars)
pairs.panels(cars)

# lm
fit.lm <- lm(Price ~ ., data = cars)
summary(fit.lm)


# lasso
fit.glmnet.lasso <- glmnet(as.matrix(cars[, -1]),
                           as.matrix(cars[, 1]),
                           alpha = 1)
plot(fit.glmnet.lasso)
## cv
fit.glmnet.lasso.cv <- cv.glmnet(as.matrix(cars[, -1]),
                                 as.matrix(cars[, 1]),
                                 nfold = 5,
                                 alpha = 1)
plot(fit.glmnet.lasso.cv)
fit.glmnet.lasso.cv$lambda.min
coef(fit.glmnet.lasso.cv, s = fit.glmnet.lasso.cv$lambda.min)

# ridge
fit.glmnet.ridge <- glmnet(as.matrix(cars[, -1]),
                           as.matrix(cars[, 1]),
                           alpha = 0)
plot(fit.glmnet.ridge)
## cv
fit.glmnet.ridge.cv <- cv.glmnet(as.matrix(cars[, -1]),
                                 as.matrix(cars[, 1]),
                                 nfold = 5,
                                 alpha = 0)
plot(fit.glmnet.ridge.cv)
fit.glmnet.ridge.cv$lambda.min
coef(fit.glmnet.ridge.cv, s = fit.glmnet.ridge.cv$lambda.min)

# ElasticNet
fit.glmnet.elasticnet.cv <- cv.glmnet(as.matrix(cars[, -1]),
                                      as.matrix(cars[, 1]),
                                      nfold = 5,
                                      alpha = 0.5)
plot(fit.glmnet.elasticnet.cv)
fit.glmnet.elasticnet.cv$lambda.min
coef(fit.glmnet.elasticnet.cv, s = fit.glmnet.elasticnet.cv$lambda.min)
####

cvfit = cva.glmnet(as.matrix(cars[, -1]),
                   as.matrix(cars[, 1]),
                   nlambda =100)
minlossplot(cvfit)
