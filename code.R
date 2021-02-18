library(e1071)
library(ipred)
library(rpart)
library(randomForest)
data=read.csv('AAPL.csv')
data=data[,c('Date','Open','Close','Volume')]
price=data$Close
change=(price[-1]-price[-length(price)])/price[-length(price)]
data$return=c(change,NA)*100
## assign direction (+/-) to volume
data$Volume=sign(data$Close-data$Open)*data$Volume
###### format data: 20 past volumes (x1-x20) vs return (y)ap=data.frame(matrix(NA,ncol=21,nrow=dim(data)[1]-21))
colnames(ap)[dim(ap)[2]]='y'
for (i in 1:(dim(data)[1]-20)){
  ap[i,'y']=data[i+20,'return']
  ap[i,20:1]=data[i:(i+19),'Volume']
}
#################
################# EDA
#################
vol=data$Volume
hist(vol[vol<quantile(vol, c(0.95)) & vol>quantile(vol,c(0.05))],
     main="Predictors", xlab='Signed Daily Trading Volume',
     col='grey')
ret=na.omit(data$return)
hist(ret[ret<quantile(ret, c(0.95)) & ret>quantile(ret, c(0.05))],
     main='Response Variable', xlab='Daily Return',
     col='grey')
################################################ modeling
########## get training set and testing set
ap=ap[240:1239,] # only use most recent 1000 observations
index=1:700train.c=ap[index,] # training set for continuous responses
test.c=ap[-index,]
############################# continuous response
### variable selection ("forward" selection)
aic=rep(NA,21)
aic[1]=AIC(lm(y~1, data=train.c))
for (i in 1:20){
  aic[i+1]=AIC(lm(y~., data=train.c[,c(1:i,21)]))
}
AIC(lm(y~1, data=train.c))
AIC(lm(y~X1, data=train.c))
AIC(lm(y~X1+X2, data=train.c))
AIC(lm(y~X1+X2+X3, data=train.c))
AIC(lm(y~X1+X2+X3+X4, data=train.c))
AIC(lm(y~X1+X2+X3+X4+X5, data=train.c))
AIC(lm(y~X1+X2+X3+X4+X5+X6, data=train.c))
plot( 0:20,aic, main='AIC values', xlab='# of most recent days to include', ylab='AIC')
# conclusion: past 3 days have predictive power on tomorrow's return###### models and predictions
model.lr=lm(y~X1+X2+X3-1, data=train.c) #no intercept
summary(model.lr)
pred.lr=predict(model.lr, test.c[,1:20])
cor(pred.lr, test.c$y, use='pairwise.complete.obs')
## sign
t=table(test.c$y[-300]>0,pred.lr[-300]>0)
# accuracy
(t[1]+t[4])/sum(t)
#sensitivity
t[4]/(t[2]+t[4])
#specificity
t[1]/(t[1]+t[3])
## MSE
mean(c(pred.lr[-300]-test.c$y[-300])^2)^0.5
#####################
###################### binary response
##########################train.b=train.c
train.b$y=as.numeric(train.b$y>0)
test.b=test.c
test.b$y=as.numeric(test.b$y>0)
################ logistic
model.log=glm(y~X1+X2+X3-1,data=train.b, family=binomial)
summary(model.log)
pred.log=as.numeric(predict(model.log, test.b[,1:3], type='response')>0.5)
t=table(test.b$y, pred.log)
t
################# svm
model.svm=svm(y~X1+X2+X3-1, data=train.b, type='C-classification')
pred.svm=predict(model.svm, test.b)
t=table(test.b$y, pred.svm)
t
################## random forest
rf.fit = randomForest(train.b[,1:3], as.factor(train.b$y), ntree = 500, mtry = 3, nodesize = 2)nfold = 5
infold = sample(rep(1:nfold, length.out=length(train.b$y)))
nd_list=seq(1,201,by=10)
K = length(nd_list)
errorMatrix = matrix(NA, K, nfold)
for (l in 1:nfold)
{
  for (k in 1:K){
    rf.fit = randomForest(train.b[infold!=l,1:3], as.factor(train.b$y[infold!=l]), ntree =
                            500, mtry = 3, nodesize = nd_list[k])
    pred.rf=predict(rf.fit, train.b[infold==l,1:3])
    errorMatrix[k, l] = mean((as.numeric(pred.rf)-1 - as.numeric(train.b$y)[infold ==
                                                                              l])^2)
  }
}
plot(rep(nd_list, nfold), as.vector(errorMatrix), pch = 19, cex = 0.5)
points(nd_list, apply(errorMatrix, 1, mean), col = "red", pch = 19, type = "l", lwd = 3)
which.min(apply(errorMatrix, 1, mean))
rf.fit = randomForest(train.b[,1:3], as.factor(train.b$y), ntree = 500, mtry = 3, nodesize = 91)
pred.rf=predict(rf.fit, test.b[,1:3])
t=table(test.b$y, pred.rf)