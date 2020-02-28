import numpy as np


# calculate objective function value for logistic regression according to sklearn
def objective_function(estimator,X,y,C):
    t_coef = np.transpose(estimator.coef())
    rgl = np.dot(estimator.coef(),t_coef)/2
    cost = C*loss_function(estimator,X,y)
    ans = cost+rgl
    return ans

# calculate loss function value for logistic regression according to sklearn
def loss_function(estimator,X,y):
    t_coef = np.transpose(estimator.coef())
    intercept = estimator.intercept_()
    estimation = np.matmul(X,t_coef)+intercept
    cost = sum(np.log(np.exp(-np.multiply(y,estimation))+1))
    return cost