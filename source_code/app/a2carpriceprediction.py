import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib
import pickle
import mlflow

modelname = 'source_code/app/A2CarPricePrediction.model'

# linear regression
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures # Import PolynomialFeatures

class LinearRegression(object):
    
    #in this class, we add cross validation as well for some spicy code....
    kfold = KFold(n_splits=3)
            
    def __init__(self, regularization, theta='zeros', use_momentum=False, momentum=0.9, lr=0.001, method='batch', num_epochs=500, batch_size=50, cv=kfold):
        self.lr         = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method     = method
        self.cv         = cv
        self.regularization = regularization
        self.theta  = theta
        self.use_momentum = use_momentum    #momentum value (0, 1)
        self.momentum = momentum
        self.prev_step=0
        
    # calculate mse
    
    def mse(self, ypred, ytrue):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]
    
    # calculate r2score
    def r2score(self, ypred, ytrue):
        return 1 - ((((ytrue - ypred)**2).sum()) / (((ytrue - ytrue.mean())**2).sum()))
    
    def avgMse(self): 
        return np.sum(np.array(self.kfold_scores_mse))/len(self.kfold_scores_mse)
    """
    Calculate the R-squared (R2) score.

    Parameters:
    ytrue: actual_labels_values
    ypred: predicted_labels_values

    Returns:
    float: The R2 score.
    """
    
    # train test split
    def fit(self, X_train, y_train):
            
        #create a list of kfold scores
        self.kfold_scores_mse = list()
        self.kfold_scores_r2score = list()
        
        #reset val loss
        self.val_loss_old = np.infty
        
        #kfold.split in the sklearn.....
        #3 splits
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            
            # --- choose theta between zeros initialization or xavier ---
            # set theta by xavier method
            if self.theta[0] == 'xavier':
                # for xavier weight initialization
                m = X_cross_train.shape[0]    # define number of samples according to the train dataset
                print(f"number of sample:{m}")
                
                # calculate the range for the weights
                lower , upper = -(1.0 / np.sqrt(m)), (1.0 / np.sqrt(m))
                # summarize the range
                print(lower , upper)
                
                # randomly pick weights within this range
                numbers = np.random.rand(X_cross_train.shape[1])     # generate random numbers according to the number of selected features
                scaled = lower + numbers * (upper - lower)
                print(scaled)
                self.theta = scaled
                
            # set theta by zeros
            else :
                self.theta = np.zeros(X_cross_train.shape[1])
            
            #define X_cross_train as only a subset of the data
            #how big is this subset?  => mini-batch size ==> 50
            
            #one epoch will exhaust the WHOLE training set
            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                params = {"method": self.method, "lr": self.lr, "reg": type(self).__name__}
                mlflow.log_params(params=params)
                
                for epoch in range(self.num_epochs):
                
                    #with replacement or no replacement
                    #with replacement means just randomize
                    #with no replacement means 0:50, 51:100, 101:150, ......300:323
                    #shuffle your index
                    perm = np.random.permutation(X_cross_train.shape[0])
                            
                    X_cross_train = X_cross_train[perm]
                    y_cross_train = y_cross_train[perm]
                    
                    # stochastic
                    if self.method == 'sto':
                        for batch_idx in range(X_cross_train.shape[0]):
                            X_method_train = X_cross_train[batch_idx].reshape(1, -1) #(11,) ==> (1, 11) ==> (m, n)
                            y_method_train = y_cross_train[batch_idx] 
                            # train_loss = self._train(X_method_train, y_method_train)
                            mse_loss, r2score_loss = self._train(X_method_train, y_method_train)
                            
                    # minibatch
                    elif self.method == 'mini':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            #batch_idx = 0, 50, 100, 150
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            # train_loss = self._train(X_method_train, y_method_train)
                            mse_loss, r2score_loss = self._train(X_method_train, y_method_train)
                    
                    # batch
                    else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        # train_loss = self._train(X_method_train, y_method_train)
                        mse_loss, r2score_loss = self._train(X_method_train, y_method_train)
                    
                    # record mse and r2score for each epoch
                    mlflow.log_metric(key="train_mse_loss", value=mse_loss, step=epoch)
                    mlflow.log_metric(key="train_r2score_loss", value=r2score_loss, step=epoch)

                    # predict for each epoch
                    yhat_val = self.predict(X_cross_val)
                    
                    # mse_epoch
                    val_loss_new_mse = self.mse(y_cross_val, yhat_val)
                    mlflow.log_metric(key="val_loss_mse", value=val_loss_new_mse, step=epoch)
                    # r2score_epoch
                    val_loss_new_r2score = self.r2score(y_cross_val, yhat_val)
                    mlflow.log_metric(key="val_loss_r2score", value=val_loss_new_r2score, step=epoch)
                    
                    #record dataset
                    # mlflow_train_data = mlflow.data.from_numpy(features=X_method_train, targets=y_method_train)
                    # mlflow.log_input(mlflow_train_data, context="training")
                    
                    # mlflow_val_data = mlflow.data.from_numpy(features=X_cross_val, targets=y_cross_val)
                    # mlflow.log_input(mlflow_val_data, context="validation")
                    
                    #early stopping
                    if np.allclose(val_loss_new_mse, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new_mse
                    
                # record fold mse score
                self.kfold_scores_mse.append(val_loss_new_mse)
                print(f"Fold {fold} mse: {val_loss_new_mse}")
                
                self.kfold_scores_r2score.append(val_loss_new_r2score)
                print(f"Fold {fold} r2score: {val_loss_new_r2score}")
            
                    
    def _train(self, X, y):
        yhat = self.predict(X)
        m    = X.shape[0]      
        
        if self.regularization is None:
            grad = (1 / m) * X.T @ (yhat - y)
        else:  
            grad = (1/m) * X.T @(yhat - y) + self.regularization.derivation(self.theta)
        step = self.lr * grad
        # self.theta = self.theta - self.lr * grad
        
        if self.use_momentum:   # if momentumm is used
            self.theta = self.theta - step + self.momentum * self.prev_step
            self.prev_step = step
        
        else:   # if momentum is not used
            self.theta = self.theta - step
            
        mse_loss = self.mse(yhat, y)
        r2_score = self.r2score(yhat, y)
        return mse_loss, r2_score  # Return both MSE and R2
    
    def predict(self, X):
        return X @ self.theta  #===>(m, n) @ (n, )
    
    def _coef(self):
        return self.theta[1:]  #remind that theta is (w0, w1, w2, w3, w4.....wn)
                               #w0 is the bias or the intercept
                               #w1....wn are the weights / coefficients / theta
    def _bias(self):
        return self.theta[0]
    
    # ------Modify the plot_feature_importance method to use _coef
    def plot_feature_importance(self, X):
        if not hasattr(self, 'theta'):
            raise ValueError("Model not trained yet. Call 'fit' method first.")

        if X.shape[1] != self.theta.shape[0]:
            raise ValueError("Number of features in X must match the number of coefficients.")

        # Get the column names (feature names) from the DataFrame
        feature_names = X.columns.tolist()

        # Calculate the absolute magnitude of coefficients as feature importance
        feature_importance = np.abs(self.theta.squeeze())

        # Create a horizontal bar plot to display feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, feature_importance)
        plt.xlabel('Feature Importance (Absolute Magnitude of Coefficients)')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top
        plt.show()
# -------------------------------
class LassoPenalty:
    
    def __init__(self, l):
        self.l = l # lambda value
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)
    
class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
    
class ElasticPenalty:
    
    def __init__(self, l = 0.1, l_ratio = 0.5):
        self.l = l 
        self.l_ratio = l_ratio

    def __call__(self, theta):  #__call__ allows us to call class as method
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return (l1_contribution + l2_contribution)

    def derivation(self, theta):
        l1_derivation = self.l * self.l_ratio * np.sign(theta)
        l2_derivation = self.l * (1 - self.l_ratio) * theta
        return (l1_derivation + l2_derivation)
    
class Lasso(LinearRegression):
    
    def __init__(self, method,theta, use_momentum, momentum, lr, l):
        self.regularization = LassoPenalty(l)
        super().__init__(self.regularization, theta, use_momentum, momentum,lr, method)
        
class Ridge(LinearRegression):
    
    def __init__(self, method, theta, use_momentum, momentum,lr, l):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, theta, use_momentum, momentum,lr, method)
        
class ElasticNet(LinearRegression):
    
    def __init__(self, method, theta, use_momentum, momentum, lr, l, l_ratio=0.5):
        self.regularization = ElasticPenalty(l, l_ratio)
        # def __init__(self, regularization, theta='zeros', use_momentum=False, momentum=0.9, lr=0.001, method='batch', num_epochs=500, batch_size=50, cv=kfold):
        super().__init__(self.regularization, theta, use_momentum,momentum, lr, method )

# -------------------------------

def fn_a2_predict(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,3)
    
    #loaded model
    pickle_model = pickle.load(open(modelname, 'rb'))
    
    #take model and scaler
    model = pickle_model['model']
    scaler = pickle_model['scaler']
    
    print("loaded a2 model")
    
    #scale the value received
    to_predict = scaler.transform(to_predict)
    
    #predict the result
    result = model.predict(to_predict)
    return np.exp(result[0])
    # return result[0]
    
print("*****  successfully called a2 car price prediction and parse the predected value *****")