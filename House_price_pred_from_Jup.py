
# coding: utf-8
#@2018 Johannes Weirich
#Kaggle House Price prediction challenge
# In[1]:
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet,Lasso,BayesianRidge,LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.model_selection import KFold,cross_val_score
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import Imputer,RobustScaler,StandardScaler,Normalizer
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

import xgboost as xgb
import numpy as np

import seaborn as sns

import scipy.stats as stats
from scipy.special import boxcox1p


# In[2]:

def remove_na(df):
  '''
  Simple way to remove NAN both for numerical and categorical features
  NAN values for categroical features are replaced by the value with highest occurence, NAN values in numerical   features are replaced by the mean. 
  Parameters:
  ------------
  df: Pandas DataFrame containing all features
  
  Returns:
  -----------
  df: Pandas DataFrame with NAN values replaced 
  '''
  fill=pd.Series([df[col].value_counts().index[0] 
       if df[col].dtype==np.dtype('O') 
       else df[col].mean() for col in df])
  #print(fill)
  for idx_col,col in enumerate(df):
    df[col]=df[col].fillna(fill[idx_col])
  return df

def rsquared(y,pred):
  '''
  Function calculating R-squared for actual y and fitted value (pred)
  '''
  ybar=y.mean()
  SStot=((y-ybar)**2).sum()
  SSres=((y-pred)**2).sum()
  return 1-SSres/SStot

#TODO data_numeric=pd.get_dummies(df) will do the whole trick
def transform_to_numeric(df):
  '''
  Encoding of categroical features with dummies. 
  Parameters:
  ------------
  df: Pandas DataFrame containing all features
  
  Returns:
  -----------
  df: Pandas DataFrame with categorical features transformed to one/zero 
  '''

  data_numeric=pd.DataFrame()
  #objs=0
  for ctr,col in enumerate(df.columns):
    if df[col].dtype=='O':
      dummies=pd.get_dummies(df[col],prefix=col)
      #objs+=1
      if data_numeric.shape[0]<1:
        data_numeric=data_numeric.join(dummies,how=outer)
      else:
        data_numeric=data_numeric.join(dummies)

      #print('adding dummies for category number') 
      #time.sleep(5) 
    
    else:
      if data_numeric.shape[0]<1:
        data_numeric[col]=df[col]
      else:
        data_numeric=data_numeric.join(df[col]) #join on index
    
      #print('adding numerical column') 
      #time.sleep(5) 
  
  return data_numeric

def encode_cat_codes(df,col,cats=['Po','Fa','TA','Gd','Ex']):
  '''
  Encoding of ordered categorical features using cat.codes 
  '''
  df[col]=df[col].astype('category',ordered=True,categories=cats).cat.codes
  return df

def plot_gallery(df,target_col,feature_cols,N_cols=4):
  '''
  gallery plotting of target variable vs ind. features in a dataframe
  Parameters:
  ------------
  df: Dataframe
  target_col: string, dependent/target column name, acting as y values in plots
  feature_cols: list of strings, feature column names that should be used as x values in plots
  N_cols: Number of columns in Gallery
  
  Returns: None
  -------
  ''' 
  print(target_col)
  print(feature_cols)
  N_rows=int(len(feature_cols)/N_cols)+1
  fig,ax=plt.subplots(N_rows,N_cols)
  corrmat=df.corr()
  ctr=0
  for idx_row in range(N_rows):
    for idx_col in range(N_cols):
      if ctr<len(feature_cols):	
        ax_curr=ax[idx_row,idx_col]
        ax_curr.plot(df[feature_cols[ctr]],df[target_col],'o')
        ax_curr.set_xlabel(feature_cols[ctr])
        ax_curr.set_ylabel(target_col)
        ax_curr.set_title('Correlation with '+target_col+'='
                           +str(round(corrmat.loc[target_col,feature_cols[ctr]],2)))
  
      ctr+=1
  fig.subplots_adjust(hspace=1.0)
  plt.show()

def rmsle_cv(model,X_train,y_train,n_folds=5):
  '''
  function for computing root mean squared error using cross_val_score on number of folds
  Parameters:
  -----------
  model: Estimator for which RMSE should be calculated
  X_train,y_train: train features and target
  n_folds: Number of folds for which CV score is to be calculated
  
  Returns:
  ---------
  rmse: list with RMSE score for folds 
  '''
  kf=KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)
  rmse=np.sqrt(-cross_val_score(model,X_train.values,y_train,scoring='neg_mean_squared_error',cv=kf))
  return rmse

def gridsearch_fit(model,X,Y,grid,scaler=None,score='neg_mean_squared_error',cv=3,verbose=3):
  '''
  function collecting Pipeline, GridSearchCV, fit
  Parameters:
  -----------
  model: Estimator
  X,Y: train features, target
  grid: struct for gridsearch, note that if scaler is used, it should be list of  dicts, with first dict 
  for Scaler and second for Estimator. These should be 
  used as prefix [{'Scaler__var1': Srange1},{'Estimator__var1': Erange1}]
  Returns: 
  ---------------
  grd: trained grid  
  '''
  if scaler:
    steps=[('Scaler',scaler),('Estimator',model)]
    pipe=Pipeline(steps) #pipeline implements fit and transform methods for steps
    grd=GridSearchCV(pipe,grid,scoring=score,cv=cv,verbose=verbose)
  else:
    grd=GridSearchCV(model,grid,scoring=score,cv=cv,verbose=verbose)

  grd.fit(X,Y)

  #print(RF_gridsearch.cv_results_['params'])
  print('Mean scores',grd.cv_results_['mean_test_score'])
  print('Sd scores',grd.cv_results_['std_test_score'])

  return grd


# In[3]:


def gridsearch_cv_two_parameters(estimator,X,Y,grid,scoring='neg_mean_squared_error',\
				verbose=3,plot_results=True):
    '''
    function performing GridSearchCV on two parameters at the time, including visualization

    Parameters:
    -----------
    estimator: Estimator
    X,Y: train features, target
    grid: struct for gridsearch 

    Returns: 
    ---------------
    estimator_gridsearch: trained grid  
    '''

    estimator_gridsearch=GridSearchCV(estimator,grid,
                         scoring=scoring,verbose=verbose)
    estimator_gridsearch.fit(X,Y)
    print('Mean scores',np.sqrt(-estimator_gridsearch.cv_results_['mean_test_score']))
    print('Sd scores',np.sqrt(estimator_gridsearch.cv_results_['std_test_score']))
    
    if plot_results:
        grid_keys=list(estimator_gridsearch.param_grid.keys())
        vals_dim0=estimator_gridsearch.param_grid[grid_keys[0]]
        len_dim0=len(vals_dim0)
        print('vals dim0={}'.format(vals_dim0))
        vals_dim1=estimator_gridsearch.param_grid[grid_keys[1]]
    
        fig,ax=plt.subplots()
        for idx in range(len(estimator_gridsearch.param_grid[grid_keys[1]])):
            ax.plot(vals_dim0,np.sqrt(-estimator_gridsearch.cv_results_['mean_test_score'])[(idx*len_dim0):(idx+1)*len_dim0],label=grid_keys[1]+'='+str(vals_dim1[idx]))
        ax.set_xlabel(grid_keys[0])
        ax.set_ylabel('mean_test_score')
        plt.legend()
    
    return estimator_gridsearch
    


# In[4]:


class AvgModels(BaseEstimator, RegressorMixin, TransformerMixin):
  '''
  Class for averaging of various estimators
  '''
  def __init__(self,models):
    self.models=models #is list of model objects, should not be a clone here
	
  def fit(self,X,y):
    self.models_=[clone(model) for model in self.models] #list of cloned models, i.e. same parameters but not fitted.
    for model in self.models_:
      model.fit(X,y)
    return self
  
  def predict(self,X):
    self.predictions=np.column_stack([model.predict(X) for model in self.models_])
    return self.predictions.mean(axis=1)

class StackingAvgModels(BaseEstimator, RegressorMixin, TransformerMixin):
  '''
  Class for stacking of estimators in order to create a meta learner
  '''
  def __init__(self,models,meta_model,n_folds=5):
    self.models=models
    self.meta_model=meta_model
    self.n_folds=n_folds

  def fit(self,X,y):
    self.meta_model_=clone(self.meta_model)
    kfold=KFold(n_splits=self.n_folds,shuffle=True,random_state=77)

    out_of_fold_predictions=np.zeros([X.shape[0],len(self.models)])
    self.models_=[list() for model in self.models]
    for idx,model in enumerate(self.models):
      for train_index,holdout_index in kfold.split(X,y):
        model_=clone(model)
        self.models_[idx].append(model_) #add 'fold' model to idx'th list i.e. we have n_fold many trained models for each classifier 
        #print('Xshape',X.shape)
        model_.fit(X[train_index],y[train_index])
        pred=model_.predict(X[holdout_index])
        out_of_fold_predictions[holdout_index,idx]=pred
    
    self.meta_model_.fit(out_of_fold_predictions,y)
    return self

  def predict(self, X):
    #self.models_ is list of list of models trained on different folds
    meta_features=np.column_stack([np.column_stack([model.predict(X) for model in models]).mean(axis=1) for models in self.models_])
    return self.meta_model_.predict(meta_features) 
    
# In[5]:

#################################################
#End function definition/ Start of mainprogram
#################################################
train_data=pd.read_csv('train.csv',index_col=0)
test_data=pd.read_csv('test.csv',index_col=0)

###################################################
#Correlation analysis for purely numerical features
###################################################
cm=train_data.corr()
corr_with_saleprice=cm.loc['SalePrice',:].sort_values(ascending=False) #Find correlation of Salesprice with all features, in sorted order. 
relevant_columns=corr_with_saleprice[1:10].index
relevant_columns2=corr_with_saleprice[11:20].index
#plot SalePrice vs. N features with highest correlation

####################################################
#plot SalePrice vs. features with highest correlation
#####################################################
plt.ion() #interactive on
plot_gallery(train_data,'SalePrice',relevant_columns)
plot_gallery(train_data,'SalePrice',relevant_columns2)


# In[6]:


#################################################################
#Oulier removal!
#################################################################
#TODO FIND SYSTEMATIC WAY FOR OUTLIER DETECTION
#Based on investigation of plots SalePrice vs features with highest corr. to SalePrice
drop_idx=train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<300000)].index
drop_idx=drop_idx.append(train_data[train_data['GarageCars']==4].index)
drop_idx=drop_idx.append(train_data[train_data['TotalBsmtSF']>5000].index)
drop_idx=drop_idx.append(train_data[train_data['1stFlrSF']>4000].index)
drop_idx=drop_idx.append(train_data[train_data['BsmtFinSF1']>4000].index)
drop_idx=drop_idx.append(train_data[train_data['LotFrontage']>250].index)
train_data=train_data.drop(index=drop_idx.unique())
#plot_gallery(train_data,'SalePrice',relevant_columns)
#plot_gallery(train_data,'SalePrice',relevant_columns2)

############################################
#Remove nan values
##############################################
#Visualize NAs
#train data NA
cols_with_NA_train=train_data.isna().any()[train_data.isna().any()].index
NA_qty_train=train_data.loc[:,cols_with_NA_train].isna().sum().sort_values(ascending=False)
plt.figure()
ax=sns.barplot(x=NA_qty_train.index,y=NA_qty_train.values/train_data.shape[0])
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)

#test data
cols_with_NA_test=test_data.isna().any()[test_data.isna().any()].index
NA_qty_test=test_data.loc[:,cols_with_NA_test].isna().sum().sort_values(ascending=False)

#Rename NA into 'None', if NA means Not existing, which is case for most categorical features 
#cols=['FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','BsmtFinType1','BsmtFinType2','BsmtCond','BsmtExposure','BsmtQual']
for col in cols_with_NA_train:
  if train_data[col].dtype==np.dtype('O'):
    train_data[col]=train_data[col].fillna('None') 

for col in cols_with_NA_test:
  if test_data[col].dtype==np.dtype('O'):
    test_data[col]=test_data[col].fillna('None')

#Drop columns with more than 50% NA values in traindata
train_data.drop(columns=NA_qty_train[NA_qty_train/train_data.shape[0]>0.5].index)
test_data.drop(columns=NA_qty_train[NA_qty_train/train_data.shape[0]>0.5].index)


train_data=remove_na(train_data) #remove NA values by using avg (for numeric)
test_data=remove_na(test_data) # and value which is most represented(for cat) 

#####################################################################
#Ordered categorical features using astype('categorical').cat.codes
#####################################################################
ord_cat_cols=['ExterQual','ExterCond','BsmtExposure','HeatingQC','KitchenQual']
ord_cat_cols_with_nos=['BsmtQual','BsmtCond','FireplaceQu','GarageQual','GarageCond'] #categorical with 'Nones'

for col in ord_cat_cols:
  train_data=encode_cat_codes(train_data,col) #TODO this is outdated way to do it, CategoricalDtype has to be passed.
  test_data=encode_cat_codes(test_data,col)

for col in ord_cat_cols_with_nos:
  train_data=encode_cat_codes(train_data,col,cats=['None','Po','Fa','TA','Gd','Ex']) #TODO this is outdated way to do it, CategoricalDtype has to be passed.
  test_data=encode_cat_codes(test_data,col,cats=['None','Po','Fa','TA','Gd','Ex'])

########################################################################
#Transformation of not ordererd categorical features to numeric (dummies)
########################################################################
#train_data_numeric=transform_to_numeric(train_data)
#test_data_numeric=transform_to_numeric(test_data)
train_data_numeric=pd.get_dummies(train_data)
test_data_numeric=pd.get_dummies(test_data)

#####################################################
#MSSubClass categorical feature with numerical values
#####################################################
dummies_train=pd.get_dummies(train_data.MSSubClass,prefix='MSSubClass')
train_data_numeric=train_data_numeric.join(dummies_train,how='outer')
train_data_numeric=train_data_numeric.drop(columns=['MSSubClass'])

dummies_test=pd.get_dummies(test_data.MSSubClass,prefix='MSSubClass')
test_data_numeric=test_data_numeric.join(dummies_test,how='outer')
test_data_numeric=test_data_numeric.drop(columns=['MSSubClass'])

#############################################################################
#Missing test and train data columns e.g. if there is more encoded categories in train data
#############################################################################
SalePrice=train_data_numeric.SalePrice

missing_test_cols=[f for f in train_data_numeric.columns.values if f not in test_data_numeric.columns.values]
missing_train_cols=[f for f in test_data_numeric.columns.values if f not in train_data_numeric.columns.values]

for col in missing_test_cols:
  train_data_numeric=train_data_numeric.drop(columns=col) #drop since no predictive value
for col in missing_train_cols:
  test_data_numeric=test_data_numeric.drop(columns=col) #drop since no predictive value

#####################################
#Add feature with all Square Feet
#####################################
##DID LEAD to a slightly worse result
train_data_numeric['SF']=train_data_numeric['TotalBsmtSF']+train_data_numeric['1stFlrSF']+train_data_numeric['2ndFlrSF']
test_data_numeric['SF']=test_data_numeric['TotalBsmtSF']+test_data_numeric['1stFlrSF']+test_data_numeric['2ndFlrSF']


# In[7]:


#################################################
#Correlation analysis for all features incl. encoded
#################################################
train_data_numeric['SalePrice']=SalePrice
cm=train_data_numeric.corr()
corr_with_saleprice_full=cm.loc['SalePrice',:].abs().sort_values(ascending=False) #Find correlation of Salesprice with all features, in sorted order. 
relevant_columns=corr_with_saleprice_full[1:20].index
#plot SalePrice vs. N features with highest correlation
plot_gallery(train_data_numeric,'SalePrice',relevant_columns)

#drop columns that have no correlation
drop_cols=corr_with_saleprice_full[corr_with_saleprice_full<0.01].index
train_data_numeric=train_data_numeric.drop(columns=drop_cols)
test_data_numeric=test_data_numeric.drop(columns=drop_cols)

###############################################
#Log transformation of SalePrice and other relevant features
##############################################
SalePrice=np.log1p(train_data_numeric.SalePrice) #log1p makes skew distribution of SalePrice more normal distributed
train_data_numeric=train_data_numeric.drop(columns=['SalePrice'])

for col in relevant_columns:
  if np.abs(stats.skew(train_data_numeric[col]))>0.5:
    train_data_numeric[col]=np.log1p(train_data_numeric[col])
    test_data_numeric[col]=np.log1p(test_data_numeric[col])
'''
print('test')
train_data_numeric['GrLivArea']=np.log1p(train_data_numeric['GrLivArea']) 
test_data_numeric['GrLivArea']=np.log1p(test_data_numeric['GrLivArea'])
'''
######################################################################
#Split train_data into train and test data, since for actual test data 
#we do not have the sales price 
#####################################################################

X,X_test,Y,Y_test=train_test_split(train_data_numeric,SalePrice,test_size=0.02,random_state=7)


# In[11]:

######################################################################
#ESTIMATORS, parameter grids and pipelines
#This section contains the model optimization by parameter search
#For various models
######################################################################

#RandomForestRegressor
RF_grid={'n_estimators': [16,20,100,200],
            'max_features':range(max(train_data_numeric.shape[1]-10,0),
            train_data_numeric.shape[1],10),'max_depth':range(18,23,1)}

#Making individual grids for two parameters at a time
RF_grid1={'n_estimators':[10,100,400,1000,2000],'max_features':range(10,111,30)}
#RF_gridsearch=gridsearch_fit(RandomForestRegressor(random_state=7),X,Y,RF_grid)

RF_gridsearch=gridsearch_cv_two_parameters(RandomForestRegressor(max_depth=4,random_state=12),\
					   X,Y,RF_grid1,scoring='neg_mean_squared_error',\
                                           verbose=3,plot_results=True)

# In[12]:
RFR=RF_gridsearch.best_estimator_
print(RFR)

RF_grid2={'max_depth':[4,6],'min_samples_split' : [4,6,8,10]}
RF_gridsearch=gridsearch_cv_two_parameters(RFR,X,Y,RF_grid2,scoring='neg_mean_squared_error',
                                           verbose=3,plot_results=True)
RFR=RF_gridsearch.best_estimator_


# In[19]:

RF_grid3={'min_samples_leaf' : [1,2,4,8,10],'min_samples_split':[5,6,7]}
RF_gridsearch=gridsearch_cv_two_parameters(RFR,X,Y,RF_grid3,scoring='neg_mean_squared_error',
                                           verbose=3,plot_results=True)
RFR=RF_gridsearch.best_estimator_


# In[20]:
print(RFR)


# In[23]:
RF_grid4={'max_leaf_nodes' : [50,100,150],'max_features':[100,110]}
RF_gridsearch=gridsearch_cv_two_parameters(RFR,X,Y,RF_grid4,scoring='neg_mean_squared_error',
                                           verbose=3,plot_results=True)
RFR=RF_gridsearch.best_estimator_


# In[20]:

#Lasso
Lasso_grid=[{},{'Estimator__alpha': np.arange(0.0001,0.001,0.0001)},]
Lasso_gridsearch=gridsearch_fit(Lasso(random_state=4),X,Y,Lasso_grid,scaler=RobustScaler(),verbose=2)
lasso=Lasso_gridsearch.best_estimator_
#print(Lasso_gridsearch.cv_results_['params'][1:])
plt.figure()
plt.plot(Lasso_grid[1]['Estimator__alpha'],np.sqrt(-Lasso_gridsearch.cv_results_['mean_test_score'][1:])) 
print(np.sqrt(-Lasso_gridsearch.cv_results_['mean_test_score']))


# In[21]:

#Elastic_Net
EN_grid=[{},{'Estimator__alpha': np.arange(0.0005,0.003,0.0001)},]
EN_gridsearch=gridsearch_fit(ElasticNet(random_state=12),X,Y,EN_grid,scaler=RobustScaler(),verbose=1)
EN=EN_gridsearch.best_estimator_

plt.figure()
plt.plot(EN_grid[1]['Estimator__alpha'],np.sqrt(-EN_gridsearch.cv_results_['mean_test_score'][1:])) 
print(np.sqrt(-EN_gridsearch.cv_results_['mean_test_score']))

# In[22]:

KRR_grid={'alpha': np.arange(0.5,1.01,0.01),'kernel': ['linear']}
KRR_gridsearch=gridsearch_cv_two_parameters(KernelRidge(),X,Y,KRR_grid,scoring='neg_mean_squared_error',
                                           verbose=3,plot_results=True)

KRR=KRR_gridsearch.best_estimator_

# In[23]:

#GradientBoostingRegressor
'''
Approach

    Choose a relatively high learning rate. Generally the default value of 0.1 works but somewhere between 0.05 to 0.2 should work for different problems
    Determine the optimum number of trees for this learning rate. Choose a learning rate with the system can work fairly fast. This is because it will be used for testing various scenarios and determining the tree parameters.
    Tune tree-specific parameters for decided learning rate and number of trees. 
    Lower the learning rate and increase the estimators proportionally to get more robust models, and better score.
'''
#TODO: use XGBoost.DMatrix class instead of pandas for train and test
GBoost_grid1={'n_estimators': range(100,1000,50),'learning_rate': [0.1]}
GBoost_gridsearch=gridsearch_cv_two_parameters(GradientBoostingRegressor(max_features='sqrt',loss='huber',random_state=12),X,Y,GBoost_grid1,
                                           scoring='neg_mean_squared_error',verbose=3,plot_results=True)

GBoost=GBoost_gridsearch.best_estimator_


# In[24]:
GBoost_grid2={'max_depth': range(2,11,2),'min_samples_split': range(4,21,2)}
GBoost_gridsearch=gridsearch_cv_two_parameters(GBoost,X,Y,GBoost_grid2,
                                           scoring='neg_mean_squared_error',verbose=3,plot_results=True)
GBoost=GBoost_gridsearch.best_estimator_


# In[25]:
print(GBoost)
# In[26]:

GBoost_grid3={'min_samples_split':[2,3,4],'min_samples_leaf': range(1,11,1)}
GBoost_gridsearch=gridsearch_cv_two_parameters(GBoost,X,Y,GBoost_grid3,
                                           scoring='neg_mean_squared_error',verbose=3,plot_results=True)

# In[27]:
GBoost=GBoost_gridsearch.best_estimator_

# In[28]:
print(GBoost)

# In[29]:
print(GBoost_gridsearch.cv_results_['std_test_score'])
GBoost_grid4={'max_features': range(10,141,10),'alpha': [0.9]}
GBoost_gridsearch=gridsearch_cv_two_parameters(GBoost,X,Y,GBoost_grid4,
                                           scoring='neg_mean_squared_error',verbose=3,plot_results=True)
# In[30]:
GBoost=GBoost_gridsearch.best_estimator_
GBoost_grid5={'n_estimators':[4000,5000,6000],'learning_rate':[0.01,0.02]}
GBoost_gridsearch=gridsearch_cv_two_parameters(GBoost,X,Y,GBoost_grid5,
                                           scoring='neg_mean_squared_error',verbose=3,plot_results=True)


# In[8]:

#XGBoost
#Notes: n_estimators: Number of decision trees. min_samples_split on decision tree are min. of samples necessary for a node to be split again. This prevents overfitting. min_samples_leaf puts a requirement on min number of samples on the final nodes. max_depth defines the max number of layers in decision tree, and max_features is the max number of features which is used for deciding each split.
XGBoost_grid1={'n_estimators':range(120,221,50),'learning_rate':np.arange(0.03,0.2,0.02)}

XGBoost_gridsearch=gridsearch_cv_two_parameters(xgb.XGBRegressor(random_state=12),X,Y,XGBoost_grid1,scoring='neg_mean_squared_error',
                                           verbose=3,plot_results=True)

XGBoost=XGBoost_gridsearch.best_estimator_

# In[19]:
print(XGBoost)


# In[13]:
XGBoost_grid1={'max_depth':range(2,10),'min_child_weight':np.arange(0.5,2.5,0.5)}
XGBoost_gridsearch=gridsearch_cv_two_parameters(XGBoost,X,Y,XGBoost_grid1,scoring='neg_mean_squared_error',
                                           verbose=3,plot_results=True)
XGBoost=XGBoost_gridsearch.best_estimator_


# In[14]:
XGBoost_grid1={'gamma':np.arange(0,1.2,0.2),'subsample':np.arange(0.2,0.55,0.1)}
XGBoost_gridsearch=gridsearch_cv_two_parameters(XGBoost,X,Y,XGBoost_grid1,scoring='neg_mean_squared_error',
                                           verbose=3,plot_results=True)
XGBoost=XGBoost_gridsearch.best_estimator_


# In[17]:
XGBoost_grid1={'subsample':np.arange(0.1,1.01,0.1),'colsample_bytree':np.arange(0.2,1.01,0.1)}
XGBoost_gridsearch=gridsearch_cv_two_parameters(XGBoost,X,Y,XGBoost_grid1,scoring='neg_mean_squared_error',
                                           verbose=3,plot_results=True)
XGBoost=XGBoost_gridsearch.best_estimator_


# In[18]:


XGBoost_grid1={'n_estimators':[3000,4000,5000,6000],'learning_rate':[0.01,0.015,0.02]}
XGBoost_gridsearch=gridsearch_cv_two_parameters(XGBoost,X,Y,XGBoost_grid1,scoring='neg_mean_squared_error',
                                           verbose=3,plot_results=True)
XGBoost=XGBoost_gridsearch.best_estimator_


# In[19]:
print(XGBoost)

# In[36]:
print(np.sqrt(-XGBoost_gridsearch.cv_results_['mean_test_score']))

# In[46]:

#TODO: LightGBM

averaged_models=AvgModels(models=(KRR,lasso,EN,GBoost))
averaged_models.fit(X,Y)

stacked_models=StackingAvgModels(models=(KRR,EN,GBoost),meta_model=lasso)
stacked_models.fit(X.values,Y.values)

#TODO: Better scores necessary. Maybe use cv_results, in order not to recalculate all
lasso_score=rmsle_cv(lasso,X,Y)
RFR_score=rmsle_cv(RFR,X,Y)
EN_score=rmsle_cv(EN,X,Y)
KRR_score=rmsle_cv(KRR,X,Y)
GBoost_score=rmsle_cv(GBoost,X,Y)

Avg_score=rmsle_cv(averaged_models,X,Y)
stacked_score=rmsle_cv(stacked_models,X,Y.values)


ID=test_data_numeric.index.values
prd=lasso.predict(test_data_numeric)
prdRfR=RFR.predict(test_data_numeric)
prdEN=EN.predict(test_data_numeric)
prdKRR=KRR.predict(test_data_numeric)
prdAVG=averaged_models.predict(test_data_numeric)
prdstacked=stacked_models.predict(test_data_numeric)


# In[45]:


prdXGBoost=XGBoost.predict(test_data_numeric)
#print('Score=',lasso_score)
#print('Mean,sd=',lasso_score.mean(),lasso_score.std())

#print('Score RfR=',RFR_score)
#print('Score EN=',EN_score)
#print('Score KRR=',KRR_score)
print('Score GBoost',GBoost_score)
print('Gboost score mean={} and sd={}'.format(GBoost_score.mean(),GBoost_score.std()))
print('Avg score',Avg_score)
print('Avg score mean={} and sd={}'.format(Avg_score.mean(),Avg_score.std()))
print('Stacked score',stacked_score)
print('Stacked score mean={} and sd={}'.format(stacked_score.mean(),stacked_score.std()))
df_export=pd.DataFrame(columns=['ID'],data=ID)
df_export['SalePrice']=np.exp(prdstacked)-1
df_export.to_csv('result.csv',index=False)

