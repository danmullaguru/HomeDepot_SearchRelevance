__author__ = 'danmullaguru'

import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn import svm
from sklearn import datasets
from sklearn import svm
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBClassifier
from sklearn import preprocessing
import xgboost as xgb
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score



'''
load df_train
load df_test

split in to df_train, df_cv


fit the algo on train
run on the cv. measure the accuracy.

run the algo on the Test

'''

#df_train = pd.read_csv("Data/train.csv", encoding="ISO-8859-1")
#print(df_train.head())



df_train_pickle = open("Data/df_train_merged.pickle","rb")
df_train_merged = pickle.load(df_train_pickle)
df_train_pickle.close()

#print(df_train_merged.loc[df_train_merged['sw_in_brand_name_ratio'] > 27])

#min_max_scaler = preprocessing.MinMaxScaler()
#df_train_merged = min_max_scaler.fit_transform(df_train_merged)
#df_train_merged = pd.DataFrame(min_max_scaler.fit_transform(df_train_merged), columns=df_train_merged.columns)
print(df_train_merged.head())

#df_train_merged = df_train_merged.sample(frac=1)
print(np.amax(df_train_merged))
print(np.amin(df_train_merged))
#print("train shape\n",df_train_merged.shape)
print(df_train_merged.tail())

df_test_pickle = open("Data/df_test_merged.pickle","rb")
df_test_merged = pickle.load(df_test_pickle)
df_test_pickle.close()
id_test = df_test_merged['id']
#print("test shape\n",df_test_merged.shape)
#print(df_test_merged.head())

#train_data_count = df_train_merged['product_uid'].count()
#cv_start = int(train_data_count * 0.8)
#print(train_data_count,"\n",cv_start)

'''
df_train = df_train_merged.ix[:cv_start, :]
print(df_train.shape)
df_cv = df_train_merged.ix[cv_start:, :]
print(df_cv.shape)
y_cv = df_cv['relevance'].values
X_cv = df_cv.drop(['relevance'],axis=1).values

'''

df_train_merged.to_csv('traindata-x.csv')
target_train = df_train_merged['relevance'].values
column_list = ['sw_in_brand_name_ratio','sw_in_material_ratio','sw_in_title_ratio','sw_in_desc_ratio','sn_in_title_ratio','sn_in_desc_ratio']
#column_list = ['sw_in_brand_name_ratio','sw_in_title_ratio','sw_in_desc_ratio','sn_in_title_ratio']
data_train = df_train_merged.ix[:, column_list]

data_test = df_test_merged.ix[:, column_list]
#data_train = df_train_merged.drop(['relevance','id'],axis=1)
#print(df_train_merged.tail())

#print(df_train.head(100))
#print(df_train_merged.loc[df_train_merged['product_uid'] == 100006])
#print(df_train.loc[df_train['product_uid'] == 100006])
#print(df_train_merged.head(100))


X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(data_train, target_train, test_size=0.2, random_state=0)

#y_cv.to_csv('cvdata_test.csv')
#print("index of max:",np.argmax(y_cv))
#print(y_cv.shape)
#print(y_cv[40990:])
#print(X_train.head())
#print(X_train)

def runRandomForest(X_train,y_train,X_cv,y_cv):
    #rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
    rf = RandomForestRegressor(n_estimators = 100, n_jobs = -1, random_state = 2016, verbose = 1)
    #clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
    print("start fitting Random Forest model")
    rf.fit(X_train, y_train)
    print("completed fit..start prediction\n")
    print("\n RMSE:\n")
    y_pred = rf.predict(X_cv)
    #print(mean_squared_error(y_cv, y_pred))
    print(mean_squared_error(y_cv, y_pred))
    #y_pred = clf.predict(X_cv)
    #print(y_pred)
    return

def runRandomForestFinal(X_train,y_train,X_test):
    #rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
    rf = RandomForestRegressor(n_estimators = 100, n_jobs = -1, random_state = 2016, verbose = 1)
    #clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
    print("start fitting Random Forest model")
    rf.fit(X_train, y_train)
    print("completed fit..start prediction\n")
    print("\n RMSE:\n")
    y_pred = rf.predict(X_test)
    write_yPred_to_file(y_pred,'DanRF_Regression_with_puid.csv')
    return


def runLinearRegression(X_train,y_train,X_cv,y_cv):
    print("start fitting Linear Regression model")
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print("completed fit..start prediction\n")
    print("\n RMSE:\n")
    y_pred = regr.predict(X_cv)
    #print(mean_squared_error(y_pred, y_pred))
    print(mean_squared_error(y_cv, y_pred))
    return


def runLinearRegressionFinal(X_train,y_train,X_test):
    print("start fitting Linear Regression model")
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print("completed fit..start prediction\n")
    y_pred = regr.predict(X_test)
    #print(y_pred.shape)
    write_yPred_to_file(y_pred,'DanLinearRegression.csv')
    return



def runSGDRegressor(X_train,y_train,X_cv,y_cv):
    print("start fitting SGDRegressor model")
    regr = linear_model.SGDRegressor()
    regr.fit(X_train, y_train)
    print("completed fit..start prediction\n")
    print("\n RMSE:\n")
    y_pred = regr.predict(X_cv)
    #print(mean_squared_error(y_pred, y_pred))
    print(mean_squared_error(y_cv, y_pred))
    return


def runSGDRegressorFinal(X_train,y_train,X_test):
    print("start fitting SGDRegressor model")
    regr = linear_model.SGDRegressor()
    regr.fit(X_train, y_train)
    print("completed fit..start prediction\n")
    y_pred = regr.predict(X_test)
    #print(y_pred.shape)
    write_yPred_to_file(y_pred,'DanSGDRegressor.csv')
    return


def write_yPred_to_file(y_pred,filename):
    print(np.amax(y_pred))
    print(np.amin(y_pred))
    y_pred = np.around(y_pred,decimals=2)
    y_pred = np.where(y_pred > 3, 3, y_pred)
    print(np.amax(y_pred))
    print(np.amin(y_pred))
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv("submissions/feb25/"+filename,index=False)
    return

def runXGBRegressor(X_train,y_train,X_cv,y_cv):
    print("start fitting XGBRegressor model")
    xgb_model = xgb.XGBRegressor(n_estimators=200,max_depth=10)
    xgb_model.fit(X_train, y_train)
    print("completed fit..start prediction\n")
    print("\n RMSE:\n")
    y_pred = xgb_model.predict(X_cv)
    #print(mean_squared_error(y_pred, y_pred))
    print(mean_squared_error(y_cv, y_pred))
    '''
    print("Parameter optimization")
    clf = GridSearchCV(xgb_model,
                   {'max_depth': [2,4,6],
                    'n_estimators': [50,100,200]}, verbose=1)
    clf.fit(X_train,y_train)
    print(clf.best_score_)
    print(clf.best_params_)
    y_pred = clf.predict(X_cv)
    #print(mean_squared_error(y_pred, y_pred))
    print(mean_squared_error(y_cv, y_pred))
    '''
    return


def runXGBRegressorFinal(X_train,y_train,X_test):
    print("start fitting XGBRegressor model")
    xgb_model = xgb.XGBRegressor(n_estimators=100,max_depth=4)
    xgb_model.fit(X_train, y_train)
    print("completed fit..start prediction\n")
    y_pred = xgb_model.predict(X_test)
    #print(y_pred.shape)
    write_yPred_to_file(y_pred,'DanXGBRegressor.csv')
    return

def runLinearLasso(X_train,y_train,X_cv,y_cv):
    alpha = 0.0001
    lasso = Lasso(alpha=alpha,max_iter=10000)
    y_pred_lasso = lasso.fit(X_train, y_train).predict(X_cv)
    r2_score_lasso = r2_score(y_cv, y_pred_lasso)
    print("Lasso:\n",lasso)
    print("r^2 on test data : %f" % r2_score_lasso)
    print("mean squared error-------\n")
    print(mean_squared_error(y_cv, y_pred_lasso))
    print("coef\n",lasso.coef_ )
    return


def runLinearLassoFinal(X_train,y_train,X_test):
    alpha = 0.1
    lasso = Lasso(alpha=alpha)
    y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
    write_yPred_to_file(y_pred_lasso,'DanLinearLassoRegressor.csv')
    return


runRandomForest(X_train,y_train,X_cv,y_cv)
runLinearRegression(X_train,y_train,X_cv,y_cv)
runSGDRegressor(X_train,y_train,X_cv,y_cv)
runXGBRegressor(X_train,y_train,X_cv,y_cv)
runLinearLasso(X_train,y_train,X_cv,y_cv)

#runLinearRegressionFinal(data_train,target_train,data_test)
#runRandomForestFinal(data_train,target_train,data_test)
#runSGDRegressorFinal(data_train,target_train,data_test)
#runXGBRegressorFinal(data_train,target_train,data_test)
#runLinearLassoFinal(data_train,target_train,data_test)

#pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)



print("Done")