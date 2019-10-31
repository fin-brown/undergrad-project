#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 13:49:55 2018

@author: finbrown1
"""

import numpy as np
import math
import pandas as pd
import warnings

from cross_validation import (InnerCVSetup, OuterCVSetup, mse_variance,
                              rmse_variance)

from data.clean import clean_data as long
from data.transforms import RemoveOutliers, TopXPercentile

from scipy.stats import wilcoxon, norm

from sklearn_estimators.purdy import PurdyEstimator
from sklearn_estimators.riegel import RiegelEstimator
from sklearn_estimators.rfwide import RandomForestWideRegressor

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


# two column selector functions for use with FunctionTransformer in pipelines 
def event_column(X):
    return X[:, 1].reshape(-1, 1)

def id_event_columns(X):
    return X[:, 0:2]
 
    
X = long.loc[:, ["ID", "Event", "Distance"]].reset_index(drop=True)
y = long.loc[:, "Time"].reset_index(drop=True)

# chain these seperately rather than in a pipeline as they only support pd  
X, y = RemoveOutliers(cutoff=.04, abs_cutoff=.01).fit(X, y).transform()
X, y = TopXPercentile(cutoff=.25).fit(X, y).transform()

ylog = y.apply(np.log)

inner_cv_params = dict(scoring="neg_mean_squared_error",
                       cv=InnerCVSetup(iters=1000, random_state=0),
                       n_jobs=-1)

# harmless - tried resolving as many of these as possible still there somewhere
if __name__=="__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # mean estimator, pipe event_column, OneHotEncoder and a linear model  
        print("Mean")
        mean = make_pipeline(FunctionTransformer(event_column), OneHotEncoder(), 
                             LinearRegression())
        
        # similar to mean but this time include ID too
        print("Two Factor")
        two_factor = make_pipeline(FunctionTransformer(id_event_columns), 
                                   OneHotEncoder(), LinearRegression())
         
        print("Riegel")
        riegel = GridSearchCV(RiegelEstimator(log=True), 
                              dict(sample_method=["nearest", "random"]),
                              **inner_cv_params)
        riegel.fit(X, ylog)
        
        print("Purdy")
        purdy = GridSearchCV(PurdyEstimator(log=True), 
                             dict(sample_method=["nearest", "random"]),
                             **inner_cv_params)
        purdy.fit(X, ylog)
        
        print("RF")
        rf = GridSearchCV(RandomForestWideRegressor(),
                          dict(n_estimators=[5, 10, 25, 50, 100, 200],
                               max_features=["auto", "sqrt", "log2"]),
                          **inner_cv_params)
        rf.fit(X, ylog)
 
# now for outer cross validation
outer_cv_params = dict(X=X, y=ylog, groups=X.ID,
                       cv=OuterCVSetup(n_splits=6, random_state=1), 
                       n_jobs=-1)

mse_var = make_scorer(mse_variance)

# harmless - tried resolving as many of these as possible still there somewhere
if __name__=="__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        print("Mean")
        mean_mse = cross_val_score(mean, 
                                   scoring="neg_mean_squared_error",
                                   **outer_cv_params)
        mean_mse_var = cross_val_score(mean, 
                                       scoring=mse_var, **outer_cv_params)
        
        print("2 Factor")
        twofac_mse = cross_val_score(two_factor, 
                                     scoring="neg_mean_squared_error",
                                     **outer_cv_params)
        twofac_mse_var = cross_val_score(two_factor, 
                                         scoring=mse_var, **outer_cv_params)
        
        print("Riegel")
        riegel_mse = cross_val_score(riegel, 
                                     scoring="neg_mean_squared_error",
                                     **outer_cv_params)
        riegel_mse_var = cross_val_score(riegel, 
                                         scoring=mse_var, **outer_cv_params)
            
        print("Purdy")
        purdy_mse = cross_val_score(purdy, 
                                    scoring="neg_mean_squared_error",
                                    **outer_cv_params)
        purdy_mse_var = cross_val_score(purdy, 
                                        scoring=mse_var, **outer_cv_params)
        
        print("RF")
        rf_mse = cross_val_score(rf, 
                                 scoring="neg_mean_squared_error",
                                 **outer_cv_params)
        rf_mse_var = cross_val_score(rf, 
                                     scoring=mse_var, **outer_cv_params)

# calculate normal 1-alpha/2 quantile
thi = norm.ppf(.975)

# calculate and store all our MSE CIs in a dict
mse_ci = dict()
mse_ci["mean"] = (-mean_mse.mean() - thi * math.sqrt(mean_mse_var.mean()), 
                  -mean_mse.mean() + thi * math.sqrt(mean_mse_var.mean()))   

mse_ci["twofac"] = (-twofac_mse.mean() - thi * math.sqrt(twofac_mse_var.mean()), 
                    -twofac_mse.mean() + thi * math.sqrt(twofac_mse_var.mean())) 

mse_ci["riegel"] = (-riegel_mse.mean() - thi * math.sqrt(riegel_mse_var.mean()), 
                    -riegel_mse.mean() + thi * math.sqrt(riegel_mse_var.mean())) 

mse_ci["purdy"] = (-purdy_mse.mean() - thi * math.sqrt(purdy_mse_var.mean()), 
                   -purdy_mse.mean() + thi * math.sqrt(purdy_mse_var.mean())) 

mse_ci["rf"] = (-rf_mse.mean() - thi * math.sqrt(rf_mse_var.mean()), 
                -rf_mse.mean() + thi * math.sqrt(rf_mse_var.mean())) 

# calculate and store all our RMSE CIs in a dict; use taylor approx for var
rmse_ci = dict()
rmse_ci["mean"] = ((math.sqrt(-mean_mse.mean()) - 0.5 * thi * mean_mse_var.mean() / 
                    math.sqrt(-mean_mse.mean())), 
                   (math.sqrt(-mean_mse.mean()) + 0.5 * thi * mean_mse_var.mean() / 
                    math.sqrt(-mean_mse.mean())))   

rmse_ci["twofac"] = (-twofac_mse.mean() - thi * math.sqrt(twofac_mse_var.mean()), 
                     -twofac_mse.mean() + thi * math.sqrt(twofac_mse_var.mean())) 

rmse_ci["riegel"] = (-riegel_mse.mean() - thi * math.sqrt(riegel_mse_var.mean()), 
                     -riegel_mse.mean() + thi * math.sqrt(riegel_mse_var.mean())) 

rmse_ci["purdy"] = (-purdy_mse.mean() - thi * math.sqrt(purdy_mse_var.mean()), 
                    -purdy_mse.mean() + thi * math.sqrt(purdy_mse_var.mean())) 

rmse_ci["rf"] = (-rf_mse.mean() - thi * math.sqrt(rf_mse_var.mean()), 
                 -rf_mse.mean() + thi * math.sqrt(rf_mse_var.mean())) 
    

# wilcoxon signed-rank tests
mean_test = [1, wilcoxon(mean_mse, twofac_mse).pvalue, 
             wilcoxon(mean_mse, riegel_mse).pvalue, 
             wilcoxon(mean_mse, purdy_mse).pvalue, 
             wilcoxon(mean_mse, rf_mse).pvalue]
twofac_test = [mean_test[1], 1, wilcoxon(twofac_mse, riegel_mse).pvalue,
               wilcoxon(twofac_mse, purdy_mse).pvalue, 
               wilcoxon(twofac_mse, rf_mse).pvalue]
riegel_test = [mean_test[2], twofac_test[2], 1, 
               wilcoxon(riegel_mse, purdy_mse).pvalue, 
               wilcoxon(riegel_mse, rf_mse).pvalue]
purdy_test = [mean_test[3], twofac_test[3], riegel_test[3], 1, 
              wilcoxon(riegel_mse, rf_mse).pvalue]
rf_test = [mean_test[4], twofac_test[4], riegel_test[4], purdy_test[4], 1]

# this gives interesting results: all p-values are the same. This is because
# their all the lowest possible p-values in this case, theres is no overlap
# between groups and test is based on rank. Maybe wilcoxon test isn't valid
# here. Can use confidence intervals otherwise
results = pd.DataFrame({"Mean": mean_test,
                        "2 Factor": twofac_test,
                        "Riegel": riegel_test,
                        "Purdy": purdy_test,
                        "RF": rf_test}, 
                       columns=["Mean", "2 Factor", "Riegel", "Purdy", "RF"])






