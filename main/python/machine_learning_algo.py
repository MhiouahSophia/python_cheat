import sys
from collections import Counter
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, f1_score
sys.path.append('../')
from matrix_utils import *
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression  # modele linear
from sklearn.ensemble import RandomForestRegressor  # Random Forest
from sklearn.ensemble import GradientBoostingRegressor  # Gradient Boosting
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn.ensemble import ExtraTreesClassifier  # Extra Trees
from sklearn.ensemble import GradientBoostingClassifier  # Gradient Boosting
from collections import Counter
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, f1_score
import lightgbm as lgb
from collections import Counter
import dask.dataframe as dd
import time
import random
import re
import imblearn.over_sampling
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier




########## SAMPLING
#exemple1:

sm = SMOTE(ratio='minority', random_state=7, n_jobs=48)
df_train, y_true_class = sm.fit_sample(train1, y_true_presence)


########## ML LGBM
#exemple1:

class_weight = dict({0: 1, 1: 2})
# class_weight = None
gbm = lgb.LGBMClassifier(max_depth = 10,
                         class_weight = class_weight,
                         num_leaves=31,
                         learning_rate=0.05,
                         random_state = 1,
                         n_estimators=5000,
                         min_data = 50)

gbm.fit(df_train, y_train_class,
        eval_set=[(val, y_val_class)],
        eval_metric='F1-score',
        early_stopping_rounds=600)
y_pred_class_gbm = gbm.predict(test)

########## ML BalanceBaggingClassifier
#exemple1:
#Create an object of the classifier.
bbc = BalancedBaggingClassifier(base_estimator= RandomForestClassifier(),
                                replacement=False,
                                n_estimators = 100,
                                random_state=0, n_jobs = -1)

#Train the classifier.
bbc.fit(df_train, y_train_class)
y_pred_class_gbm = bbc.predict(df_test)

########## ML RandomForestClassifier
#exemple1:
# class_weight = dict({0:1, 1:14})
class_weight = None
rdf = RandomForestClassifier(bootstrap=True,class_weight=class_weight,
            criterion='gini',
            max_depth=8, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=4, min_samples_split=10,
            min_weight_fraction_leaf=0.0, n_estimators=300,
            oob_score=False,
            random_state=1,
            verbose=0, warm_start=False)
rdf.fit(df_train, y_train_class)
y_pred_class_gbm = rdf.predict(df_test)

########## Confusion Matrix/ F1_score / Accuracy_score / LogLoss
#exemple1:
cfm = confusion_matrix(y_test_class, y_pred_class_gbm)
print(cfm)
F1_score = f1_score(y_test_class, y_pred_class_gbm)
print('F1_score: ' + str(F1_score))
print('Accuracy y_pred_class : %.2f%%'%(100.*accuracy_score(y_test_class, y_pred_class_gbm)))
print('Log loss y_pred_class: %.3f'%log_loss(y_test_class, y_pred_class_gbm))

########## Feature importance
#exemple1:
# Plot feature importanceapp
feature_importance = gbm.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 2)
plt.barh(pos[-30:], feature_importance[sorted_idx][-30:], align='center')
plt.yticks(pos[-30:], df_train.columns[sorted_idx][-30:])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

