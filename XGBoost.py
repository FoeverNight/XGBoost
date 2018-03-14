import numpy as np
import pandas as pd
import xgboost as xgb
import os
import time
os.chdir('./data')


dtrain = xgb.DMatrix('single byte count-04.txt')
dtest = xgb.DMatrix('single byte count-05.txt')
'''dtrain = xgb.DMatrix('easytrain.txt')
dtest = xgb.DMatrix('easytest.txt')'''
t1=time.time()
# specify parameters via map, definition are same as c++ version
'''param = {'max_depth': 13, 'eta': 0.1, 'silent': 0, 'objective': 'binary:logistic', 'min_child_weight': 3, 'gamma': 14,
         'tree_method': 'gpu_hist'}

# specify validations set to watch performance
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 200
bst = xgb.train(param, dtrain, num_round, watchlist)

# this is prediction
preds = bst.predict(dtest)
labels = dtest.get_label()
t2=time.time()
print(t2-t1)
print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))))

print('correct=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) == labels[i]) / float(len(preds))))'''

t1 = time.time()
params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eta': 0.1,
            'max_depth': 11,
            'subsample': 1,
            'min_child_weight': 2,
            'colsample_bytree': 0.5,
            'scale_pos_weight': 0.1,
            'eval_metric': 'auc',
            'gamma': 0.2,
            'lambda': 300,
            'tree_method': 'gpu_hist'
        }
watchlist = [(dtrain, 'train'), (dtest, 'test')]
model = xgb.train(params, dtrain, num_boost_round=15400, evals=watchlist)

pred = model.predict(dtest)
labels = dtest.get_label()
pred2 = model.predict(dtrain)
labels2 = dtrain.get_label()
t2 = time.time()
print(t2-t1)
print('trainerror=%f' % (sum(1 for i in range(len(pred2)) if int(pred2[i] > 0.5) != labels2[i]) / float(len(pred2))))

print('traincorrect=%f' % (sum(1 for i in range(len(pred2)) if int(pred2[i] > 0.5) == labels2[i]) / float(len(pred2))))

print('testerror=%f' % (sum(1 for i in range(len(pred)) if int(pred[i] > 0.5) != labels[i]) / float(len(pred))))

print('testcorrect=%f' % (sum(1 for i in range(len(pred)) if int(pred[i] > 0.5) == labels[i]) / float(len(pred))))
