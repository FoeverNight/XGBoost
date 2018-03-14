import xgboost as xgb
# 读取数据
dtrain = xgb.DMatrix('agaricus.txt.train')
dtest = xgb.DMatrix('agaricus.txt.test')
# 通过 map 指定参数
param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# 预测
preds = bst.predict(dtest)
