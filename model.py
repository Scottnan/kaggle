import pandas as pd
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation

configs = json.loads(open('config.json').read())


class KaggleModel(object):
    def __init__(self, configs):
        self.data_path = configs['data']['path']
        self.predict_path = configs['data']['predict']
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test, self.X_predict = self.preprocess()

    def preprocess(self):
        data = pd.read_hdf(self.data_path)
        predict = pd.read_hdf(self.predict_path)
        data = data.ix[:, data.isna().sum(axis=0) / len(data) <= 0.2]
        data.dropna(inplace=True)
        columns = data.columns.tolist()
        columns.remove('TARGET')
        predict = predict[columns]
        predict.fillna(predict.mean(), inplace=True)
        X, y = data.drop("TARGET", axis=1), data.TARGET
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_predict = scaler.transform(predict)
        return X_train, X_test, y_train, y_test, X_predict

    def cluster(self):
        # af = AffinityPropagation(preference=-50).fit(data)
        pass

    def classify(self):
        param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
        dtrain = xgb.DMatrix(self.X_train, self.y_train)
        bst = xgb.train(param, dtrain, 50)
        self.model = bst

    def cross_validation(self):
        dtrain = xgb.DMatrix(self.X_train, self.y_train)
        param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
        cv_output = xgb.cv(param, dtrain, num_boost_round=100, early_stopping_rounds=20, verbose_eval=10,
                           show_stdv=False)
        cv_output[['train-auc-mean', 'test-auc-mean']].plot()

    def predict(self):
        dpre = xgb.DMatrix(self.X_predict)
        bst = self.model
        pre = bst.predict(dpre)
        pd.Series(pre).to_csv("pre.csv")


if __name__ == "__main__":
    configs = json.loads(open('config.json').read())
    model = KaggleModel(configs)
    # print(model.X_predict)
    model.classify()
    model.predict()

