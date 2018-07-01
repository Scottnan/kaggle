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
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess()

    def preprocess(self):
        data = pd.read_hdf(self.data_path)
        data = data.ix[:, data.isna().sum(axis=0) / len(data) <= 0.2]
        data.dropna(inplace=True)
        X, y = data.drop("TARGET", axis=1), data.TARGET
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    def cluster(self):
        # af = AffinityPropagation(preference=-50).fit(data)
        pass

    def classify(self):
        param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
        bst = xgb.train(param, self.X_train, num_round=10)
        self.model = bst


if __name__ == "__main__":
    configs = json.loads(open('config.json').read())
    model = KaggleModel(configs)
    print(model.X_train)
