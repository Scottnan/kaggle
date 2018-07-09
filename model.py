import pandas as pd
import json
import gc
import warnings
import xgboost as xgb
import numpy as np
from xgboost.core import XGBoostError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")

configs = json.loads(open('config.json').read())


class KaggleModel(object):
    def __init__(self, configs):
        self.data_path = configs['data']['path']
        # self.predict_path = configs['data']['predict']
        self.model = None
        self.X_train, self.y_train, self.X_predict = self.preprocess()

    def preprocess(self):
        df = pd.read_hdf(self.data_path)
        df.drop(['index', 'NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER', 'PREV_APP_CREDIT_PERC_MAX',
                 'PREV_APP_CREDIT_PERC_MEAN', 'INSTAL_PAYMENT_PERC_MAX', 'INSTAL_PAYMENT_PERC_MEAN'],
                axis=1, inplace=True)
        data = df[df['TARGET'].notnull()]
        predict = df[df['TARGET'].isnull()]
        del df
        gc.collect()

        label_train, label_pred = self.cluster()
        self.n_clusters_ = len(set(label_train.TYPE))
        data = data.merge(label_train)
        predict = predict.merge(label_pred)
        del label_train, label_pred
        gc.collect()

        data = data.ix[:, data.isna().sum(axis=0) / len(data) <= 0.2]
        data.dropna(inplace=True)
        self.label_train = data[['SK_ID_CURR', 'TYPE']]
        self.label_pred = predict[['SK_ID_CURR', 'TYPE']]
        columns = data.columns.tolist()
        columns.remove('SK_ID_CURR')
        columns.remove('TARGET')
        predict = predict[columns]
        predict.fillna(predict.mean(), inplace=True)

        X, y = data.drop(["SK_ID_CURR", "TARGET"], axis=1), data.TARGET
        '''
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=666)
        '''
        scaler = StandardScaler().fit(X)
        X_train = scaler.transform(X)
        y_train = y
        # X_test = scaler.transform(X_test)
        X_predict = scaler.transform(predict)
        # self.X = X_train
        # self.y = y.tolist()
        return X_train, y_train, X_predict

    @staticmethod
    def cluster():
        # 聚类应用的列：性别，年龄，工作年限，收入水平，贷款水平，工作性质，是否有不动产，有无固定电话，婚姻情况，住房性质，收入类型
        cols = ['CODE_GENDER', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'OCCUPATION_TYPE',
                'FLAG_OWN_REALTY', 'FLAG_PHONE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE']
        train = pd.read_hdf("clean_data/main_train.h5")
        train_id = train['SK_ID_CURR']
        train = train[cols]
        test = pd.read_hdf("clean_data/main_test.h5")
        test_id = test['SK_ID_CURR']
        test = test[cols]
        scaler = StandardScaler().fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)
        cluster = KMeans(n_clusters=4).fit(train)
        label_train = pd.DataFrame({'SK_ID_CURR': train_id, 'TYPE': cluster.labels_})
        label_pred = pd.DataFrame({'SK_ID_CURR': test_id, 'TYPE': cluster.predict(test)})
        label_pred.to_csv("pred.csv")
        return label_train, label_pred

    @staticmethod
    def classify(X, y, num_boost):
        param = {'max_depth': 2,
                 'min_child_weight': 30,
                 'gamma': 0,
                 'subsample': 0.85,
                 'colsample_bytree': 0.7,
                 'colsample_bylevel': 0.632,
                 'eta': 0.1,
                 'silent': 1,
                 'objective': 'binary:logistic',
                 'eval_metric': 'auc'}
        dtrain = xgb.DMatrix(X, y)
        bst = xgb.train(param, dtrain, num_boost_round=num_boost)
        return bst

    @staticmethod
    def cross_validation(X, y):
        dtrain = xgb.DMatrix(X, y)
        param = {'max_depth': 2,
                 'min_child_weight': 30,
                 'gamma': 0,
                 'subsample': 0.85,
                 'colsample_bytree': 0.7,
                 'colsample_bylevel': 0.632,
                 'eta': 0.1,
                 'silent': 1,
                 'objective': 'binary:logistic',
                 'eval_metric': 'auc'}
        cv_output = xgb.cv(param, dtrain, num_boost_round=5000, early_stopping_rounds=100, verbose_eval=10,
                           show_stdv=False)
        cv_output[['train-auc-mean', 'test-auc-mean']].plot()
        return cv_output.shape[0]

    @staticmethod
    def predict(model, X_predict):
        dpre = xgb.DMatrix(X_predict)
        bst = model
        pre = bst.predict(dpre)
        return pre
        # df = pd.read_hdf(self.data_path)
        # pred = df[df['TARGET'].isnull()]
        # del df

    def main(self, use_cv=True):
        if use_cv:
            for t in range(self.n_clusters_):
                FLAG = self.label_train['TYPE'] == t
                X = self.X_train[FLAG]
                y = self.y_train[FLAG].tolist()
                print('> cross validation for type {}'.format(t))
                try:
                    self.cross_validation(X, y)
                except XGBoostError:
                    print('y is only contains {}'.format(np.mean(y)))
        else:
            result = pd.DataFrame(columns=['SK_ID_CURR', 'TARGET'])
            for t in range(self.n_clusters_):
                # train
                FLAG_train = self.label_train['TYPE'] == t
                X = self.X_train[FLAG_train]
                y = self.y_train[FLAG_train].tolist()
                print('> fit xgboost for type {}'.format(t))
                try:
                    num = self.cross_validation(X, y)
                    model = self.classify(X, y, num)
                except XGBoostError:
                    print('y is only contains {}'.format(np.mean(y)))
                    FLAG_pred = self.label_pred['TYPE'] == t
                    X_ID = self.label_pred.ix[FLAG_pred, 'SK_ID_CURR']
                    pred = pd.DataFrame({'SK_ID_CURR': X_ID, 'TARGET': np.mean(y)})
                    result = pd.concat([result, pred])
                    continue

                # predict
                FLAG_pred = self.label_pred['TYPE'] == t
                X_pred = self.X_predict[FLAG_pred]
                X_ID = self.label_pred.ix[FLAG_pred, 'SK_ID_CURR']
                pred = pd.DataFrame({'SK_ID_CURR': X_ID, 'TARGET': self.predict(model, X_pred)})
                result = pd.concat([result, pred])

            result.to_csv("pre.csv", index=False)


if __name__ == "__main__":
    configs = json.loads(open('config.json').read())
    model = KaggleModel(configs)
    model.main(use_cv=False)
    # model.cross_validation()
    # print(model.X_predict)
    # model.classify()
    # model.predict()
