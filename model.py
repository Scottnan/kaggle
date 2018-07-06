import pandas as pd
import json
import gc
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.cluster import KMeans

configs = json.loads(open('config.json').read())


class KaggleModel(object):
    def __init__(self, configs):
        self.data_path = configs['data']['path']
        # self.predict_path = configs['data']['predict']
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test, self.X_predict = self.preprocess()

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
        data = data.merge(label_train)
        predict = predict.merge(label_pred)
        del label_train, label_pred
        gc.collect()

        data = data.ix[:, data.isna().sum(axis=0) / len(data) <= 0.2]
        data.dropna(inplace=True)
        columns = data.columns.tolist()
        columns.remove('SK_ID_CURR')
        columns.remove('TARGET')
        predict = predict[columns]
        predict.fillna(predict.mean(), inplace=True)
        X, y = data.drop(["SK_ID_CURR", "TARGET"], axis=1), data.TARGET
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=666)
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_predict = scaler.transform(predict)
        self.X = scale(X)
        self.y = y.tolist()
        return X_train, X_test, y_train, y_test, X_predict

    def cluster(self):
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
        km_cluster = KMeans(n_clusters=4)
        km_cluster.fit(train)
        label_train = pd.DataFrame({'SK_ID_CURR': train_id, 'TYPE': km_cluster.labels_})
        label_train.to_csv("train.csv")
        label_pred = pd.DataFrame({'SK_ID_CURR': test_id, 'TYPE': km_cluster.predict(test)})
        label_pred.to_csv("pred.csv")
        return label_train, label_pred

    def classify(self):
        param = {'max_depth': 6,
                 'min_child_weight': 30,
                 'gamma': 0,
                 'subsample': 0.85,
                 'colsample_bytree': 0.7,
                 'colsample_bylevel': 0.632,
                 'eta': 0.2,
                 'silent': 0,
                 'objective': 'binary:logistic',
                 'eval_metric': 'auc'}
        dtrain = xgb.DMatrix(self.X, self.y)
        bst = xgb.train(param, dtrain, 110)  # , early_stopping_rounds=20)
        self.model = bst

    def cross_validation(self):
        dtrain = xgb.DMatrix(self.X_train, self.y_train)
        param = {'max_depth': 4,
                 'min_child_weight': 30,
                 'gamma': 0,
                 'subsample': 0.85,
                 'colsample_bytree': 0.7,
                 'colsample_bylevel': 0.632,
                 'eta': 0.2,
                 'silent': 1,
                 'objective': 'binary:logistic',
                 'eval_metric': 'auc'}
        cv_output = xgb.cv(param, dtrain, num_boost_round=500, early_stopping_rounds=20, verbose_eval=10,
                           show_stdv=False)
        cv_output[['train-auc-mean', 'test-auc-mean']].plot()

    def predict(self):
        dpre = xgb.DMatrix(self.X_predict)
        bst = self.model
        pre = bst.predict(dpre)
        df = pd.read_hdf(self.data_path)
        pred = df[df['TARGET'].isnull()]
        del df
        gc.collect()
        pre = pd.DataFrame({'SK_ID_CURR': pred['SK_ID_CURR'], 'TARGET': pre})
        pre.to_csv("pre.csv", index=False)


if __name__ == "__main__":
    configs = json.loads(open('config.json').read())
    model = KaggleModel(configs)
    model.cluster()
    # model.cross_validation()
    # print(model.X_predict)
    # model.classify()
    # model.predict()
