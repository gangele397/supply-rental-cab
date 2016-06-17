

import pandas as pd

from sklearn import preprocessing

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

def loadData(datafile):
    return pd.read_csv(datafile)

def splitDatetime(data):
    sub = pd.DataFrame(data.datetime.str.split(' ').tolist(), columns = "date time".split())
    date = pd.DataFrame(sub.date.str.split('-').tolist(), columns="year month day".split())
    time = pd.DataFrame(sub.time.str.split(':').tolist(), columns = "hour minute second".split())
    data['year'] = date['year']
    data['month'] = date['month']
    data['day'] = date['day']
    data['hour'] = time['hour'].astype(int)
    return data

def normalize(train, test):
    norm = preprocessing.Normalizer()
    train = norm.fit_transform(train)
    test = norm.transform(test)
    return train, test

def DecisionTree():
    est = DecisionTreeRegressor()
    return est

def RandomForest():
    est = RandomForestRegressor(n_estimators=500)
    return est

def ExtraTree():
    est = ExtraTreesRegressor(n_estimators=700)
    return est

def GradientBoostingRegressor():
    est = GradientBoostingRegressor()
    return est

def KNN():
    est = KNeighborsRegressor(n_neighbors=2)
    return est

def prediction(est, train, test, features, target):

    est.fit(train[features], train[target])

    with open(outPath + "submission-knn.csv", 'wb') as f:
        f.write("datetime,count\n")

        for index, value in enumerate(list(est.predict(test[features]))):
            f.write("%s,%s\n" % (test['datetime'].loc[index], int(value)))


def main():

    train = loadData(dataPath + "train.csv")
    test = loadData(dataPath + "test.csv")

    train = splitDatetime(train)
    test = splitDatetime(test)

    target = 'count'
    features = [column for column in train.columns if column not in ['datetime', 'casual', 'registered', 'count']]
    #try anyone of the below
    est = DecisionTree()
    est = RandomForest()
    est = ExtraTree()
    est = GradientBoostingRegressor()
    est = KNN()
   
    prediction(est, train, test, features, target)



if __name__ == "__main__":
    main()
