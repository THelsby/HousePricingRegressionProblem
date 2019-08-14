import re

import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor


def scatterPlot(dataset):
    for data in dataset.columns:
        plt.scatter(dataset[data], dataset["SalePrice"])
        plt.title(data)
        plt.savefig("Plots/" + data + ".png")
        plt.show()
        plt.close()


def readInData():
    train = pd.read_csv("prediction/pricePrediction/train.csv")
    test = pd.read_csv("prediction/pricePrediction/test.csv")
    return train, test


def checkNAValues(data):
    dataNa = (data.isnull().sum() / len(data)) * 100
    dataNa = dataNa.drop(dataNa[dataNa == 0].index).sort_values(ascending=False)
    missingData = pd.DataFrame({'Missing Ratio': dataNa})
    print(missingData)


def dropSalesPrice(train):
    train.drop(["SalePrice"], axis=1, inplace=True)
    return train


def getTrainingLabels(train):
    return train.SalePrice.values


def dropIdColumn(train, test):
    trainId = train['Id']
    testId = test['Id']

    train.drop("Id", axis=1, inplace=True)
    test.drop("Id", axis=1, inplace=True)
    return train, test, trainId, testId


def applyNormalDistributionToSales(train):
    train["SalePrice"] = np.log1p(train["SalePrice"])
    return train


def fillNaValues(data):
    data = fillNaToNone(data)
    data = fillNaToMedian(data)
    data = fillNaToMode(data)
    data = fillNaToZero(data)
    data = fillNaToCustom(data)
    return data


def removeColumns(data):
    columns = ["Utilities", "SaleType", "MoSold", "YrSold", "SaleCondition"]
    for column in columns:
        data = data.drop([column], axis=1)
    return data


def fillNaToNone(data):
    columns = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish",
               "GarageQual", "GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
               "MasVnrType"]
    for column in columns:
        data[column] = data[column].fillna("None")
    return data


def fillNaToMedian(data):
    data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))
    return data


def fillNaToMode(data):
    columns = ["MSZoning", "Functional", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd"]
    for column in columns:
        data[column] = data[column].fillna(data[column].mode()[0])
    return data


def fillNaToZero(data):
    columns = ["GarageYrBlt", "GarageArea", "GarageCars", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
               "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"]
    for column in columns:
        data[column] = data[column].fillna(0)
    return data


def fillNaToCustom(data):
    data["Functional"] = data["Functional"].fillna("Typ")
    return data


def ordinalEncoder(data):
    columns = list(data.select_dtypes(include=[np.object]))
    enc = preprocessing.OrdinalEncoder()
    for column in columns:
        data[column] = enc.fit_transform(data[column].values.reshape(-1, 1))
    return data


# def ordinalEncoderPred(data):
#     columns = list(data.select_dtypes(include=[np.object]))
#     for column in columns:
#         filename = 'prediction/ordinalEncoders/' + column + '.sav'
#         enc = pickle.loads(open(filename, 'wb'))
#         data[column] = enc.fit_transform(data[column].values.reshape(-1, 1))
#     return data


def trainTestSplit(trainData, trainLabels):
    TrainX, TestX, TrainY, TestY = train_test_split(trainData, trainLabels, test_size=0.33, random_state=42)
    return TrainX, TestX, TrainY, TestY


def supportVectorRegression(trainX, testX, trainY, testY):
    svr = SVR(gamma='scale', C=10.0, epsilon=0.2)
    svr.fit(trainX, trainY)
    confidence = svr.score(testX, testY)
    print("SVM Prediction Score {}".format(confidence))
    predictions = svr.predict(testX)
    return predictions


def rootMeanSquareError(predictions, testLabels):
    meanSquaredLogError = metrics.mean_squared_log_error(testLabels, predictions)
    return meanSquaredLogError


def printPredictionToCSV(_id, predictions):
    res = pd.DataFrame({"Id": _id, "SalePrice": predictions})
    res.to_csv("predictions.csv", index=False)


def treeRegression(trainX, testX, trainY, testY):
    treeReg = DecisionTreeRegressor()
    treeReg.fit(trainX, trainY)
    confidence = treeReg.score(testX, testY)
    print("TR Prediction Score {}".format(confidence))
    predictions = treeReg.predict(testX)
    return predictions


def linearRegression(trainX, testX, trainY, testY):
    linReg = LinearRegression()
    linReg.fit(trainX, trainY)
    confidence = linReg.score(testX, testY)
    print("LR Prediction Score {}".format(confidence))
    return linReg


# def xgBoost(trainX, testX, trainY, testY):
#     xgbReg = XGBRegressor()
#     xgbReg.fit(trainX, trainY, verbose=False)
#     confidence = xgbReg.score(testX, testY)
#     print("XGB Regressor Prediction Score {}".format(confidence))
#     return 0


def trainModel():
    print("HELLO")
    train, test = readInData()

    print("The Shape of the training data : {} ".format(train.shape))
    print("The Shape of the testing data : {} ".format(test.shape))

    train, test, trainId, testId = dropIdColumn(train, test)

    train = applyNormalDistributionToSales(train)

    trainLabels = getTrainingLabels(train)

    # checkNAValues(train)
    test = fillNaValues(test)
    train = fillNaValues(train)
    train = removeColumns(train)
    test = removeColumns(test)
    # checkNAValues(train)
    train = dropSalesPrice(train)
    train.rename(columns={'1stFlrSF': 'OneFlrSF',
                          '2ndFlrSF': 'TwoFlrSF',
                          '3SsnPorch': 'ThreeSsnPorch'},
                 inplace=True)
    export_csv = train.to_csv(r'prediction/pricePrediction/trainingData.csv', index=None, header=True)
    test = ordinalEncoder(test)
    train = ordinalEncoder(train)
    convertDtypes(train)
    # scatterPlot(train)

    # corr_matrix = train.corr()
    # print(corr_matrix['SalePrice'].sort_values(ascending=False))

    trainX, testX, trainY, testY = trainTestSplit(train, trainLabels)
    predictionsSVM = supportVectorRegression(trainX, testX, trainY, testY)
    predictionsTR = treeRegression(trainX, testX, trainY, testY)
    linReg = linearRegression(trainX, testX, trainY, testY)
    filename = 'finalized_model.sav'
    pickle.dump(linReg, open(filename, 'wb'))
    # predictionsXGBR = xgBoost(trainX, testX, trainY, testY)
    # printPredictionToCSV(testId, np.exp(predictionsLR))
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(testX, testY)
    print("RESULT ", result)


def convertDtypes(data):
    columnDtypes = {}
    for col in list(data.columns):
        col_dtype = re.sub(r'\d+', '', type(data[col][0]).__name__)
        try:
            if col_dtype == "int":
                columnDtypes[col] = "int"
            elif col_dtype == "float":
                columnDtypes[col] = "float"
            else:
                columnDtypes = "float"
        except:
            continue
        pickle.dump(columnDtypes, open("dtypes.sav", "wb"))


def predictPrice(request):
    filename = 'finalized_model.sav'
    for key, val in request.items():
        print(key, " ", val)
    dataFrame = pd.DataFrame(columns=list(request.keys()))
    dataFrame.loc[0] = list(request.values())
    # valueStore = []
    # for values in list(request.values()):
    #     print(values[0])
    #     valueStore.append(values[0])
    # print(dataFrame.shape)
    # print("LENGTH", len(valueStore))
    columnDtypes = pickle.load(open("dtypes.sav", "rb"))
    for col in columnDtypes:
        colDtype = columnDtypes[col]
        try:
            if colDtype == "int":
                dataFrame[col] = int(dataFrame[col])
            elif colDtype == "float":
                dataFrame[col] = float(dataFrame[col])
        except:
            continue
    print(dataFrame)
    loaded_model = pickle.load(open(filename, 'rb'))
    data = fillNaValues(dataFrame)
    train = pd.read_csv("prediction/pricePrediction/trainingData.csv", keep_default_na=False)
    print(data.dtypes)
    print(train.dtypes)
    fullset = pd.concat([train, data], sort=False)
    print(fullset)
    export_csv = fullset.to_csv(r'prediction/pricePrediction/predictionData.csv', index=None,
                                header=True)
    fullset = ordinalEncoder(fullset)
    data = fullset.tail(1)
    print(data)
    export_csv = data.to_csv(r'prediction/pricePrediction/predictionData.csv', index=None,
                             header=True)
    prediction = loaded_model.predict(data)
    return round(np.exp(prediction[0]), 2)
