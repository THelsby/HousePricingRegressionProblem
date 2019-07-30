import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns


def scatterPlot(dataset):
    for data in dataset.columns:
        plt.scatter(dataset[data], dataset["SalePrice"])
        plt.title(data)
        plt.savefig("Plots/" + data + ".png")
        plt.show()
        plt.close()


def readInData():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
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
    data = removeColumns(data)
    return data


def removeColumns(data):
    data = data.drop(['Utilities'], axis=1)
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
    columns = ["MSZoning", "Functional", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType"]
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


def trainTestSplit(trainData, trainLabels):
    TrainX, TestX, TrainY, TestY = train_test_split(trainData, trainLabels, test_size=0.33, random_state=42)
    return TrainX, TestX, TrainY, TestY


def supportVectorRegression(trainX, testX, trainY, testY):
    svr = SVR(gamma="scale", C=1.0, epsilon=0.001)
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
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(trainX, trainY)
    confidence = tree_reg.score(testX, testY)
    print("TR Prediction Score {}".format(confidence))
    predictions = tree_reg.predict(testX)
    return predictions


def linearRegression(trainX, testX, trainY, testY):
    lin_reg = LinearRegression()
    lin_reg.fit(trainX, trainY)
    confidence = lin_reg.score(testX, testY)
    print("LR Prediction Score {}".format(confidence))
    predictions = lin_reg.predict(testX)
    return predictions


def dataPipeline():
    train, test = readInData()

    print("The Shape of the training data : {} ".format(train.shape))
    print("The Shape of the testing data : {} ".format(test.shape))

    train, test, trainId, testId = dropIdColumn(train, test)

    train = applyNormalDistributionToSales(train)

    trainLabels = getTrainingLabels(train)

    # checkNAValues(train)
    test = fillNaValues(test)
    train = fillNaValues(train)
    # checkNAValues(train)
    test = ordinalEncoder(test)
    train = ordinalEncoder(train)
    scatterPlot(train)

    corr_matrix = train.corr()
    print(corr_matrix['SalePrice'].sort_values(ascending=False))
    train = dropSalesPrice(train)

    trainX, testX, trainY, testY = trainTestSplit(train, trainLabels)
    predictionsSVM = supportVectorRegression(trainX, testX, trainY, testY)
    predictionsTR = treeRegression(trainX, testX, trainY, testY)
    predictionsLR = linearRegression(trainX, testX, trainY, testY)
    # printPredictionToCSV(testId, np.exp(predictionsLR))


dataPipeline()
