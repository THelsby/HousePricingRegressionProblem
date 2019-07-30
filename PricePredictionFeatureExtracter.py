import matplotlib
import pandas as pd
from sklearn import preprocessing, svm, impute, model_selection, metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def readInData(fileName):
    housingData = pd.read_csv(fileName)
    return housingData


def removeColumns(dataSet):
    return dataSet.drop(["Id", "Condition2", "RoofMatl", "RoofStyle", "Street", "Utilities", "BldgType", "3SsnPorch",
                         "BsmtFinSF2", "EnclosedPorch", "Functional", "Heating", "LowQualFinSF", "MiscVal", "MoSold",
                         "PoolArea"], axis=1)


# def changeNAToMedian(dataSet):
#     columnsToChange = ["LotFrontage", ]
#     return housingData


def scatterPlot(dataSet):
    for data in dataSet.columns:
        try:
            plt.scatter(dataSet[data], dataSet["SalePrice"])
            plt.title(data)
            plt.savefig("Plots/" + data + ".png")
            plt.show()
            plt.close()
        except:
            pass


def splitFeaturesAndLabels(dataSet):
    labels = dataSet.SalePrice
    housingDataWithoutLabels = dataSet.drop(columns="SalePrice")
    return labels, housingDataWithoutLabels


def num_missing(x):
    return sum(x.isnull())


def ordinalEncoder(dataset):
    columns = list(dataset.select_dtypes(include=[np.object]))
    enc = preprocessing.OrdinalEncoder()
    for column in columns:
        dataset[column] = enc.fit_transform(dataset[column].values.reshape(-1, 1))
    return dataset


def handleNullsNumber(dataset):
    columns = list(dataset.select_dtypes(include=[np.number]))
    for column in columns:
        imr = impute.SimpleImputer(strategy='median')
        imr = imr.fit(dataset[[column]])
        dataset[column] = imr.transform(dataset[[column]]).ravel()
    return dataset


def handleNullsText(dataset):
    columns = list(dataset.select_dtypes(include=[np.object]))
    for column in columns:
        imr = impute.SimpleImputer(strategy='most_frequent')
        imr = imr.fit(dataset[[column]])
        dataset[column] = imr.transform(dataset[[column]]).ravel()
    return dataset


def svmTest(features, labels):
    clf = svm.SVR(gamma="scale", C=1.0, epsilon=0.2)
    clf.fit(features, labels)
    return clf


housingDataTrain = readInData("train.csv")
housingDataTest = readInData("test.csv")

Ids = housingDataTest["Id"]
housingDataTrain = removeColumns(housingDataTrain)
housingDataTest = removeColumns(housingDataTest)
labelsTrain, housingDataWithoutLabelsTrain = splitFeaturesAndLabels(housingDataTrain)
housingDataTrain, housingDataValidation, housingDataTrainLabels, housingDataValidationLabels = model_selection.train_test_split(housingDataWithoutLabelsTrain, labelsTrain, test_size=0.25, random_state=42)


housingDataNullsToMeanTrain = handleNullsNumber(housingDataTrain)
housingDataNullsToMeanValidation = handleNullsNumber(housingDataValidation)
housingDataNullsToMeanTest = handleNullsNumber(housingDataTest)

housingDataNoNullsTrain = handleNullsText(housingDataNullsToMeanTrain)
housingDataNoNullsValidation = handleNullsText(housingDataNullsToMeanValidation)
housingDataNoNullsTest = handleNullsText(housingDataNullsToMeanTest)

# print("Missing Values Per Column:")
# print(housingDataNoNullsTrain.apply(num_missing, axis=0))
housingDataNormalisedTrain = ordinalEncoder(housingDataNoNullsTrain)
housingDataNormalisedValidation = ordinalEncoder(housingDataNoNullsValidation)
housingDataNormalisedTest = ordinalEncoder(housingDataNoNullsTest)

# print(housingDataNormalisedTrain.head())
# print(housingDataWithoutLabelsTrain.shape)
# print(housingDataNormalisedTest.head())
# print(housingDataNormalisedTest.shape)
svr = svmTest(housingDataNormalisedTrain, housingDataTrainLabels)
predictions = svr.predict(housingDataNormalisedValidation)
confidence = svr.score(housingDataValidation, housingDataValidationLabels)
print(confidence)
print(metrics.mean_squared_log_error(housingDataValidationLabels, predictions))

predictionsTest = svr.predict(housingDataNormalisedTest)

res = pd.DataFrame({"Id":Ids, "SalePrice": np.exp(predictionsTest)})
res.to_csv("predictions.csv", index=False)

scatterPlot(housingDataNormalisedTrain)


# print(housing_data.head())
#
# Labels = housing_data.SalePrice
#
# housing_data_without_labels = housing_data.drop(columns="SalePrice")
#
# print(housing_data_without_labels)
#
# print(housing_data_without_labels["YrSold"])


# x_array = np.array(housing_data_without_labels['MSZoning'])
# normalized_X = preprocessing.normalize([x_array])
# print(normalized_X)

# ordinalEncoder(housing_data_without_labels)
