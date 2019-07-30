import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
import seaborn as sns

color = sns.color_palette()
sns.set_style('darkgrid')
import warnings


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# print(train.head(5))
# print(test.head(5))

print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

trainId = train['Id']
testId = test['Id']

train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)

print("\nThe train data size after dropping Id feature is : {} ".format(train.shape))
print("The test data size after dropping Id feature is : {} ".format(test.shape))


def createScatterPlot(column1, column2):
    fig, ax = plt.subplots()
    ax.scatter(x=train[column1], y=train[column2])
    plt.ylabel(column2, fontsize=13)
    plt.xlabel(column1, fontsize=13)
    plt.show()


createScatterPlot("GrLivArea", "SalePrice")

train = train.drop(train[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)].index)
createScatterPlot("GrLivArea", "SalePrice")

sns.distplot(train['SalePrice'], fit=norm)
(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
           loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

train["SalePrice"] = np.log1p(train["SalePrice"])
sns.distplot(train['SalePrice'], fit=norm);
(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
           loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

ntrain = train.shape[0]
ntest = test.shape[0]
yTrain = train.SalePrice.values
allData = pd.concat((train, test)).reset_index(drop=True)
allData.drop(["SalePrice"], axis=1, inplace=True)
print("all_data size is : {}".format(allData.shape))

allDataNa = (allData.isnull().sum() / len(allData)) * 100
allDataNa = allDataNa.drop(allDataNa[allDataNa == 0].index).sort_values(ascending=False)[:30]
missingData = pd.DataFrame({'Missing Ratio': allDataNa})
print(missingData.head(20))

f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=allDataNa.index, y=allDataNa)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
plt.show()

corrmat = train.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()

allData["PoolQC"] = allData["PoolQC"].fillna("None")
allData["MiscFeature"] = allData["MiscFeature"].fillna("None")
allData["Alley"] = allData["Alley"].fillna("None")
allData["Fence"] = allData["Fence"].fillna("None")
allData["FireplaceQu"] = allData["FireplaceQu"].fillna("None")
allData["LotFrontage"] = allData.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    allData[col] = allData[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    allData[col] = allData[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    allData[col] = allData[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    allData[col] = allData[col].fillna('None')
allData["MasVnrType"] = allData["MasVnrType"].fillna("None")
allData["MasVnrArea"] = allData["MasVnrArea"].fillna(0)
allData['MSZoning'] = allData['MSZoning'].fillna(allData['MSZoning'].mode()[0])
allData = allData.drop(['Utilities'], axis=1)
allData["Functional"] = allData["Functional"].fillna("Typ")
allData['Electrical'] = allData['Electrical'].fillna(allData['Electrical'].mode()[0])
allData['KitchenQual'] = allData['KitchenQual'].fillna(allData['KitchenQual'].mode()[0])
allData['Exterior1st'] = allData['Exterior1st'].fillna(allData['Exterior1st'].mode()[0])
allData['Exterior2nd'] = allData['Exterior2nd'].fillna(allData['Exterior2nd'].mode()[0])
allData['SaleType'] = allData['SaleType'].fillna(allData['SaleType'].mode()[0])
allData['MSSubClass'] = allData['MSSubClass'].fillna("None")

allDataNa = (allData.isnull().sum() / len(allData)) * 100
allDataNa = allDataNa.drop(allDataNa[allDataNa == 0].index).sort_values(ascending=False)
missingData = pd.DataFrame({'Missing Ratio': allDataNa})
print(missingData.head())
