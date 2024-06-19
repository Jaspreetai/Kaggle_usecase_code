import os
import pandas as pd
from sklearn import tree
from sklearn import model_selection
import numpy as np

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("C:\\Users\\jaspr\\Downloads")
#os.chdir("/home/algo/Downloads")

house_train = pd.read_csv("house_train.csv")
house_train.shape
house_train.info()

house_test = pd.read_csv("house_test.csv")
house_test.shape
house_test.info()

house_data = pd.concat([house_train, house_test],ignore_index=True)
house_data.drop(["Id","SalePrice"], 1, inplace=True)
house_data.shape
house_data.info()

#convert numerical columns to categorical type              
house_data['MSSubClass'] = house_data['MSSubClass'].astype('category')

#convert categorical columns to numeric type
ordinal_features1 = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond", "PoolQC", "FireplaceQu", "KitchenQual", "HeatingQC"]
#ordinal_features1 = [col for col in house_train if 'TA' in list(house_train[col])]
quality_dict = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
for feature in ordinal_features1:
    null_idx = house_data[feature].isnull()
    house_data.loc[null_idx, feature] = None 
    house_data[feature] = house_data[feature].map(quality_dict)


#handle missing data columns
total_missing = house_data.isnull().sum()
to_delete = total_missing[total_missing>0]
house_data.drop(list(to_delete.index), axis=1, inplace=True)
house_data.shape

numeric_cols = house_data.select_dtypes(include=['number']).columns
cat_cols = house_data.select_dtypes(exclude = ['number']).columns

house_data1 = pd.get_dummies(house_data, columns=cat_cols)
house_data1.shape

house_train1 = house_data1[:house_train.shape[0]]
house_test1 = house_data1[house_train.shape[0]:]
house_train['log_sale_price'] = np.log(house_train['SalePrice'])

X_train = house_train1
y_train = house_train['log_sale_price']

dt_estimator = tree.DecisionTreeRegressor(random_state=2017)

#evaluate using r^2
model_selection.cross_val_score(dt_estimator, X_train, y_train, cv=10,scoring="r2").mean()

#evaluate using rmse - 1
res = model_selection.cross_val_score(dt_estimator, X_train, y_train, cv=10,scoring="neg_root_mean_squared_error").mean()
result =(-res)


dt_estimator.fit(X_train, y_train)

