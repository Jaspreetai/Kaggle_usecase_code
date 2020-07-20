import os
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
import seaborn as sns
from sklearn import model_selection


#returns current working directory
os.getcwd()
#changes working directory
os.chdir("C:\\Users\\jaspr\\Desktop")

#Data Collection
titanic_train = pd.read_csv("titanic_train.csv")

#EDA
titanic_train.shape
titanic_train.info()
sns.distplot(titanic_train['Fare'])
sns.factorplot(x="Survived", hue="Sex", data=titanic_train, kind="count", size=6)

#Data preprocessing
sum(titanic_train['Pclass'].isnull())
titanic_train.apply(lambda x : sum(x.isnull()))
pd.crosstab(index=titanic_train["Embarked"], columns="count")
titanic_train.Embarked[titanic_train['Embarked'].isnull()] = 'S'


titanic_train1 = titanic_train.copy()
le = preprocessing.LabelEncoder()
titanic_train1.Sex = le.fit_transform(titanic_train1.Sex)
titanic_train1.Embarked = le.fit_transform(titanic_train.Embarked)
titanic_train1.Pclass = le.fit_transform(titanic_train1.Pclass)


#Feature Engineering
X_train = titanic_train1[['Fare','Pclass','Sex','Embarked']]
y_train = titanic_train1['Survived']

#Model Building
dt_estimator = tree.DecisionTreeClassifier()

param_grid = {'max_depth':[13,14,15], 'min_samples_split':[2,3,4], 'min_samples_leaf':[1,2]}
dt_grid = model_selection.GridSearchCV(dt_estimator, param_grid, cv=7, n_jobs=5)
#build model using entire train data
dt_grid.fit(X_train,y_train)

dt_grid.grid_scores_
dt_grid.best_params_
dt_grid.best_score_



#pipeline for test data
titanic_test = pd.read_csv("test.csv")
titanic_test.apply(lambda x : sum(x.isnull()))
titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()


titanic_test1 = titanic_test.copy()
titanic_test1.Sex = le.fit_transform(titanic_test1.Sex)
titanic_test1.Embarked = le.fit_transform(titanic_test1.Embarked)
titanic_test1.Pclass = le.fit_transform(titanic_test1.Pclass)

X_test = titanic_test1[['Fare','Pclass','Sex','Embarked']]
titanic_test1['Survived'] = dt_grid.predict(X_test)
titanic_test1.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)