import os
import pandas as pd
from sklearn.feature_extraction import text
from bs4 import BeautifulSoup
import re
from sklearn import tree
from sklearn import model_selection

os.getcwd()
os.chdir("C:\\Users\\jaspr\\Downloads\\labeledTrainData.tsv")

    
movie_train=pd.read_csv("labeledTrainData.tsv", delimiter="\t", header=0, quoting=3)

movie_train.shape
movie_train.info()
movie_train.loc[0:4,'review']


def preprocess_review(review):        #
        # 1. Remove HTML
        review_text = BeautifulSoup(review).get_text()
        #
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case
        return review_text.lower()

def tokenize(review):
    return review.split()

ngram_vectorizer = text.CountVectorizer(preprocessor = preprocess_review, \
                                  ngram_range=(1, 2),  \
                                  tokenizer = tokenize,    \
                                  stop_words = 'english',   \
                                  max_features = 9000)

#transform the reviews to count vectors(dtm)
features = ngram_vectorizer.fit_transform(movie_train.loc[0:3,'review']).toarray()


#get the mapping between the term features and dtm column index
ngram_vectorizer.vocabulary_
#get the feature names
vocab = ngram_vectorizer.get_feature_names()


X_train=features
y_train=movie_train['sentiment']

dt=tree.DecisionTreeClassifier()
param_grid={'max_depth':[25,44,57], 'min_samples_split':[5,6,7], 'min_samples_leaf':[4,5]}
dt_grid=model_selection.GridSearchCV(dt,param_grid,cv=7,n_jobs=5)
dt_grid.fit(features,y_train)

dt_grid.grid_scores_
dt_grid.best_params_
dt_grid.best_score_






