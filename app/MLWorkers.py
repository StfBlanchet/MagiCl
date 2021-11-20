# ! /usr/bin/env python3
# coding: utf-8


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB


WORKERS = {
    "vectorizers": {
        "tfidf": TfidfVectorizer(max_features=1500),
        "bow": CountVectorizer(),
    },
    "classifiers": {
        "MNB": MultinomialNB(),
        "BNB": BernoulliNB(),
        "CNB": ComplementNB(),
        # Other possible clf with different GridSearchCV params
        # 'CANB': CategoricalNB(),
        # 'GNB': GaussianNB(),
        # 'SVC': SVC(),
        # 'LR': LogisticRegression(),
        # 'RF': RandomForestClassifier(),
        # 'DT': DecisionTreeClassifier(), ...
    },
}
