# ! /usr/bin/env python3
# coding: utf-8

import json
import pickle

import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    jaccard_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

from .Preprocessor import Preprocessor
from .NLPWorker import NLPWorker
from .MLWorkers import WORKERS


class Modeler:
    def __init__(self, directory, file, params):
        self.dir = directory
        self.ref = file
        self.params = params

    def run(self):
        params = self.__set_params()
        preprocessed_data = self.__prepare_data(params)
        tokenized_data = self.__tokenize(preprocessed_data, params)
        if params["resampler"]:
            resampled_data = self.__resample(tokenized_data)
            model = self.__generate_model(resampled_data, params)
        else:
            model = self.__generate_model(tokenized_data, params)

        return model

    def __set_params(self):
        custom_params = list(self.params)
        params = {
            "target": self.params["target"],
            "target_value": self.params["target_value"],
            "factor": self.params["factor"],
            "pipe": self.params["pipe"] if "pipe" in custom_params else "min",
            "metric": self.params["metric"]
            if "metric" in custom_params
            else "f1",
            "lemmatizer": self.params["lemmatizer"]
            if "lemmatizer" in custom_params
            else False,
            "narrow_lang": self.params["narrow_lang"]
            if "narrow_lang" in custom_params
            else "en",
            "resampler": self.params["resampler"]
            if "resampler" in custom_params
            else True,
            "mode": self.params["mode"]
            if "mode" in custom_params
            else "benchmark",
        }
        if "vectorizer" in custom_params:
            params["vectorizer"] = (
                self.params["vectorizer"],
                WORKERS["vectorizers"][self.params["vectorizer"]],
            )
        else:
            params["vectorizer"] = ("tfidf", WORKERS["vectorizers"]["tfidf"])
        if "classifiers" in custom_params:
            clf_list = self.params["classifiers"].split(" ")
            params["classifiers"] = [
                (c, WORKERS["classifiers"][c]) for c in clf_list
            ]
        else:
            params["classifiers"] = [
                ("BNB", WORKERS["classifiers"]["BNB"]),
                ("CNB", WORKERS["classifiers"]["CNB"]),
                ("MNB", WORKERS["classifiers"]["MNB"]),
            ]

        return params

    def __prepare_data(self, params):
        preprocessed_data = Preprocessor(
            self.dir, self.ref, params={"pipe": params["pipe"]}
        ).run()
        preprocessed_data["target"] = preprocessed_data[params["target"]].apply(
            lambda x: 1 if x == params["target_value"] else 0
        )

        return preprocessed_data

    def __tokenize(self, preprocessed_data, params):
        tokenized_data = NLPWorker(
            preprocessed_data[[params["factor"], "lang", "target"]],
            params["factor"],
            params["narrow_lang"],
            params["lemmatizer"],
        ).process()

        return tokenized_data[[params["factor"], "target"]]

    def __resample(self, tokenized_data):
        # Mix over and under-sampling methods
        # so to limit data artificiality
        target_ratio = self.__compute_target_ratio(tokenized_data)
        if target_ratio < 0.85:
            df_maj = tokenized_data[tokenized_data.target == 0]
            df_min = tokenized_data[tokenized_data.target == 1]
            return self.__balance_samples(df_maj, df_min)
        elif target_ratio > 1.15:
            df_maj = tokenized_data[tokenized_data.target == 1]
            df_min = tokenized_data[tokenized_data.target == 0]
            return self.__balance_samples(df_maj, df_min)
        else:
            return tokenized_data

    def __balance_samples(self, df_maj, df_min):
        # 1. Enhance min sample up to 25% of maj sample
        balanced_min = resample(
            df_min, n_samples=int(len(df_maj) * 0.25), random_state=0
        )
        # 2. Reduce maj sample so to equal 75% more than min sample
        balanced_maj = resample(
            df_maj, n_samples=int(len(df_min) * 1.75), random_state=0
        )

        return pd.concat([balanced_min, balanced_maj])

    def __generate_model(self, ready_data, params):
        X = ready_data[params["factor"]]
        y = ready_data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        pipelines = {}
        performances = {
            "1. Processing info": {
                "factor": params["factor"],
                "target field": params["target"],
                "target value": params["target_value"],
                "initial size": len(ready_data),
                "preprocessing pipe": params["pipe"],
                "resampled": params["resampler"],
                "narrow_lang": params["narrow_lang"],
                "lemmatized": params["lemmatizer"],
                "train size": len(X_train),
                "test size": len(X_test),
                "balancing": self.__assess_balancing(ready_data, params),
                "vectorizer_type": params["vectorizer"][0],
                "vectorizer_parameters": "should be customizable",
            },
            "2. Best model": {},
            "3. Detailed classifiers": {},
        }

        for clf_name, clf in params["classifiers"]:
            grid_params = {"alpha": [0.01, 0.1, 0.5, 1.0, 10.0]}
            clf = GridSearchCV(
                clf,
                param_grid=grid_params,
                n_jobs=-1,
                cv=5,
                scoring=params["metric"],
            )

            pipeline = Pipeline(
                [("vectorizer", params["vectorizer"][1]), ("classifier", clf)]
            )
            pipeline.fit(X_train, y_train)
            yhat_train = pipeline.predict(X_train)
            yhat = pipeline.predict(X_test)

            pipelines[clf] = pipeline

            train_accuracy = accuracy_score(y_train, yhat_train)
            test_accuracy = accuracy_score(y_test, yhat)
            # if train_accuracy > test_accuracy --> consider over-fitting

            confusion = confusion_matrix(y_test, yhat).tolist()

            false_negative = json.loads(
                X_test[(y_test == 1) & (yhat == 0)].to_json(orient="records")
            )
            false_positive = json.loads(
                X_test[(y_test == 0) & (yhat == 1)].to_json(orient="records")
            )

            performances["3. Detailed classifiers"].update(
                {
                    clf_name: {
                        "best score": clf.best_score_,
                        "best parameters": clf.best_params_,
                        "confusion matrix": {
                            "1. TP": confusion[0][0],
                            "2. FN": confusion[1][0],
                            "2. FP": confusion[0][1],
                            "4. TN": confusion[1][1],
                            "false positive": false_positive,
                            "false negative": false_negative,
                        },
                        "precision": precision_score(y_test, yhat),
                        "recall": recall_score(y_test, yhat),
                        "f1": f1_score(y_test, yhat),
                        "train accuracy": train_accuracy,
                        "accuracy": test_accuracy,
                        "jaccard": jaccard_score(y_test, yhat),
                        "auc": roc_auc_score(y_test, yhat),
                        "log loss": log_loss(y_test, yhat),
                    }
                }
            )

        if params["mode"] == "single":
            with open(
                f"data/{self.dir}/model/{self.dir}.pickle", "wb"
            ) as model:
                pickle.dump(pipelines[params["classifiers"][0]], model)
            return performances
        else:
            return self.__benchmark_classifiers(performances, params)

    def __benchmark_classifiers(self, performances, params):
        classifier_data = {
            k: v[params["metric"]]
            for k, v in performances["3. Detailed classifiers"].items()
        }
        best_model = max(classifier_data, key=classifier_data.get)
        performances["2. Best model"] = {
            "metric": params["metric"],
            "model": best_model,
            "performances": performances["3. Detailed classifiers"][best_model],
        }

        with open(
            f"data/{self.dir}/model/benchmark_{self.dir}.json", "w"
        ) as file:
            json.dump(performances, file)

        return performances

    def __assess_balancing(self, ready_data, params):
        target_ratio = self.__compute_target_ratio(ready_data)
        balancing = "equal"
        if target_ratio > 1:
            balancing = (
                f"The model was trained with a {round((target_ratio - 1) * 100, 0)}% "
                f"over-representation of the {params['target_value']} values in the dataset."
            )
        elif target_ratio < 1:
            balancing = (
                f"The model was trained with a {round((1 - target_ratio) * 100, 2)}% "
                f"under-representation of the non-{params['target_value']} values in the dataset."
            )

        return balancing

    def __compute_target_ratio(self, dataset):
        return (
            dataset.target.value_counts()[1] / dataset.target.value_counts()[0]
        )
