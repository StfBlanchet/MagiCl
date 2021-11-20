# ! /usr/bin/env python3
# coding: utf-8

import json
import pickle

from .Preprocessor import Preprocessor
from .NLPWorker import NLPWorker


class Classifier:
    def __init__(self, directory, file):
        self.dir = directory
        self.ref = file

    def run(self):
        preprocessed_data = Preprocessor(
            self.dir, self.ref, params={"pipe": "min"}
        ).run()
        (
            model,
            factor,
            narrow_lang,
            lemmatize,
            target_value,
        ) = self.__load_model()
        df = NLPWorker(
            preprocessed_data, factor, narrow_lang, lemmatize
        ).process()
        X = df[factor]
        other = [
            cat for cat in df.category.unique().tolist() if cat != target_value
        ][0]
        df["prediction"] = model.predict(X)
        df["predicted_category"] = df.prediction.apply(
            lambda x: target_value if x == 1 else other
        )

        return df[
            ["category", "clean_text", "prediction", "predicted_category"]
        ].to_json(orient="index")

    def __load_model(self):
        with open(
            f"data/{self.dir}/model/benchmark_{self.dir}.json", "r"
        ) as json_file:
            params = json.load(json_file)
        factor = params["1. Processing info"]["factor"]
        narrow_lang = params["1. Processing info"]["narrow_lang"]
        lemmatize = params["1. Processing info"]["lemmatized"]
        target_value = params["1. Processing info"]["target value"]

        with open(
            f"data/{self.dir}/model/model_{self.dir}.pickle", "rb"
        ) as model:
            model = pickle.load(model)

        return model, factor, narrow_lang, lemmatize, target_value
