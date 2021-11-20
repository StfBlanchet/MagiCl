# ! /usr/bin/env python3
# coding: utf-8

import json
import re
from urllib.parse import urlparse

import pandas as pd
from langdetect import detect
import numpy as np
from unidecode import unidecode

from .extractors import CAPS, CLEANER, EMAIL, EXCL, PHONE, QUEST, URL


class Preprocessor:
    def __init__(self, directory, file, params):
        self.dir = directory
        self.ref = file
        self.params = params
        self.pipe = "min"

    def run(self):
        df = self.__build_dataset()
        if self.params and self.params["pipe"] == "max":
            df = self.__extract_metadata(df)
        clean_text = self.__clean_text(df)
        df = self.__detect_language(clean_text)

        return df

    def __extract_metadata(self, df):
        df["text_len"] = df.text.str.count(" ") + 1
        extractors = [("links", URL), ("emails", EMAIL), ("phones", PHONE)]
        for col, pattern in extractors:
            df[col] = df.text.str.findall(pattern)
            df[col + "_rate"] = df[col].apply(lambda x: len(x))
            df[col + "_rate"] = df[col + "_rate"] / df.text_len
        df["domains"] = df.links.apply(
            lambda x: [urlparse(link).netloc for link in x]
        )

        df = self.__extract_emphasis(df)
        self.__save_dataset(df)

        return df

    def __extract_emphasis(self, df):
        # Place spaces before emphasis markers
        # so to allow a reliable extraction
        df["text"] = df.text.apply(
            lambda x: x.replace("!", " !").replace("?", " ?")
        )
        extractors = [
            ("cap_emphasis", CAPS),
            ("quest_emphasis", QUEST),
            ("excl_emphasis", EXCL),
        ]
        for col, pattern in extractors:
            df[col] = df.text.str.findall(pattern)
            df[col + "_rate"] = df[col].apply(lambda x: len(x))
            df[col + "_rate"] = df[col + "_rate"] / df.text_len
        df["emphasis"] = (
            df.cap_emphasis_rate
            + df.quest_emphasis_rate
            + df.excl_emphasis_rate
        )

        return df

    def __clean_text(self, df):
        df["clean_text"] = df.text.apply(lambda x: re.sub(CLEANER, "", x))
        cleaner = r"[^a-zA-Z]+"
        df["clean_text"] = df.clean_text.apply(
            lambda x: " ".join([re.sub(cleaner, "", t) for t in x.split(" ")])
        )
        df = df.replace({None: ""})
        df.replace("", np.nan, inplace=True)
        df.dropna(subset=["clean_text"], inplace=True)

        return df

    def __detect_language(self, df):
        df["lang"] = [
            detect(df.clean_text[i]) if len(df.clean_text[i]) > 3 else "none"
            for i, col in df.iterrows()
        ]
        df["clean_text"] = df.clean_text.apply(lambda x: unidecode(x))

        return df

    def __save_dataset(self, df):
        parsed_df = json.loads(df.to_json())
        with open(
            f"data/{self.dir}/dataset/clean_{self.ref}.json", "w"
        ) as file:
            json.dump(parsed_df, file)

    def __build_dataset(self):
        df = pd.read_csv(f"data/{self.dir}/dataset/{self.ref}.csv")
        df.drop_duplicates(subset=["text"])
        df.replace("", np.nan, inplace=True)
        df.dropna(subset=["text"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df
