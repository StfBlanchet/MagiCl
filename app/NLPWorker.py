# ! /usr/bin/env python3
# coding: utf-8

import numpy as np
import spacy

from spacy.lang.en.stop_words import STOP_WORDS


class NLPWorker:
    def __init__(self, text_df, corpus_name, narrow_lang, lemmatize):
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.text_df = text_df
        self.corpus_name = corpus_name
        self.narrow_lang = narrow_lang
        self.lemmatize = lemmatize

    def process(self):
        if self.narrow_lang is not None:
            # Narrow dataset to the chosen language
            self.text_df = self.text_df[self.text_df.lang == self.narrow_lang]
        if self.lemmatize:
            self.text_df[self.corpus_name] = self.text_df[
                self.corpus_name
            ].apply(
                lambda x: " ".join(
                    [
                        w.lemma_ if w.lemma_ != "-PRON-" else w.lower_
                        for w in self.nlp(x)
                    ]
                )
            )

        self.text_df[self.corpus_name] = self.text_df[self.corpus_name].apply(
            lambda x: " ".join(
                [
                    w.lower().strip()
                    for w in x.split(" ")
                    if w.lower().strip() not in STOP_WORDS
                    if w != ""
                ]
            )
        )

        self.text_df.replace("", np.nan, inplace=True)
        self.text_df.dropna(subset=[self.corpus_name], inplace=True)

        return self.text_df
