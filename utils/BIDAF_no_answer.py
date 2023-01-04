import json, pickle, re, os, string, typing, gc
import pandas as pd
import numpy as np
import config
import nltk, spacy
from collections import Counter
import transformers
from datasets import load_dataset, load_metric
from datasets import ClassLabel, Sequence


def create_df():
    datasets = load_dataset("squad_v2")
    data = datasets['train']
    df = pd.DataFrame(data[list(range(len(data)))])

    for column, typ in data.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
            # print(typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])

    return df


def preprocess_df(df):

    def to_lower(text):
        return text.lower()

    df.context = df.context.apply(to_lower)
    df.question = df.question.apply(to_lower)
    df['answer'] = df.answers.apply(lambda an: an['text'].lower())
    df['label'] = df 



    return df



def index_answer(row, idx2word):
    '''
    Takes in a row of the dataframe or one training example and
    returns a tuple of start and end positions of answer by calculating
    spans.
    '''

    context_span = [(word.idx, word.idx + len(word.text)) for word in
                    nlp(row.context, disable=['parser', 'tagger', 'ner'])]

    starts, ends = zip(*context_span)

    answer_start, answer_end = row.label
    start_idx = starts.index(answer_start)

    end_idx = ends.index(answer_end)

    ans_toks = [w.text for w in nlp(row.answer, disable=['parser', 'tagger', 'ner'])]
    ans_start = ans_toks[0]
    ans_end = ans_toks[-1]
    assert idx2word[row.context_ids[start_idx]] == ans_start
    assert idx2word[row.context_ids[end_idx]] == ans_end

    return [start_idx, end_idx]


