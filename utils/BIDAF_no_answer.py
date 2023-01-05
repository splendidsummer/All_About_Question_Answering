import json, pickle, re, os, string, typing, gc
import pandas as pd
import numpy as np
import config
import nltk, spacy
from collections import Counter
import transformers
from datasets import load_dataset, load_metric
from datasets import ClassLabel, Sequence
nlp = spacy.load('en_core_web_sm')
from BIDAF_preprocess import gather_text, build_word_vocab, build_char_vocab, \
     postprocess_df, save_noanswer_features


def create_df(dataset_type='train'):
    datasets = load_dataset("squad_v2")
    data = datasets[dataset_type]
    df = pd.DataFrame(data[list(range(len(data)))])

    for column, typ in data.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
            # print(typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])

    return df


def preprocess_noanswer_df(df):

    def to_lower(text):
        return text.lower()

    df.context = df.context.apply(to_lower)
    df.question = df.question.apply(to_lower)
    df['answer'] = df.answers.apply(lambda an: an['text'][0].lower() if len(an['text']) != 0 else '')
    df['label'] = df.answers.apply(lambda an: [0 if len(an['text']) == 0 else an['answer_start'][0],
                                               0 if len(an['text']) == 0 else an['answer_start'][0] + len(an['text'])])
    df.drop(columns=['answers'])

    return df


if __name__ == '__main__':
    train_path = config.train_file
    dev_path = config.dev_file
    train_df = create_df()
    dev_df = create_df('validation')
    train_df = preprocess_noanswer_df(train_df)
    dev_df = preprocess_noanswer_df(dev_df)

    vocab_text = gather_text(train_df, dev_df)
    word2idx, idx2word, _ = build_word_vocab(vocab_text)
    char2idx, idx2char, _ = build_char_vocab(vocab_text)

    word2idx_file = config.full_data_dir + 'vocab_word2idx.pkl'
    idx2word_file = config.full_data_dir + 'vocab_idx2word.pkl'
    pickle.dump(word2idx, open(word2idx_file, 'wb'))
    pickle.dump(idx2word, open(idx2word_file, 'wb'))

    char2idx_file = config.full_data_dir + 'char2idx.pkl'
    idx2char_file = config.full_data_dir + 'idx2char.pkl'
    pickle.dump(char2idx, open(char2idx_file, 'wb'))
    pickle.dump(idx2char, open(idx2char_file, 'wb'))

    train_df = postprocess_df(train_df, word2idx, idx2word, char2idx, prex_filename='train')
    train_df.to_pickle(config.full_data_dir + f'train_df.pkl')

    dev_df = postprocess_df(dev_df, word2idx, idx2word, char2idx, prex_filename='dev')
    dev_df.to_pickle(config.full_data_dir + f'dev_df.pkl')

    save_noanswer_features(train_df.context_ids, train_df.context_char_ids, train_df.question_ids,
                           train_df.question_char_ids, train_df.label_ids)
    save_noanswer_features(dev_df.context_ids, dev_df.context_char_ids, dev_df.question_ids,
                           dev_df.question_char_ids, dev_df.label_ids, prex='dev')
