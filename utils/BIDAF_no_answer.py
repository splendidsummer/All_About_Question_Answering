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


def preprocess_df(df):

    def to_lower(text):
        return text.lower()

    df.context = df.context.apply(to_lower)
    df.question = df.question.apply(to_lower)
    df['answer'] = df.answers.apply(lambda an: an['text'].lower())
    df['label'] = df.answers.apply(lambda an: [0 if an['text'] is not None else an['answer_start'],
                                               0 if an['text'] is not None else an['answer_start'] + len(an['text'])])
    df.drop(columns=['answers'])

    return df


def gather_text(train_df, dev_df):
    train_context = train_df.context
    train_question = train_df.question

    dev_context = dev_df.context
    dev_question = dev_df.question
    text = []
    total = 0

    for content in [train_context, train_question, dev_context, dev_question]:
        unique_content = list(content.unique())
        total += content.nunique()
        text.extend(unique_content)

    assert total == len(text)

    return text


def build_word_vocab(text):
    words = []
    for sent in text:
        for word in nlp(sent, disable=['parser', 'tagger', 'ner']):
            words.append(word.text)

    word_counter = Counter(words)
    word_vocab = sorted(word_counter, key=word_counter.get, reverse=True)
    print(f"raw-vocab: {len(word_vocab)}")
    word_vocab.insert(0, '<unk>')
    word_vocab.insert(1, '<pad>')
    print(f"vocab-length: {len(word_vocab)}")
    word2idx = {word: idx for idx, word in enumerate(word_vocab)}
    print(f"word2idx-length: {len(word2idx)}")
    idx2word = {v: k for k, v in word2idx.items()}
    word2idx_file = config.full_data_dir + 'vocab_word2idx.pkl'
    idx2word_file = config.full_data_dir + 'vocab_idx2word.pkl'
    pickle.dump(word2idx, open(word2idx_file, 'wb'))
    pickle.dump(idx2word, open(idx2word_file, 'wb'))

    return word2idx, idx2word, word_vocab


def build_char_vocab(vocab_text):

    chars = []
    for sent in vocab_text:
        for ch in sent:
            chars.append(ch)

    char_counter = Counter(chars)
    char_vocab = sorted(char_counter, key=char_counter.get, reverse=True)
    print(f"raw-char-vocab: {len(char_vocab)}")
    high_freq_char = [char for char, count in char_counter.items() if count >= 20]
    char_vocab = list(set(char_vocab).intersection(set(high_freq_char)))
    print(f"char-vocab-intersect: {len(char_vocab)}")
    char_vocab.insert(0, '<unk>')
    char_vocab.insert(1, '<pad>')
    char2idx = {char: idx for idx, char in enumerate(char_vocab)}
    idx2char = {idx: char for idx, char in enumerate(char_vocab)}
    print(f"char2idx-length: {len(char2idx)}")
    char2idx_file = config.data_dir + 'char2idx.pkl'
    idx2char_file = config.data_dir + 'idx2char.pkl'

    pickle.dump(char2idx, open(char2idx_file, 'wb'))
    pickle.dump(idx2char, open(idx2char_file, 'wb'))

    return char2idx, idx2char, char_vocab


def get_error_indices(df, idx2word):
    start_value_error, end_value_error, assert_error = test_indices(df, idx2word)
    err_idx = start_value_error + end_value_error + assert_error
    err_idx = set(err_idx)
    print(f"Number of error indices: {len(err_idx)}")

    return err_idx


def test_indices(df, idx2word):
    '''
    Performs the tests mentioned above. This method also gets the start and end of the answers
    with respect to the context_ids for each example.

    :param dataframe df: SQUAD df
    :param dict idx2word: inverse mapping of token ids to words
    :returns
        list start_value_error: example idx where the start idx is not found in the start spans
                                of the text
        list end_value_error: example idx where the end idx is not found in the end spans
                              of the text
        list assert_error: examples that fail assertion errors. A majority are due to the above errors

    '''

    start_value_error = []
    end_value_error = []
    assert_error = []
    for index, row in df.iterrows():

        # 这里要修改出来需要考虑 没有answer的情况

        answer_tokens = [w.text for w in nlp(row['answer'], disable=['parser', 'tagger', 'ner'])]

        start_token = answer_tokens[0]
        end_token = answer_tokens[-1]

        context_span = [(word.idx, word.idx + len(word.text))
                        for word in nlp(row['context'], disable=['parser', 'tagger', 'ner'])]

        starts, ends = zip(*context_span)

        answer_start, answer_end = row['label']

        try:
            start_idx = starts.index(answer_start)
        except:

            start_value_error.append(index)
        try:
            end_idx = ends.index(answer_end)
        except:
            end_value_error.append(index)

        try:
            assert idx2word[row['context_ids'][start_idx]] == answer_tokens[0]
            assert idx2word[row['context_ids'][end_idx]] == answer_tokens[-1]
        except:
            assert_error.append(index)

    return start_value_error, end_value_error, assert_error


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


def postprocess_df(df, word2idx, idx2word, char2idx, prex_filename='train'):
    def text2ids(text, word2idx):
        words = [w.text for w in nlp(text, disable=['parser', 'tagger', 'ner'])]
        ids = [word2idx.get(w, word2idx['<unk>']) for w in words]

        return ids

    def text2charids(text, char2idx):
        words = [w.text for w in nlp(text, disable=['parser', 'tagger', 'ner'])]
        ids = [[char2idx.get(c, char2idx['<unk>']) for c in w] for w in words]
        return ids

    df['context_ids'] = df.context.apply(text2ids, word2idx=word2idx)
    df['question_ids'] = df.question.apply(text2ids, word2idx=word2idx)
    df['context_char_ids'] = df.context.apply(text2charids, char2idx=char2idx)
    df['question_char_ids'] = df.question.apply(text2charids, char2idx=char2idx)
    df_error = get_error_indices(df, idx2word)
    df.drop(df_error, inplace=True)
    df['lable_ids'] = df.apply(index_answer, axis=1, idx2word=idx2word)
    df.to_pickle(config.data_dir + f'{prex_filename}_df.pkl')

    return df


if __name__ == '__main__':
    train_path = config.train_file
    dev_path = config.dev_file

    vocab_text = gather_text(train_df, dev_df)

    word2idx, idx2word, _ = build_word_vocab(vocab_text)
    char2idx, idx2char, _ = build_char_vocab(vocab_text)

    train_df = postprocess_df(train_df, word2idx, idx2word, char2idx, prex_filename='train')
    dev_df = postprocess_df(dev_df, word2idx, idx2word, char2idx, prex_filename='dev')

    save_features(train_df.context_ids, train_df.context_char_ids, train_df.question_ids,
                  train_df.question_char_ids, train_df.label_ids)
    save_features(dev_df.context_ids, dev_df.context_char_ids, dev_df.question_ids,
                  dev_df.question_char_ids, dev_df.label_ids, prex='dev')

    glove_path = config.glove_path
    glove_dict = load_pretrain_embedding(glove_path)
    embedding_matrix = create_embedding_matrix(word2idx, glove_dict)


