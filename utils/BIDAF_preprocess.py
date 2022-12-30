import json, pickle, re, os, string, typing, gc
import pandas as pd
import numpy as np
import config
import nltk, spacy
import torch
from collections import Counter
nlp = spacy.load('en_core_web_sm')


def load_json(path):
    with open(path, 'r') as f:
        content = json.load(f)
    return content


def parse_data(data: dict) -> list:
    data = data['data']
    qa_pair_lst = []
    for element in data:
        for para in element['paragraphs']:
            context = para['context']
            for qa_pair in para['qas']:
                id = qa_pair['id']
                question = qa_pair['question']
                ans = qa_pair['answers']
                for an in ans:
                    answer = an['text']
                    answer_start = an['answer_start']
                    answer_end = answer_start + len(answer)
                    qa_pair_dict = {}
                    qa_pair_dict['id'] = id
                    qa_pair_dict['context'] = context
                    qa_pair_dict['question'] = question
                    qa_pair_dict['answer'] = answer
                    qa_pair_dict['label'] = [answer_start, answer_end]
                    qa_pair_lst.append(qa_pair_dict)

    return qa_pair_lst


def preprocess_df(qa_pair_lst):
    df = pd.DataFrame(qa_pair_lst)

    def to_lower(text):
        return text.lower()

    df.context = df.context.apply(to_lower)
    df.question = df.question.apply(to_lower)
    df.answer = df.answer.apply(to_lower)

    return df


def parse_df(path):
    data = load_json(path)
    qa_pair_lst = parse_data(data)
    processed_df = preprocess_df(qa_pair_lst)

    return processed_df


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
    word2idx_file = './data/vocab_word2idx.pkl'
    idx2word_file = './data/vocab_idx2word.pkl'
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
    char2idx_file = './data/char2idx.pkl'
    idx2char_file = './data/idx2char.pkl'

    pickle.dump(char2idx, open(char2idx_file, 'wb'))
    pickle.dump(idx2char, open(idx2char_file, 'wb'))

    return char2idx, idx2char, char_vocab


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


def postprocess_df(df, word2idx, idx2word, prex_filename='train'):
    def text2ids(text, word2idx):
        words = [w.text for w in nlp(text, disable=['parser', 'tagger', 'ner'])]
        ids = [word2idx[w] for w in words]

        return ids

    df['context_ids'] = df.context.apply(text2ids, word2idx=word2idx)
    df['question_ids'] = df.question.apply(text2ids, word2idx=word2idx)
    df['lable_ids'] = df.label.apply(index_answer, axis=1, idx2word=idx2word)
    df.to_pickle(f'{prex_filename}_df.pkl')

    return df


def load_pretrain_embedding(embedding_file):
    with open(embedding_file, 'r', encoding='utf-8') as f:
        word_embedding_dict = {}
        lines = f.readlines()
        for line in lines:
            word, vec = line.split()[0], line.split()[1:]
            word_embedding_dict[word] = np.array(vec, dtype=np.float32)

    print('Total number of words in Glove txt:  ', len(word_embedding_dict))

    return word_embedding_dict


def create_embedding_matrix(word2idx, embedding_dict):
    vocab_size = len(word2idx)
    embedding_dim = embedding_dict[list(embedding_dict.keys())[0]].shape[0]
    print('embedding dimension: ', embedding_dim)
    embedding_matrix = np.zeros((vocab_size + 2, embedding_dim))
    for word, idx in word2idx.items():
        if embedding_dict.get(word) is not None:
            embedding_matrix[idx] = embedding_dict[word]
        elif word == '<pad>':
            embedding_matrix[idx] = np.random.randn(1, embedding_dim)

    for word, idx in word2idx.items():
        if word == '<unk>':
            embedding_matrix[idx] = np.mean(embedding_matrix, axis=0, keepdims=True)

    glove_mat_file = './data/glove_matrix.pkl'
    pickle.dump(embedding_matrix, open(glove_mat_file, 'wb'))

    return embedding_matrix


# def simplied_build_char_vocab(vocab_text):
#
#     chars = []
#     for sent in vocab_text:
#         for ch in sent:
#             chars.append(ch)
#
#     char_counter = Counter(chars)
#     char_vocab = [char for char, count in char_counter.items() if count >= 20]
#     char_vocab.insert(0, '<unk>')
#     char_vocab.insert(1, '<pad>')
#     char2idx = {char: idx for idx, char in enumerate(char_vocab)}
#     print(f"char2idx-length: {len(char2idx)}")
#     char2idx_file = './data/char2idx.pkl'
#
#     return char2idx, char_vocab


if __name__ == '__main__':
    train_path = config.train_file
    dev_path = config.dev_file

    train_df = parse_df(train_path)
    dev_df = parse_df(dev_path)

    vocab_text = gather_text(train_df, dev_df)

    word2idx, idx2word, _ = build_word_vocab(vocab_text)
    char2idx, idx2char, _ = build_char_vocab(vocab_text)

    train_df = postprocess_df(train_df, word2idx, idx2word, prex_filename='train')
    dev_df = postprocess_df(dev_df, word2idx, idx2word, prex_filename='dev')

    glove_path = config.glove_path
    glove_dict = load_pretrain_embedding(glove_path)
    embedding_matrix = create_embedding_matrix(word2idx, glove_dict)

    









