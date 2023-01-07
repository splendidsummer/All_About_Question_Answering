import json, pickle, re, os, string, typing, gc
import pandas as pd
import numpy as np
import config
import spacy
from collections import Counter
nlp = spacy.load('en_core_web_sm')


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
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

    # pickle.dump(char2idx, open(char2idx_file, 'wb'))
    # pickle.dump(idx2char, open(idx2char_file, 'wb'))

    return char2idx, idx2char, char_vocab


def get_error_indices(df, idx2word):
    start_value_error, end_value_error, assert_error, id_error = test_indices(df, idx2word)
    err_idx = start_value_error + end_value_error + assert_error
    id_error = set(id_error)
    err_idx = set(err_idx)
    assert len(err_idx) == len(id_error)
    print(f"Number of error indices: {len(err_idx)}")
    return err_idx, id_error


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
    id_error = []
    for index, row in df.iterrows():

        answer_tokens = [w.text for w in nlp(row['answer'], disable=['parser', 'tagger', 'ner'])]
        context_span = [(word.idx, word.idx + len(word.text))
                        for word in nlp(row['context'], disable=['parser', 'tagger', 'ner'])]

        starts, ends = zip(*context_span)

        answer_start, answer_end = row['label']

        if answer_start == answer_end == 0:
            continue

        try:
            start_idx = starts.index(answer_start)
        except:

            start_value_error.append(index)
            id_error.append(row['id'])
        try:
            end_idx = ends.index(answer_end)
        except:
            end_value_error.append(index)
            id_error.append(row['id'])

        try:
            assert idx2word[row['context_ids'][start_idx]] == answer_tokens[0]
            assert idx2word[row['context_ids'][end_idx]] == answer_tokens[-1]
        except:
            assert_error.append(index)
            id_error.append(row['id'])

    return start_value_error, end_value_error, assert_error, id_error


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
    if answer_start == answer_end == 0:
        return [answer_start, answer_end]

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
    df.to_csv('tmp_df.csv')

    df['question_ids'] = df.question.apply(text2ids, word2idx=word2idx)
    df['context_char_ids'] = df.context.apply(text2charids, char2idx=char2idx)
    df['question_char_ids'] = df.question.apply(text2charids, char2idx=char2idx)

    df_error, id_error = get_error_indices(df, idx2word)
    df.drop(df_error, inplace=True)
    df['label_ids'] = df.apply(index_answer, axis=1, idx2word=idx2word)
    # json.dump(id_error, open('id_error.txt', 'w'))
    for i in id_error:
        print(i)

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
    num_valid_word = 0
    for word, idx in word2idx.items():
        if embedding_dict.get(word) is not None:
            embedding_matrix[idx] = embedding_dict[word]
            num_valid_word += 1
        elif word == '<pad>':
            embedding_matrix[idx] = np.random.randn(1, embedding_dim)

    for word, idx in word2idx.items():
        if word == '<unk>':
            embedding_matrix[idx] = np.mean(embedding_matrix, axis=0, keepdims=True)

    glove_mat_file = config.data_dir + 'glove_matrix.pkl'
    pickle.dump(embedding_matrix, open(glove_mat_file, 'wb'))
    print('number of valid word is ', num_valid_word )

    return embedding_matrix


def save_features(context_ids, context_char_ids, question_ids, question_char_ids, labels, prex='train'):
    np.savez(
                os.path.join(config.data_dir, f"{prex}_features.npz"),
                context_ids=np.array(context_ids),
                context_char_ids=np.array(context_char_ids),
                question_ids=np.array(question_ids),
                question_char_ids=np.array(question_char_ids),
                labels=np.array(labels)
            )


def save_noanswer_features(context_ids, context_char_ids, question_ids, question_char_ids, labels, prex='train'):
    np.savez(
                os.path.join(config.full_data_dir, f"{prex}_features.npz"),
                context_ids=np.array(context_ids),
                context_char_ids=np.array(context_char_ids),
                question_ids=np.array(question_ids),
                question_char_ids=np.array(question_char_ids),
                labels=np.array(labels)
            )


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

    train_df = postprocess_df(train_df, word2idx, idx2word, char2idx, prex_filename='train')
    dev_df = postprocess_df(dev_df, word2idx, idx2word, char2idx, prex_filename='dev')

    save_features(train_df.context_ids, train_df.context_char_ids, train_df.question_ids,
                  train_df.question_char_ids, train_df.label_ids)
    save_features(dev_df.context_ids, dev_df.context_char_ids, dev_df.question_ids,
                  dev_df.question_char_ids, dev_df.label_ids, prex='dev')

    glove_path = config.glove_path
    glove_dict = load_pretrain_embedding(glove_path)
    embedding_matrix = create_embedding_matrix(word2idx, glove_dict)


