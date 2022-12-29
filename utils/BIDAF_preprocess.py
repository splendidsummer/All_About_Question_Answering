import json, pickle, re, os, string, typing, gc
import pandas as pd
import config
import spacy
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



    return None


def build_char_vocab(text):
    pass




if __name__ == '__main__':

    train_path = config.train_file
    dev_path = config.dev_file

    train_df = parse_df(train_path)
    dev_df = parse_df(dev_path)












