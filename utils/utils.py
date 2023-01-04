from typing import Tuple, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, json, os, random, re
from mpl_toolkits.axes_grid1 import ImageGrid
import shutil, wandb, torch, string
from collections import Counter


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = json.load(f)
    return content


def print_samples(data: dict) -> Tuple[List[Any], List[Any], List[List[Any]]]:
    data = data['data']
    context_lst = []
    ans_lst = []
    question_lst = []
    for element in data[:1]:
        for para in element['paragraphs'][:1]:
            context = para['context']
            for qa_pair in para['qas']:
                id = qa_pair['id']
                question = qa_pair['question']
                ans = qa_pair['answers']

                an_lst = []
                for an in ans:
                    answer = an['text']
                    an_lst.append(answer)
                    print('context is: ', '\n',  context)
                    print('question is: ', '\n', question)
                    print('answer is: ', '\n', answer)

                context_lst.append(context)
                question_lst.append(question)
                if an_lst is None:
                    an_lst.append([' ', ' ', ' ', ' '])
                else:
                    ans_lst.append(an_lst)

    return context_lst, question_lst, ans_lst


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def normalize_answer(s):
    """
    Performs a series of cleaning steps on the ground truth and
    predicted answer.
    """

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    '''
    Returns maximum value of metrics for predicition by model against
    multiple ground truths.

    :param func metric_fn: can be 'exact_match_score' or 'f1_score'
    :param str prediction: predicted answer span by the model
    :param list ground_truths: list of ground truths against which
                               metrics are calculated. Maximum values of
                               metrics are chosen.


    '''
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)

    return max(scores_for_ground_truths)


def f1_score(prediction, ground_truth):
    '''
    Returns f1 score of two strings.
    '''
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    '''
    Returns exact_match_score of two strings.
    '''
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def epoch_time(start_time, end_time):
    '''
    Helper function to record epoch time.
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs