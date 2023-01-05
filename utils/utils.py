from typing import Tuple, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, json, os, random, re
from mpl_toolkits.axes_grid1 import ImageGrid
import shutil, wandb, torch, string
from collections import Counter
import collections
from datasets import load_dataset, load_metric


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


def setup_seed(seed=3407):
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


def evaluate_bert_metric(predictions):
    """

    :param predictions:
    :return:
        {'exact': 71.58258232965552,
     'f1': 75.0429473498408,
     'total': 11873,
     'HasAns_exact': 71.72739541160594,
     'HasAns_f1': 78.65804890092097,
     'HasAns_total': 5928,
     'NoAns_exact': 71.43818334735072,
     'NoAns_f1': 71.43818334735072,
     'NoAns_total': 5945,
     'best_exact': 71.58258232965552,
     'best_exact_thresh': 0.0,
     'best_f1': 75.04294734984086,
     'best_f1_thresh': 0.0}

    """

    datasets = load_dataset("squad_v2")
    metric = load_metric("squad_v2")
    predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]
    metric.compute(predictions=predictions, references=references)



def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
    return qid_to_has_ans


def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    # getting splited word list of golden answer and predicted answer
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    # list_01 = [1,9,9,5,0,8,0,9]   # print(Counter(list_01))  #Counter({9: 3, 0: 2, 1: 1, 5: 1, 8: 1})
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    # There are 3 cases below
    # First case is about if the golden or predicted answer length is 0,
    # in this case it can not be used as denominator
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(dataset, preds):
    exact_scores = {}
    f1_scores = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid = qa['id']
                gold_answers = [a['text'] for a in qa['answers']
                                if normalize_answer(a['text'])]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = ['']
                if qid not in preds:
                    print('Missing prediction for %s' % qid)
                    continue
                a_pred = preds[qid]
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
    return exact_scores, f1_scores


# scores == exact_raw,  na_probs = {k: 0.0 for k in preds},
def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('total', total),
        ])


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]


def evaluate(predictions):
    '''
    Gets a dictionary of predictions with question_id as key
    and prediction as value. The validation dataset has multiple
    answers for a single question. Hence we compare our prediction
    with all the answers and choose the one that gives us
    the maximum metric (em or f1).
    This method first parses the JSON file, gets all the answers
    for a given id and then passes the list of answers and the
    predictions to calculate em, f1.


    :param dict predictions
    Returns
    : exact_match: 1 if the prediction and ground truth
      match exactly, 0 otherwise.
    : f1_score:
    '''

    with open('./data/dev-v2.0.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    data = data['data']   # dataset is a list object
    f1 = exact_match = total = 0
    for article in data:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    continue

                ground_truths = list(map(lambda x: x['text'].lower(), qa['answers']))

                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return exact_match, f1


#
# def find_n_best(output, n_best_size):
#     start_logits = output.start_logits[0].cpu().numpy()
#     end_logits = output.end_logits[0].cpu().numpy()
#     offset_mapping = validation_features[0]["offset_mapping"]
#     # The first feature comes from the first example. For the more general case, we will need to be match the example_id to
#     # an example index
#     context = datasets["validation"][0]["context"]
#
#     # Gather the indices the best start/end logits:
#     start_indexes = np.argsort(start_logits)[-1: -n_best_size - 1: -1].tolist()
#     end_indexes = np.argsort(end_logits)[-1: -n_best_size - 1: -1].tolist()
#     valid_answers = []
#     for start_index in start_indexes:
#         for end_index in end_indexes:
#             # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
#             # to part of the input_ids that are not in the context.
#             if (
#                     start_index >= len(offset_mapping)
#                     or end_index >= len(offset_mapping)
#                     or offset_mapping[start_index] is None
#                     or offset_mapping[end_index] is None
#             ):
#                 continue
#             # Don't consider answers with a length that is either < 0 or > max_answer_length.
#             if end_index < start_index or end_index - start_index + 1 > max_answer_length:
#                 continue
#             if start_index <= end_index:  # We need to refine that test to check the answer is inside the context
#                 start_char = offset_mapping[start_index][0]
#                 end_char = offset_mapping[end_index][1]
#                 valid_answers.append(
#                     {
#                         "score": start_logits[start_index] + end_logits[end_index],
#                         "text": context[start_char: end_char]
#                     }
#                 )
#
#     valid_answers = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[:n_best_size]
#     return valid_answers
#


