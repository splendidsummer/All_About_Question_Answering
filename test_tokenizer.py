import torch
import json, pickle, re, os, string, typing, gc
import pandas as pd
import numpy as np
import config
import nltk, spacy
from collections import Counter
from utils.utils import *
nlp = spacy.load('en_core_web_sm')

chars = """ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]"""
char_lst = [w for w in chars]
chars = ' '.join(char_lst)
tokens = nlp(chars)

data = load_json(config.dev_file)
context_lst, question_lst, ans_lst = print_samples(data)

for context, question, an in zip(context_lst, question_lst, ans_lst):
    context = [w.text for w in nlp(context, disable=['parser', 'tagger', 'ner'])]
    question = [w.text for w in nlp(question, disable=['parser', 'tagger', 'ner'])]
    answers = []
    for ans in ans_lst:
        answers.append([w.text for w in nlp(an, disable=['parse', 'tagger', 'ner'])] for an in ans if an is not None)

    print('context is: ', ' '. join(context))
    print('question is: ', ' '.join(question))
    for answer in answers:
        print('answer is: ', ' '.join(answer))



# answer_tokens = [w.text for w in nlp(row['answer'], disable=['parser', 'tagger', 'ner'])]
# Model estimates of probability of no answer.

