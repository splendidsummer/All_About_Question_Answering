import json, pickle, re, os, string, typing, gc
import pandas as pd
import config
import spacy
from collections import Counter
nlp = spacy.load('en_core_web_sm')


def load_json(path):
    with open(path, 'r') as f:
        content = json.load(f)

    s
    return content


def parse_data(dictionary):
    df = None
    return df




