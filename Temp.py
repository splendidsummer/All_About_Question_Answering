
# def exact_sy_features(sent):
#     doc = nlp(sent)
#     tokens = [[token.text, token.pos_, token.tag_, token.lemma_, str(token.is_stop), token.dep_
#                # token.ent_type_, token.dep_
#                ] for token in doc if '\n' not in token.text and '\r' not in token.text]
#     # tokens = [[token.text, str(token.is_stop)] for token in doc if '\n' not in token.text and '\r' not in token.text]
#     return tokens


# def build_vocab_nltk(data_dirs):
#     vocab = {}
#     for data_dir in data_dirs:
#         for f in os.listdir(data_dir):
#             tree = parse(data_dir + "/" + f)
#             sentences = tree.getElementsByTagName("sentence")
#             for s in sentences:
#                 stext = s.attributes["text"].value  # get sentence text
#
#                 for t in nltk.tokenize.word_tokenize(stext):
#                     if t.lower() not in vocab:
#                         vocab[t.lower()] = len(vocab)
#                     else:
#                         continue
#     return vocab

