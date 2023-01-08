from datasets import load_dataset, load_metric
import transformers, pickle
import config as cfg
from transformers import BertModel, BertTokenizer, BertConfig, BertPreTrainedModel, \
AutoConfig, AutoTokenizer
import numpy as np

config = AutoConfig.from_pretrained(cfg.config_path)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_checkpoint)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)


def prepare_train_features(examples):
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if cfg.pad_on_right else "context"],
        examples["context" if cfg.pad_on_right else "question"],
        # Note that we never want to truncate the question, only the context, else the only_second truncation picked.
        truncation="only_second" if cfg.pad_on_right else "only_first",
        max_length=cfg.max_length,
        stride=cfg.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    tokenized_examples["context_indices"] = []
    tokenized_examples["ques_indices"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # tokenizer.cls_token_id = 101 sep_token_id = 102
        # tokenizer.cls_token_id = 101 sep_token_id = 102

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        sequence_arr = np.array(sequence_ids, dtype=np.float32)
        context_idx = np.where(sequence_arr == 1)[0].tolist()
        sep_index = input_ids.index(tokenizer.sep_token_id)
        question_idx = list(range(1, sep_index))
        assert context_idx[0] == (sep_index + 1)
        tokenized_examples["context_indices"].append(context_idx)
        tokenized_examples["ques_indices"].append(question_idx)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if cfg.pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if cfg.pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


if __name__ == '__main__':
    # examples = {}
    # question =
    # context =
    # examples.update{'question': question }
    # examples.update{'context': context}
    #

    # examples = datasets['train'][0: 5]
    # context = examples['context'][1]
    # context = context + ' ' + context + ' ' + context + ' ' + context + ' ' + context + ' ' + context + ' ' + context + ' ' + context
    # examples['context'][1] = context
    # print(examples['context'][1].split())
    # print(len(examples['context'][1].split())
    #
    datasets = load_dataset("squad_v2")
    print("datasets['train'] length: ", len(datasets['train']))

    tokenized_datasets = datasets.map(prepare_train_features, batched=True,
                                      remove_columns=datasets["train"].column_names)
    print("tokenized_datasets['train'] length: ", len(tokenized_datasets['train']))

    pickle.dump(tokenized_datasets, open('../data/Bert_Result/tokenized_datasets', 'wb'))

    datasets = pickle.load(open('../data/Bert_Result/tokenized_datasets', 'rb'))
    print(111)



