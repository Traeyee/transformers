#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyi@zuoshouyisheng.com
# Created Time: 16 May 2020 19:17
import logging
import torch
from torch import nn
from transformers import (
    AlbertModel,
    AlbertPreTrainedModel,
)
from transformers.file_utils import add_start_docstrings_to_callable

logger = logging.getLogger(__name__)


class AlbertForSiameseSimilaritySimple(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.albert = AlbertModel(config)
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(config.hidden_size, config.mlp_size)
        self.layer2 = nn.Linear(config.mlp_size, config.mlp_size)
        self.layer3 = nn.Linear(config.mlp_size, config.hidden_size)

        self.init_weights()

    def vector(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]
        layer1_output = self.relu(self.layer1(pooled_output))
        layer2_output = self.relu(self.layer2(layer1_output))
        layer3_output = self.relu(self.layer3(layer2_output))
        return layer3_output

    @add_start_docstrings_to_callable("For siamese similarity")
    def forward(
        self,
        input_ids1=None,
        attention_mask1=None,
        token_type_ids1=None,
        position_ids1=None,
        head_mask1=None,
        inputs_embeds1=None,
        input_ids2=None,
        attention_mask2=None,
        token_type_ids2=None,
        position_ids2=None,
        head_mask2=None,
        inputs_embeds2=None,
        label=None,
    ):
        vec1 = self.vector(
            input_ids=input_ids1,
            attention_mask=attention_mask1,
            token_type_ids=token_type_ids1,
            position_ids=position_ids1,
            head_mask=head_mask1,
            inputs_embeds=inputs_embeds1,
        )
        vec2 = self.vector(
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            token_type_ids=token_type_ids2,
            position_ids=position_ids2,
            head_mask=head_mask2,
            inputs_embeds=inputs_embeds2,
        )

        cos = nn.CosineSimilarity()
        similarity = cos(vec1, vec2)
        if label is not None:
            loss_func = nn.MSELoss()
            loss = loss_func(similarity, label)
            return (loss, similarity)
        return similarity


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def clean_text(text):
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True

    # Split on whitespace so that different tokens may be attributed to their original position.
    for c in whitespace_tokenize(" ".join(text)):  # here is the difference
        if _is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)
    cleaned_text = "".join(doc_tokens)
    return cleaned_text


class SentenceExample:
    def __init__(self, text):
        self.text = clean_text(text)
        self.tokens = None
        self.input_ids = None
        self.attention_mask = None
        self.token_type_ids = None

    def convert_to_features(self, tokenizer):
        self.tokens = tokenizer.tokenize(self.text)
        encoded_dict = tokenizer.encode_plus(self.tokens,
                                             truncation=True,
                                             max_length=32,
                                             pad_to_max_length=True,
                                             return_tensors="pt",
                                             return_token_type_ids=True,
                                             return_attention_mask=True)
        self.input_ids = encoded_dict["input_ids"]
        self.attention_mask = encoded_dict["attention_mask"]
        self.token_type_ids = encoded_dict["token_type_ids"]


class SentencePairExample:
    def __init__(self, sentence1, sentence2, tokenizer, label=None):
        self.sentence1 = SentenceExample(sentence1)
        self.sentence1.convert_to_features(tokenizer)
        self.sentence2 = SentenceExample(sentence2)
        self.sentence2.convert_to_features(tokenizer)
        self.label = None
        if label is not None:
            self.label = float(label)


def mpredict(model, examples):
    model.eval()
    with torch.no_grad():
        input_ids1 = torch.cat([_eg.sentence1.input_ids for _eg in examples])
        attention_mask1 = torch.cat(
            [_eg.sentence1.attention_mask for _eg in examples])
        token_type_ids1 = torch.cat(
            [_eg.sentence1.token_type_ids for _eg in examples])
        input_ids2 = torch.cat([_eg.sentence2.input_ids for _eg in examples])
        attention_mask2 = torch.cat(
            [_eg.sentence2.attention_mask for _eg in examples])
        token_type_ids2 = torch.cat(
            [_eg.sentence2.token_type_ids for _eg in examples])
        inputs = {
            "input_ids1": input_ids1,
            "attention_mask1": attention_mask1,
            "token_type_ids1": token_type_ids1,
            "input_ids2": input_ids2,
            "attention_mask2": attention_mask2,
            "token_type_ids2": token_type_ids2,
        }
        outputs = model(**inputs)
        return outputs


def predict(model, example):
    return mpredict(model, [example])[0]
