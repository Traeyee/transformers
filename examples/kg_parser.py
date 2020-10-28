#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyiwork@foxmail.com
# Created Time: 01 September 2020 15:33
"""相当复杂的任务设计，很多代码段可以从中copy"""
import os
import sys
import json
import logging
import random
import timeit
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    AlbertModel,
    AlbertPreTrainedModel,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    get_linear_schedule_with_warmup,
    BertTokenizer,
)
from transformers.file_utils import add_start_docstrings_to_callable

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class AlbertForKgParser(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.albert = AlbertModel(config)
        model_dict = config.model_dict
        self.model_dict = model_dict
        self.spo_class_layer = nn.Linear(config.hidden_size,
                                         len(model_dict["spo"]))
        feature_vector1_size = len(model_dict["spo"])  # SPO=3
        feature_vector2_size = feature_vector1_size + len(
            model_dict["entity_type"]) + len(model_dict["special_entity"]) + 1
        self.entity_type_layer = nn.Linear(
            config.hidden_size + feature_vector1_size,
            len(model_dict["entity_type"]))
        self.special_entity_type_layer = nn.Linear(
            config.hidden_size + feature_vector1_size,
            len(model_dict["special_entity"]) + 1,
        )  # +1 for null entity_type
        self.entity_span_start_layer = nn.Linear(config.hidden_size, 1)
        self.entity_span_end_layer = nn.Linear(config.hidden_size, 1)
        self.property_type_class_layer = nn.Linear(
            config.hidden_size + feature_vector2_size,
            len(model_dict["pat_dic_property_type_dict"]))
        self.property_value_span_layer = nn.Linear(config.hidden_size, 2)

        self.sigmoid = nn.Sigmoid()

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.cross_entropy_loss_ignore_0 = nn.CrossEntropyLoss(ignore_index=0)
        self.bce_loss = nn.BCELoss(reduction="mean")

        self.init_weights()

    def encode_text(
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
        return outputs

    def forward_spo(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sequence_output=None,
        pooled_output=None,
    ):
        spo_class_logits = self.spo_class_layer(pooled_output)
        return spo_class_logits

    def forward_entity_type(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sequence_output=None,
        pooled_output=None,
        feature_vector1=None,
    ):
        input_vec = torch.cat([pooled_output, feature_vector1], dim=-1)
        entity_type_logits = self.entity_type_layer(input_vec)
        return entity_type_logits

    def forward_generalized_entity_type(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sequence_output=None,
        pooled_output=None,
        feature_vector1=None,
    ):
        entity_type_logits = self.forward_entity_type(
            pooled_output=pooled_output, feature_vector1=feature_vector1)
        input_vec = torch.cat([pooled_output, feature_vector1], dim=-1)
        special_entity_type_logits = self.special_entity_type_layer(input_vec)
        generalized_entity_type_logits = torch.cat(
            [entity_type_logits, special_entity_type_logits], dim=-1)
        return generalized_entity_type_logits

    def forward_entity_span(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sequence_output=None,
        pooled_output=None,
        feature_vector1=None,
    ):
        entity_span_start_logits = self.entity_span_start_layer(
            sequence_output)
        entity_span_start_logits = self.sigmoid(entity_span_start_logits)
        entity_span_start_logits = entity_span_start_logits.squeeze(-1)
        entity_span_start_logits = entity_span_start_logits * attention_mask

        entity_span_end_logits = self.entity_span_end_layer(sequence_output)
        entity_span_end_logits = self.sigmoid(entity_span_end_logits)
        entity_span_end_logits = entity_span_end_logits.squeeze(-1)
        entity_span_end_logits = entity_span_end_logits * attention_mask
        return entity_span_start_logits, entity_span_end_logits

    def forward_property_type(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sequence_output=None,
        pooled_output=None,
        feature_vector1=None,
        feature_vector2=None,
    ):
        input_vec = torch.cat([pooled_output, feature_vector2], dim=-1)
        property_type_logits = self.property_type_class_layer(input_vec)
        property_type_logits = self.sigmoid(property_type_logits)
        property_type_logits = property_type_logits.squeeze(-1)
        return property_type_logits

    def forward_property_value(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sequence_output=None,
        pooled_output=None,
        feature_vector1=None,
        feature_vector2=None,
    ):
        # 仿bert做法
        property_value_span_logits = self.property_value_span_layer(
            sequence_output)
        property_value_span_logits = property_value_span_logits.permute(
            2, 0, 1)
        property_value_span_start_logits, property_value_span_end_logits = \
            torch.unbind(property_value_span_logits)
        return property_value_span_start_logits, property_value_span_end_logits

    def train_N(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        spo_label=None,
    ):
        outputs = self.encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        spo_class_logits = self.spo_class_layer(pooled_output)
        spo_loss = self.cross_entropy_loss(spo_class_logits, spo_label)

        loss = spo_loss
        loss_dict = {
            "spo_loss": spo_loss,
        }
        return loss, loss_dict

    def train_S(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        feature_vector=None,
        spo_label=None,
        entity_type_label=None,
    ):
        outputs = self.encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        spo_class_logits = self.spo_class_layer(pooled_output)
        spo_loss = self.cross_entropy_loss(spo_class_logits, spo_label)

        input_vec = torch.cat([pooled_output, feature_vector], dim=-1)
        entity_type_logits = self.entity_type_layer(input_vec)

        entity_type_loss = self.cross_entropy_loss(entity_type_logits,
                                                   entity_type_label)

        loss = spo_loss + entity_type_loss
        loss_dict = {
            "spo_loss": spo_loss,
            "entity_type_loss": entity_type_loss,
        }
        return loss, loss_dict

    def train_P(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        feature_vector1=None,
        feature_vector2=None,
        spo_label=None,
        generalized_entity_type_label=None,
        entity_span_start_label=None,
        entity_span_end_label=None,
        property_type_label=None,
    ):
        outputs = self.encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        spo_class_logits = self.spo_class_layer(pooled_output)
        spo_loss = self.cross_entropy_loss(spo_class_logits, spo_label)

        input_vec = torch.cat([pooled_output, feature_vector1], dim=-1)
        entity_type_logits = self.entity_type_layer(input_vec)
        special_entity_type_logits = self.special_entity_type_layer(input_vec)
        generalized_entity_type_logits = torch.cat(
            [entity_type_logits, special_entity_type_logits], dim=-1)
        special_entity_type_loss = self.cross_entropy_loss(
            generalized_entity_type_logits, generalized_entity_type_label)

        entity_span_start_logits = self.entity_span_start_layer(
            sequence_output)
        entity_span_start_logits = self.sigmoid(entity_span_start_logits)
        entity_span_start_logits = entity_span_start_logits.squeeze(-1)
        entity_span_start_logits = entity_span_start_logits * attention_mask
        entity_span_start_loss = self.bce_loss(
            entity_span_start_logits,
            entity_span_start_label.type(torch.float))

        entity_span_end_logits = self.entity_span_end_layer(sequence_output)
        entity_span_end_logits = self.sigmoid(entity_span_end_logits)
        entity_span_end_logits = entity_span_end_logits.squeeze(-1)
        entity_span_end_logits = entity_span_end_logits * attention_mask
        entity_span_end_loss = self.bce_loss(
            entity_span_end_logits, entity_span_end_label.type(torch.float))

        input_vec = torch.cat([pooled_output, feature_vector2], dim=-1)
        property_type_logits = self.property_type_class_layer(input_vec)
        property_type_logits = self.sigmoid(property_type_logits)
        property_type_logits = property_type_logits.squeeze(-1)
        property_type_loss = self.bce_loss(
            property_type_logits, property_type_label.type(torch.float))

        loss = spo_loss + special_entity_type_loss + entity_span_start_loss + entity_span_end_loss + property_type_loss
        loss_dict = {
            "spo_loss": spo_loss,
            "special_entity_type_loss": special_entity_type_loss,
            "entity_span_start_loss": entity_span_start_loss,
            "entity_span_end_loss": entity_span_end_loss,
            "property_type_loss": property_type_loss,
        }
        return loss, loss_dict

    def train_O(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        feature_vector1=None,
        feature_vector2=None,
        feature_vector3=None,
        spo_label=None,
        generalized_entity_type_label=None,
        entity_span_start_label=None,
        entity_span_end_label=None,
        property_type_label=None,
        property_value_span_start_label=None,
        property_value_span_end_label=None,
    ):
        outputs = self.encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        spo_class_logits = self.spo_class_layer(pooled_output)
        spo_loss = self.cross_entropy_loss(spo_class_logits, spo_label)

        input_vec = torch.cat([pooled_output, feature_vector1], dim=-1)
        entity_type_logits = self.entity_type_layer(input_vec)
        special_entity_type_logits = self.special_entity_type_layer(input_vec)
        generalized_entity_type_logits = torch.cat(
            [entity_type_logits, special_entity_type_logits], dim=-1)
        special_entity_type_loss = self.cross_entropy_loss(
            generalized_entity_type_logits, generalized_entity_type_label)

        entity_span_start_logits = self.entity_span_start_layer(
            sequence_output)
        entity_span_start_logits = self.sigmoid(entity_span_start_logits)
        entity_span_start_logits = entity_span_start_logits.squeeze(-1)
        entity_span_start_logits = entity_span_start_logits * attention_mask
        entity_span_start_loss = self.bce_loss(
            entity_span_start_logits,
            entity_span_start_label.type(torch.float))

        entity_span_end_logits = self.entity_span_end_layer(sequence_output)
        entity_span_end_logits = self.sigmoid(entity_span_end_logits)
        entity_span_end_logits = entity_span_end_logits.squeeze(-1)
        entity_span_end_logits = entity_span_end_logits * attention_mask
        entity_span_end_loss = self.bce_loss(
            entity_span_end_logits, entity_span_end_label.type(torch.float))

        input_vec = torch.cat([pooled_output, feature_vector2], dim=-1)
        property_type_logits = self.property_type_class_layer(input_vec)
        property_type_logits = self.sigmoid(property_type_logits)
        property_type_logits = property_type_logits.squeeze(-1)
        property_type_loss = self.bce_loss(
            property_type_logits, property_type_label.type(torch.float))

        # 仿bert做法
        property_value_span_logits = self.property_value_span_layer(
            sequence_output)
        property_value_span_logits = property_value_span_logits.permute(
            2, 0, 1)
        property_value_span_start_logits, property_value_span_end_logits = \
                                            torch.unbind(property_value_span_logits)
        property_value_span_start_loss = self.cross_entropy_loss_ignore_0(
            property_value_span_start_logits, property_value_span_start_label)
        property_value_span_end_loss = self.cross_entropy_loss_ignore_0(
            property_value_span_end_logits, property_value_span_end_label)

        loss = spo_loss + special_entity_type_loss + entity_span_start_loss + entity_span_end_loss + property_type_loss + property_value_span_start_loss + property_value_span_end_loss
        loss_dict = {
            "spo_loss": spo_loss,
            "special_entity_type_loss": special_entity_type_loss,
            "entity_span_start_loss": entity_span_start_loss,
            "entity_span_end_loss": entity_span_end_loss,
            "property_type_loss": property_type_loss,
            "property_value_span_start_loss": property_value_span_start_loss,
            "property_value_span_end_loss": property_value_span_end_loss,
        }
        return loss, loss_dict

    @add_start_docstrings_to_callable("For KgParser")
    def forward(self, ):
        raise NotImplementedError


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

    def convert_to_features(self, tokenizer, max_length=32):
        self.tokens = tokenizer.tokenize(self.text)
        encoded_dict = tokenizer.encode_plus(self.tokens,
                                             truncation=True,
                                             max_length=max_length,
                                             pad_to_max_length=True,
                                             return_tensors="pt",
                                             return_token_type_ids=True,
                                             return_attention_mask=True)
        self.input_ids = encoded_dict["input_ids"]
        self.attention_mask = encoded_dict["attention_mask"]
        self.token_type_ids = encoded_dict["token_type_ids"]


class InputExample:
    def __init__(self, record, tokenizer, max_text_length):
        self.record = record
        sentence = SentenceExample(record["question"])
        sentence.convert_to_features(tokenizer, max_length=max_text_length)
        self.sentence_example = sentence
        self.is_valid = True

        self.spo_vec = None
        self.spo_label = None
        self.feature_vector1 = None
        self.entity_type_label = None
        self.generalized_entity_type_label = None
        self.feature_vector2 = None
        self.entity_span_start_label = None
        self.entity_span_end_label = None
        self.property_type_label = None
        self.property_value_span_start_label = None
        self.property_value_span_end_label = None

    def convert_to_features1(self, spo_index, model_dict):
        spo_vec = [0] * len(model_dict["spo"])
        spo_vec[spo_index] = 1
        self.spo_vec = spo_vec
        self.feature_vector1 = torch.tensor(spo_vec, dtype=torch.float)

    def convert_to_features2(self, generalized_entity_type_idx, model_dict):
        generalized_entity_type_space = len(model_dict["entity_type"]) + len(
            model_dict["special_entity"]) + 1
        generalized_entity_type_vec = [0] * generalized_entity_type_space
        generalized_entity_type_vec[generalized_entity_type_idx] = 1
        self.feature_vector2 = torch.tensor(self.spo_vec +
                                            generalized_entity_type_vec,
                                            dtype=torch.float)

    def convert_to_features(self, model_dict):
        record = self.record
        spo_vec = [0] * len(model_dict["spo"])
        spo_str = record["spo"]
        spo_vec[model_dict["spo"][spo_str]] = 1
        self.spo_vec = spo_vec
        self.spo_label = torch.tensor(model_dict["spo"][spo_str],
                                      dtype=torch.long)

        if "N" == spo_str:
            return
        else:
            self.feature_vector1 = torch.tensor(spo_vec, dtype=torch.float)
            if "S" == spo_str:
                self.entity_type_label = torch.tensor(
                    model_dict["entity_type"][record["entity_type"]],
                    dtype=torch.long)
            elif spo_str in ["P", "O"]:  # P-O shared
                # generalized_entity_type
                if record["generalized_entity_type"] in model_dict[
                        "entity_type"]:
                    generalized_entity_type_idx = model_dict["entity_type"][
                        record["generalized_entity_type"]]
                elif record["generalized_entity_type"] in model_dict[
                        "special_entity"]:
                    generalized_entity_type_idx = len(
                        model_dict["entity_type"]
                    ) + model_dict["special_entity"][
                        record["generalized_entity_type"]]
                elif record["generalized_entity_type"] is None:
                    generalized_entity_type_idx = len(
                        model_dict["entity_type"]) + len(
                            model_dict["special_entity"])
                else:
                    raise Exception(json.dumps(record, ensure_ascii=False))
                self.generalized_entity_type_label = torch.tensor(
                    generalized_entity_type_idx, dtype=torch.long)

                # feature_vector2
                generalized_entity_type_space = len(
                    model_dict["entity_type"]) + len(
                        model_dict["special_entity"]) + 1
                generalized_entity_type_vec = [
                    0
                ] * generalized_entity_type_space
                generalized_entity_type_vec[generalized_entity_type_idx] = 1
                self.feature_vector2 = torch.tensor(
                    spo_vec + generalized_entity_type_vec, dtype=torch.float)

                # entity_span_label
                entity_span_start_label_bitmap = [
                    0
                ] * self.sentence_example.input_ids.shape[-1]
                entity_span_end_label_bitmap = [
                    0
                ] * self.sentence_example.input_ids.shape[-1]
                for entity_info in record["entity_name"]:
                    start_label_idx = 1 + entity_info["unicode_offset"]
                    if start_label_idx < len(entity_span_start_label_bitmap):
                        entity_span_start_label_bitmap[start_label_idx] = 1
                    end_label_idx = entity_info[
                        "unicode_offset"] + entity_info["unicode_length"]
                    if end_label_idx < len(entity_span_end_label_bitmap):
                        entity_span_end_label_bitmap[end_label_idx] = 1
                self.entity_span_start_label = entity_span_start_label_bitmap
                self.entity_span_end_label = entity_span_end_label_bitmap

                # property_type_label
                property_type_bitmap = [0] * len(
                    model_dict["pat_dic_property_type_dict"])
                for property_type in record["property_type"]:
                    property_type_bitmap[model_dict[
                        "pat_dic_property_type_dict"][property_type]] = 1
                self.property_type_label = property_type_bitmap

                # additional for O
                if "O" == spo_str:
                    start_label_idx = 1 + record["property_value"][
                        "unicode_offset"]
                    if start_label_idx >= self.sentence_example.input_ids.shape[
                            -1]:
                        sys.stderr.write(
                            "property_value_span_start_label out of index\t{}\n"
                            .format(json.dumps(record, ensure_ascii=False)))
                        self.is_valid = False
                        return
                    self.property_value_span_start_label = start_label_idx
                    end_label_idx = record["property_value"][
                        "unicode_offset"] + record["property_value"][
                            "unicode_length"]
                    if end_label_idx >= self.sentence_example.input_ids.shape[
                            -1]:
                        sys.stderr.write(
                            "property_value_span_end_label out of index\t{}\n".
                            format(json.dumps(record, ensure_ascii=False)))
                        self.is_valid = False
                        return
                    self.property_value_span_end_label = end_label_idx
            else:
                raise Exception(self.record["spo"])


def get_examples(jsonl_path, model_dict, tokenizer, max_text_length, evaluate):
    examples = {"N": [], "S": [], "P": [], "O": []}
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            input_example = InputExample(record, tokenizer, max_text_length)
            input_example.convert_to_features(model_dict)
            if input_example.is_valid:
                examples[record["spo"]].append(input_example)
    return examples


def convert_examples_to_dataset(examples, example_type, evaluate):
    input_ids = torch.cat(
        [_egz.sentence_example.input_ids for _egz in examples])
    attention_mask = torch.cat(
        [_egz.sentence_example.attention_mask for _egz in examples])
    token_type_ids = torch.cat(
        [_egz.sentence_example.token_type_ids for _egz in examples])
    spo_label = torch.tensor([_egz.spo_label for _egz in examples])
    if "N" == example_type:
        return TensorDataset(
            input_ids,
            attention_mask,
            token_type_ids,
            spo_label,
        )
    elif "S" == example_type:
        feature_vector1 = torch.stack([_egz.feature_vector1 for _egz in examples])
        entity_type_label = torch.tensor(
            [_egz.entity_type_label for _egz in examples])
        return TensorDataset(
            input_ids,
            attention_mask,
            token_type_ids,
            feature_vector1,
            spo_label,
            entity_type_label,
        )
    elif example_type in ["P", "O"]:
        feature_vector1 = torch.stack([_egz.feature_vector1 for _egz in examples])
        generalized_entity_type_label = torch.tensor(
            [_egz.generalized_entity_type_label for _egz in examples])
        feature_vector2 = torch.stack(
            [_egz.feature_vector2 for _egz in examples])
        entity_span_start_label = torch.tensor(
            [_egz.entity_span_start_label for _egz in examples])
        entity_span_end_label = torch.tensor(
            [_egz.entity_span_end_label for _egz in examples])
        property_type_label = torch.tensor(
            [_egz.property_type_label for _egz in examples])
        if "P" == example_type:
            return TensorDataset(
                input_ids,
                attention_mask,
                token_type_ids,
                feature_vector1,
                feature_vector2,
                spo_label,
                generalized_entity_type_label,
                entity_span_start_label,
                entity_span_end_label,
                property_type_label,
            )
        else:
            property_value_span_start_label = torch.tensor(
                [_egz.property_value_span_start_label for _egz in examples])
            property_value_span_end_label = torch.tensor(
                [_egz.property_value_span_end_label for _egz in examples])
            return TensorDataset(
                input_ids,
                attention_mask,
                token_type_ids,
                feature_vector1,
                feature_vector2,
                spo_label,
                generalized_entity_type_label,
                entity_span_start_label,
                entity_span_end_label,
                property_type_label,
                property_value_span_start_label,
                property_value_span_end_label,
            )
    else:
        raise Exception(example_type)


def load_and_cache_examples(args,
                            model_dict,
                            tokenizer,
                            evaluate=False,
                            output_examples=False):
    # Load data features from cache or dataset file
    input_dir = "QUESTION_KG_PARSER"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    cached_features_file = os.path.join(
        args.output_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            list(filter(None, input_dir.split("/"))).pop(),
        ),
    )
    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and args.read_cache:
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        examples = features_and_dataset["examples"]
        dataset_N = features_and_dataset["dataset_N"]
        dataset_S = features_and_dataset["dataset_S"]
        dataset_P = features_and_dataset["dataset_P"]
        dataset_O = features_and_dataset["dataset_O"]
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if evaluate:
            examples = get_examples(args.predict_file,
                                    model_dict,
                                    tokenizer,
                                    args.max_text_length,
                                    evaluate=evaluate)
        else:
            examples = get_examples(args.train_file,
                                    model_dict,
                                    tokenizer,
                                    args.max_text_length,
                                    evaluate=evaluate)
        dataset_N = convert_examples_to_dataset(examples["N"], "N", evaluate)
        dataset_S = convert_examples_to_dataset(examples["S"], "S", evaluate)
        dataset_P = convert_examples_to_dataset(examples["P"], "P", evaluate)
        dataset_O = convert_examples_to_dataset(examples["O"], "O", evaluate)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save(
                {
                    # "features": features,
                    "dataset_N": dataset_N,
                    "dataset_S": dataset_S,
                    "dataset_P": dataset_P,
                    "dataset_O": dataset_O,
                    "examples": examples
                },
                cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    dataset_dict = {
        "N": dataset_N,
        "S": dataset_S,
        "P": dataset_P,
        "O": dataset_O
    }
    if output_examples:
        return dataset_dict, examples
    return dataset_dict


def train(args, train_dataset_dict, model, tokenizer):
    """ Train the model """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        from tensorboardX import SummaryWriter
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    dataloader_dict = {}
    for spo, dataset in train_dataset_dict.items():
        train_sampler = RandomSampler(
            dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        train_dataloader = DataLoader(dataset,
                                      sampler=train_sampler,
                                      batch_size=args.train_batch_size)
        dataloader_dict[spo] = train_dataloader

    num_examples = sum(map(len, list(train_dataset_dict.values())))
    steps_per_epoch = sum(map(len, list(dataloader_dict.values())))
    t_total = steps_per_epoch // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(
            args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
                os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps *
        (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split(
                "/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (steps_per_epoch //
                                             args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (
                steps_per_epoch // args.gradient_accumulation_steps)

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d",
                        global_step)
            logger.info("  Will skip the first %d steps in the first epoch",
                        steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    last_total_loss = {"N": {}, "S": {}, "P": {}, "O": {}}
    current_total_loss = {"N": {}, "S": {}, "P": {}, "O": {}}
    model.zero_grad()
    train_iterator = trange(epochs_trained,
                            int(args.num_train_epochs),
                            desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    # Added here for reproductibility
    set_seed(args)

    for _ in train_iterator:
        with tqdm(total=steps_per_epoch) as pbar:
            step = 0
            dataloader_iter_dict = {
                "N": iter(dataloader_dict["N"]),
                "S": iter(dataloader_dict["S"]),
                "P": iter(dataloader_dict["P"]),
                "O": iter(dataloader_dict["O"]),
            }
            while len(dataloader_iter_dict) > 0:
                sample_keys = []
                sample_weights = []
                for spo in dataloader_iter_dict:
                    sample_keys.append(spo)
                    sample_weights.append(len(dataloader_dict[spo]))
                selected_spo_key = random.choices(sample_keys,
                                                  sample_weights,
                                                  k=1)[0]
                data_loader_iter = dataloader_iter_dict[selected_spo_key]
                try:
                    batch = next(data_loader_iter)
                except StopIteration:
                    dataloader_iter_dict.pop(selected_spo_key)
                    continue
                step += 1

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                model.train()
                batch = tuple(t.to(args.device) for t in batch)
                if "N" == selected_spo_key:
                    input_ids = batch[0]
                    attention_mask = batch[1]
                    token_type_ids = batch[2]
                    spo_label = batch[3]
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids,
                        "spo_label": spo_label,
                    }
                    outputs = model.train_N(**inputs)
                    loss = outputs[0]
                elif "S" == selected_spo_key:
                    input_ids = batch[0]
                    attention_mask = batch[1]
                    token_type_ids = batch[2]
                    feature_vector = batch[3]
                    spo_label = batch[4]
                    entity_type_label = batch[5]
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids,
                        "feature_vector": feature_vector,
                        "spo_label": spo_label,
                        "entity_type_label": entity_type_label,
                    }
                    outputs = model.train_S(**inputs)
                    loss = outputs[0]
                elif "P" == selected_spo_key:
                    input_ids = batch[0]
                    attention_mask = batch[1]
                    token_type_ids = batch[2]
                    feature_vector1 = batch[3]
                    feature_vector2 = batch[4]
                    spo_label = batch[5]
                    generalized_entity_type_label = batch[6]
                    entity_span_start_label = batch[7]
                    entity_span_end_label = batch[8]
                    property_type_label = batch[9]
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "token_type_ids": token_type_ids,
                        "feature_vector1": feature_vector1,
                        "feature_vector2": feature_vector2,
                        "spo_label": spo_label,
                        "generalized_entity_type_label":
                        generalized_entity_type_label,
                        "entity_span_start_label": entity_span_start_label,
                        "entity_span_end_label": entity_span_end_label,
                        "property_type_label": property_type_label,
                    }
                    outputs = model.train_P(**inputs)
                    loss = outputs[0]
                elif "O" == selected_spo_key:
                    input_ids = batch[0]
                    attention_mask = batch[1]
                    token_type_ids = batch[2]
                    feature_vector1 = batch[3]
                    feature_vector2 = batch[4]
                    spo_label = batch[5]
                    generalized_entity_type_label = batch[6]
                    entity_span_start_label = batch[7]
                    entity_span_end_label = batch[8]
                    property_type_label = batch[9]
                    property_value_span_start_label = batch[10]
                    property_value_span_end_label = batch[11]
                    inputs = {
                        "input_ids":
                        input_ids,
                        "attention_mask":
                        attention_mask,
                        "token_type_ids":
                        token_type_ids,
                        "feature_vector1":
                        feature_vector1,
                        "feature_vector2":
                        feature_vector2,
                        "spo_label":
                        spo_label,
                        "generalized_entity_type_label":
                        generalized_entity_type_label,
                        "entity_span_start_label":
                        entity_span_start_label,
                        "entity_span_end_label":
                        entity_span_end_label,
                        "property_type_label":
                        property_type_label,
                        "property_value_span_start_label":
                        property_value_span_start_label,
                        "property_value_span_end_label":
                        property_value_span_end_label,
                    }
                    outputs = model.train_O(**inputs)
                    loss = outputs[0]
                else:
                    raise Exception(selected_spo_key)

                pbar.update(1)

                if args.n_gpu > 1:
                    loss = loss.mean(
                    )  # mean() to average on multi-gpu parallel (not distributed) training
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                loss_dict = outputs[1]
                for loss_key, loss_ten in loss_dict.items():
                    if loss_key not in last_total_loss[selected_spo_key]:
                        last_total_loss[selected_spo_key][loss_key] = 0.0
                    if loss_key not in current_total_loss[selected_spo_key]:
                        current_total_loss[selected_spo_key][loss_key] = 0.0
                    current_total_loss[selected_spo_key][
                        loss_key] += loss_ten.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    # Log metrics
                    if args.local_rank in [
                            -1, 0
                    ] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        tb_writer.add_scalar("lr",
                                             scheduler.get_lr()[0],
                                             global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) /
                                             args.logging_steps, global_step)
                        logging_loss = tr_loss
                        for loss_key, cur_loss in current_total_loss[
                                selected_spo_key].items():
                            tb_writer.add_scalar(
                                selected_spo_key + "_" + loss_key,
                                (cur_loss -
                                 last_total_loss[selected_spo_key][loss_key]) /
                                args.logging_steps, global_step)
                            last_total_loss[selected_spo_key][
                                loss_key] = cur_loss
                        results = evaluate(args, model, tokenizer)
                        for key, value in results["scalar"].items():
                            tb_writer.add_scalar("eval_{}".format(key), value,
                                                 global_step)
                        for spo_key, text_list in results["text"].items():
                            for text_idx, text in enumerate(text_list):
                                tb_writer.add_text(
                                    spo_key + "_" + str(text_idx), text,
                                    global_step)
                    # Save model checkpoint
                    if args.local_rank in [
                            -1, 0
                    ] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        output_dir = os.path.join(
                            args.output_dir,
                            "checkpoint-{}".format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(
                            model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(
                            args, os.path.join(output_dir,
                                               "training_args.bin"))
                        logger.info("Saving model checkpoint to %s",
                                    output_dir)

                        torch.save(optimizer.state_dict(),
                                   os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(),
                                   os.path.join(output_dir, "scheduler.pt"))
                        logger.info(
                            "Saving optimizer and scheduler states to %s",
                            output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    dataset_dict, examples = load_and_cache_examples(args,
                                                     model.model_dict,
                                                     tokenizer,
                                                     evaluate=True,
                                                     output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    dataloader_dict = {}
    for spo, dataset in dataset_dict.items():
        sampler = RandomSampler(
            dataset) if args.local_rank == -1 else DistributedSampler(dataset)
        dataloader = DataLoader(dataset,
                                sampler=sampler,
                                batch_size=args.eval_batch_size)
        dataloader_dict[spo] = dataloader

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    num_examples = sum(map(len, list(dataset_dict.values())))
    steps_per_epoch = sum(map(len, list(dataloader_dict.values())))
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    eval_loss_dict = {}
    with tqdm(total=steps_per_epoch) as pbar:
        dataloader_iter_dict = {
            "N": iter(dataloader_dict["N"]),
            "S": iter(dataloader_dict["S"]),
            "P": iter(dataloader_dict["P"]),
            "O": iter(dataloader_dict["O"]),
        }
        for selected_spo_key in ["N", "S", "P", "O"]:
            data_loader_iter = dataloader_iter_dict[selected_spo_key]
            for batch in data_loader_iter:
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    if "N" == selected_spo_key:
                        input_ids = batch[0]
                        attention_mask = batch[1]
                        token_type_ids = batch[2]
                        spo_label = batch[3]
                        inputs = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "token_type_ids": token_type_ids,
                            "spo_label": spo_label,
                        }
                        outputs = model.train_N(**inputs)
                        loss = outputs[0]
                    elif "S" == selected_spo_key:
                        input_ids = batch[0]
                        attention_mask = batch[1]
                        token_type_ids = batch[2]
                        feature_vector = batch[3]
                        spo_label = batch[4]
                        entity_type_label = batch[5]
                        inputs = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "token_type_ids": token_type_ids,
                            "feature_vector": feature_vector,
                            "spo_label": spo_label,
                            "entity_type_label": entity_type_label,
                        }
                        outputs = model.train_S(**inputs)
                        loss = outputs[0]
                    elif "P" == selected_spo_key:
                        input_ids = batch[0]
                        attention_mask = batch[1]
                        token_type_ids = batch[2]
                        feature_vector1 = batch[3]
                        feature_vector2 = batch[4]
                        spo_label = batch[5]
                        generalized_entity_type_label = batch[6]
                        entity_span_start_label = batch[7]
                        entity_span_end_label = batch[8]
                        property_type_label = batch[9]
                        inputs = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "token_type_ids": token_type_ids,
                            "feature_vector1": feature_vector1,
                            "feature_vector2": feature_vector2,
                            "spo_label": spo_label,
                            "generalized_entity_type_label":
                            generalized_entity_type_label,
                            "entity_span_start_label": entity_span_start_label,
                            "entity_span_end_label": entity_span_end_label,
                            "property_type_label": property_type_label,
                        }
                        outputs = model.train_P(**inputs)
                        loss = outputs[0]
                    elif "O" == selected_spo_key:
                        input_ids = batch[0]
                        attention_mask = batch[1]
                        token_type_ids = batch[2]
                        feature_vector1 = batch[3]
                        feature_vector2 = batch[4]
                        spo_label = batch[5]
                        generalized_entity_type_label = batch[6]
                        entity_span_start_label = batch[7]
                        entity_span_end_label = batch[8]
                        property_type_label = batch[9]
                        property_value_span_start_label = batch[10]
                        property_value_span_end_label = batch[11]
                        inputs = {
                            "input_ids":
                            input_ids,
                            "attention_mask":
                            attention_mask,
                            "token_type_ids":
                            token_type_ids,
                            "feature_vector1":
                            feature_vector1,
                            "feature_vector2":
                            feature_vector2,
                            "spo_label":
                            spo_label,
                            "generalized_entity_type_label":
                            generalized_entity_type_label,
                            "entity_span_start_label":
                            entity_span_start_label,
                            "entity_span_end_label":
                            entity_span_end_label,
                            "property_type_label":
                            property_type_label,
                            "property_value_span_start_label":
                            property_value_span_start_label,
                            "property_value_span_end_label":
                            property_value_span_end_label,
                        }
                        outputs = model.train_O(**inputs)
                        loss = outputs[0]
                    else:
                        raise Exception(selected_spo_key)

                    pbar.update(1)
                    loss_dict = outputs[1]
                    for loss_key, loss_val in loss_dict.items():
                        tb_key = selected_spo_key + "_" + loss_key
                        if tb_key not in eval_loss_dict:
                            eval_loss_dict[tb_key] = []
                        eval_loss_dict[tb_key].append(loss_val.item())

    for tb_key in eval_loss_dict:
        eval_loss_dict[tb_key] = np.mean(eval_loss_dict[tb_key])
    results = {"scalar": {}, "text": {}}
    results["scalar"].update(eval_loss_dict)

    model_dict_index_to_key = {}
    model_dict = model.model_dict
    for dict_type in [
            "spo", "entity_type", "pat_dic_property_type_dict",
            "special_entity"
    ]:
        model_dict_index_to_key[dict_type] = {}
        for key, index in model_dict[dict_type].items():
            model_dict_index_to_key[dict_type][index] = key

    for spo_key in ["N", "S", "P", "O"]:
        tb_text_list = []
        for i, example in enumerate(examples[spo_key][:20]):
            model.eval()
            with torch.no_grad():
                inputs = {
                    "input_ids": example.sentence_example.input_ids,
                    "attention_mask": example.sentence_example.attention_mask,
                    "token_type_ids": example.sentence_example.token_type_ids,
                }
                for param_name, param_ten in inputs.items():
                    inputs[param_name] = param_ten.to(args.device)

                outputs = model.encode_text(**inputs)
                sequence_output = outputs[0]
                pooled_output = outputs[1]

                inputs["sequence_output"] = sequence_output
                inputs["pooled_output"] = pooled_output
                outputs = model.forward_spo(**inputs)
                spo_idx = torch.argmax(outputs[0]).item()
                if model_dict["spo"]["N"] == spo_idx:
                    tb_text = "predict:  \nspo={}".format(
                        model_dict_index_to_key["spo"][spo_idx])
                elif model_dict["spo"]["S"] == spo_idx:
                    example.convert_to_features1(spo_idx, model_dict)
                    inputs[
                        "feature_vector1"] = example.feature_vector1.unsqueeze(
                            0).to(args.device)
                    outputs = model.forward_entity_type(**inputs)
                    entity_type_idx = torch.argmax(outputs[0]).item()

                    tb_text = "predict:  \nspo={}, entity_type={}".format(
                        model_dict_index_to_key["spo"][spo_idx],
                        model_dict_index_to_key["entity_type"]
                        [entity_type_idx])
                elif spo_idx in [
                        model_dict["spo"]["P"], model_dict["spo"]["O"]
                ]:
                    example.convert_to_features1(spo_idx, model_dict)
                    inputs[
                        "feature_vector1"] = example.feature_vector1.unsqueeze(
                            0).to(args.device)
                    # generalized_entity_type
                    outputs = model.forward_entity_type(**inputs)
                    generalized_entity_type_idx = torch.argmax(
                        outputs[0]).item()

                    entity_type = ""
                    entity_name = ""
                    if generalized_entity_type_idx < len(
                            model_dict["entity_type"]):
                        entity_type = model_dict_index_to_key["entity_type"][
                            generalized_entity_type_idx]
                    elif generalized_entity_type_idx - len(
                            model_dict["entity_type"]) < len(
                                model_dict["special_entity"]):
                        entity_name += "[special]" + model_dict_index_to_key[
                            "special_entity"][generalized_entity_type_idx -
                                              len(model_dict["entity_type"])]

                    # no special_entity, predict multi entity
                    property_value = None
                    if len(entity_name) == 0:
                        outputs = model.forward_entity_span(**inputs)
                        entity_span_start_logits, entity_span_end_logits = outputs


                        entity_span_start_indices, entity_span_end_indices = [], []
                        for idx, prob in enumerate(entity_span_start_logits[0].
                                                   cpu().numpy().tolist()):
                            if prob > 0.5:
                                entity_span_start_indices.append(idx)
                        for idx, prob in enumerate(entity_span_end_logits[0].
                                                   cpu().numpy().tolist()):
                            if prob > 0.5:
                                entity_span_end_indices.append(idx)
                        entity_name_list = []
                        for start_idx in entity_span_start_indices:
                            for end_idx in entity_span_end_indices:
                                if 0 < start_idx <= end_idx:
                                    span_text = example.record["question"][
                                        start_idx - 1:end_idx]
                                    entity_name_list.append(span_text)
                                    break
                        entity_name += "{" + ",".join(entity_name_list) + "}"

                    # property_type
                    example.convert_to_features2(generalized_entity_type_idx,
                                                 model_dict)
                    inputs[
                        "feature_vector2"] = example.feature_vector2.unsqueeze(
                            0).to(args.device)
                    outputs = model.forward_property_type(**inputs)

                    property_type_list = []
                    for property_type_idx, property_type_score in enumerate(
                            outputs[0].cpu().numpy().tolist()):
                        if property_type_score > 0.5:
                            property_type_list.append(model_dict_index_to_key[
                                "pat_dic_property_type_dict"]
                                                      [property_type_idx])

                    tb_text = "predict:  \nspo={}, entity_type={}, entity_name={}, property_type={}".format(
                        model_dict_index_to_key["spo"][spo_idx], entity_type,
                        entity_name, ", ".join(property_type_list))
                    # O-type
                    if model_dict["spo"]["O"] == spo_idx:
                        # property_value
                        outputs = model.forward_property_value(**inputs)
                        property_value_span_start_logits, property_value_span_end_logits = outputs
                        start_idx = torch.argmax(
                            property_value_span_start_logits[0]).item()
                        end_idx = torch.argmax(
                            property_value_span_end_logits[0]).item()
                        property_value = ""
                        if 0 < start_idx <= end_idx:
                            property_value = example.record["question"][
                                start_idx - 1:end_idx]
                        tb_text += "property_value=" + property_value
                tb_text += "  \n"
                tb_text += "label:  \nspo={}, entity_type={}, generalized_entity_type={}, entity_name={}, property_type={}, property_value={}".format(
                    example.record["spo"],
                    example.record.get("entity_type", ""),
                    example.record.get("generalized_entity_type", ""),
                    ",".join([
                        _en["term"]
                        for _en in example.record.get("entity_name", [])
                    ]),
                    ",".join(example.record.get("property_type", [])),
                    example.record.get("property_value", {}).get("term", ""),
                )

                tb_text += "  \n"
                tb_text += "pat_info:" + example.record.get("pat_info", "")
                tb_text += "  \n"
                tb_text += "question:" + example.record["question"]
                tb_text_list.append(tb_text)

        results["text"][spo_key] = tb_text_list

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path",
                        default=None,
                        type=str,
                        required=True,
                        help="Path to pre-trained model")
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help=
        "The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help=
        "The input training file. If a data dir is specified, will look for the file there"
        +
        "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help=
        "The input evaluation file. If a data dir is specified, will look for the file there"
        +
        "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help=
        "Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_text_length",
        default=64,
        type=int,
        help=
        "The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train",
                        action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--per_gpu_train_batch_size",
                        default=8,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=8,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action="store_true",
                        help="Whether not to use CUDA when available")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--read_cache",
                        action="store_true",
                        help="Read the cached training and evaluation sets")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps",
                        type=int,
                        default=500,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help=
        "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    # Set seed
    set_seed(args)

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=None,
    )
    model_dict = json.load(open("./model_dict.json", "r"))
    config.model_dict = model_dict
    model = AlbertForKgParser.from_pretrained(
        args.model_name_or_path,
        from_tf=False,
        config=config,
        cache_dir=None,
    )
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset_dict = load_and_cache_examples(args,
                                                     model_dict,
                                                     tokenizer,
                                                     evaluate=False,
                                                     output_examples=False)
        global_step, tr_loss = train(args, train_dataset_dict, model,
                                     tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step,
                    tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1
                          or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AlbertForKgParser.from_pretrained(
            args.output_dir)  # , force_download=True)
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model.to(args.device)

        # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info(
                "Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                import glob
                checkpoints = list(
                    os.path.dirname(c) for c in sorted(
                        glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME,
                                  recursive=True)))
                logging.getLogger("transformers.modeling_utils").setLevel(
                    logging.WARN)  # Reduce model loading logs
        else:
            logger.info("Loading checkpoint %s for evaluation",
                        args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split(
                "-")[-1] if len(checkpoints) > 1 else ""
            model = AlbertForKgParser.from_pretrained(
                checkpoint)  # , force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict(
                (k + ("_{}".format(global_step) if global_step else ""), v)
                for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    return results


if __name__ == '__main__':
    main()
