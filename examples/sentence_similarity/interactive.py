#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: cuiyi@zuoshouyisheng.com
# Created Time: 17 June 2020 14:18
from transformers import (
    AutoConfig,
    BertTokenizer,
)
from sentence_similarity_utilization import (AlbertForSiameseSimilaritySimple,
                                             SentencePairExample, predict)

import sys
model_name_or_path = sys.argv[1]
config = AutoConfig.from_pretrained(model_name_or_path, )
model = AlbertForSiameseSimilaritySimple.from_pretrained(
    model_name_or_path,
    config=config,
)
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
while True:
    sentence1 = input("sentence1:")
    sentence2 = input("sentence2:")
    example = SentencePairExample(sentence1, sentence2, tokenizer)
    print(predict(model, example))
