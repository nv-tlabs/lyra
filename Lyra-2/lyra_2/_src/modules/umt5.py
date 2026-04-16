# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import html
import string

import ftfy
import regex as re
from transformers import AutoTokenizer


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def canonicalize(text, keep_punctuation_exact_string=None):
    text = text.replace("_", " ")
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans("", "", string.punctuation))
            for part in text.split(keep_punctuation_exact_string)
        )
    else:
        text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class HuggingfaceTokenizer:
    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        assert clean in (None, "whitespace", "lower", "canonicalize")
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop("return_mask", False)

        # arguments
        _kwargs = {"return_tensors": "pt"}
        if self.seq_len is not None:
            _kwargs.update({"padding": "max_length", "truncation": True, "max_length": self.seq_len})
        _kwargs.update(**kwargs)

        # tokenization
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(u) for u in sequence]
        ids = self.tokenizer(sequence, **_kwargs)

        # output
        if return_mask:
            return ids.input_ids, ids.attention_mask
        else:
            return ids.input_ids

    def _clean(self, text):
        if self.clean == "whitespace":
            text = whitespace_clean(basic_clean(text))
        elif self.clean == "lower":
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == "canonicalize":
            text = canonicalize(basic_clean(text))
        return text


