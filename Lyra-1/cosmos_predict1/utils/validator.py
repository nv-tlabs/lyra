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

import ast
import base64
import itertools
import json
import os
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, List, Union


# from https://docs.python.org/3/howto/descriptor.html#validator-class
# For usage of hidden flag see the ModelParams class in apis/utils/model_params.py
class Validator(ABC):
    # set name is called when the validator is created as class variable
    # name is the name of the variable in the owner class, so here we create the name for the backing variable
    def __set_name__(self, owner, name):
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name, self.default)

    def __set__(self, obj, value):
        value = self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value):
        pass

    def json(self):
        pass


class MultipleOf(Validator):
    def __init__(self, default: int, multiple_of: int, type_cast=None, hidden=False, tooltip=None):
        if type(multiple_of) is not int:
            raise ValueError(f"Expected {multiple_of!r} to be an int")
        self.multiple_of = multiple_of
        self.default = default
        self.type_cast = type_cast

        # For usage of hidden flag see the ModelParams class in apis/utils/model_params.py
        # if a parameter is hidden then probe() can't expose the param
        # and the param can't be set anymore
        self.hidden = hidden
        self.tooltip = tooltip

    def validate(self, value):
        if self.type_cast:
            try:
                value = self.type_cast(value)
            except ValueError:
                raise ValueError(f"Expected {value!r} to be castable to {self.type_cast!r}")

        if value % self.multiple_of != 0:
            raise ValueError(f"Expected {value!r} to be a multiple of {self.multiple_of!r}")

        return value

    def get_range_iterator(self):
        return itertools.count(0, self.multiple_of)

    def __repr__(self) -> str:
        return f"MultipleOf({self.private_name=} {self.multiple_of=} {self.hidden=})"

    def json(self):
        return {
            "type": MultipleOf.__name__,
            "default": self.default,
            "multiple_of": self.multiple_of,
            "tooltip": self.tooltip,
        }


class OneOf(Validator):
    def __init__(self, default, options, type_cast=None, hidden=False, tooltip=None):
        self.options = set(options)
        self.default = default
        self.type_cast = type_cast  # Cast the value to this type before checking if it's in options
        self.tooltip = tooltip
        self.hidden = hidden

    def validate(self, value):
        if self.type_cast:
            try:
                value = self.type_cast(value)
            except ValueError:
                raise ValueError(f"Expected {value!r} to be castable to {self.type_cast!r}")

        if value not in self.options:
            raise ValueError(f"Expected {value!r} to be one of {self.options!r}")

        return value

    def get_range_iterator(self):
        return self.options

    def __repr__(self) -> str:
        return f"OneOf({self.private_name=} {self.options=} {self.hidden=})"

    def json(self):
        return {
            "type": OneOf.__name__,
            "default": self.default,
            "values": list(self.options),
            "tooltip": self.tooltip,
        }


class HumanAttributes(Validator):
    def __init__(self, default, hidden=False, tooltip=None):
        self.default = default
        self.hidden = hidden
        self.tooltip = tooltip

    # hard code the options for now
    # we extend this to init parameter as needed
    valid_attributes = {
        "emotion": ["angry", "contemptful", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"],
        "race": ["asian", "indian", "black", "white", "middle eastern", "latino hispanic"],
        "gender": ["male", "female"],
        "age group": [
            "young",
            "teen",
            "adult early twenties",
            "adult late twenties",
            "adult early thirties",
            "adult late thirties",
            "adult middle aged",
            "older adult",
        ],
    }

    def get_range_iterator(self):
        # create a list of all possible combinations
        l1 = self.valid_attributes["emotion"]
        l2 = self.valid_attributes["race"]
        l3 = self.valid_attributes["gender"]
        l4 = self.valid_attributes["age group"]
        all_combinations = list(itertools.product(l1, l2, l3, l4))
        return iter(all_combinations)

    def validate(self, value):
        human_attributes = value.lower()
        if human_attributes not in ["none", "random"]:
            # In this case, we need for custom attribute string

            attr_string = human_attributes
            for attr_key in ["emotion", "race", "gender", "age group"]:
                attr_detected = False
                for attr_label in self.valid_attributes[attr_key]:
                    if attr_string.startswith(attr_label):
                        attr_string = attr_string[len(attr_label) + 1 :]  # noqa: E203
                        attr_detected = True
                        break

                if attr_detected is False:
                    raise ValueError(f"Expected {value!r} to be one of {self.valid_attributes!r}")

        return value

    def __repr__(self) -> str:
        return f"HumanAttributes({self.private_name=} {self.hidden=})"

    def json(self):
        return {
            "type": HumanAttributes.__name__,
            "default": self.default,
            "values": self.valid_attributes,
            "tooltip": self.tooltip,
        }


class Bool(Validator):
    def __init__(self, default, hidden=False, tooltip=None):
        self.default = default
        self.hidden = hidden
        self.tooltip = tooltip

    def validate(self, value):
        if isinstance(value, int):
            value = value != 0
        elif isinstance(value, str):
            value = value.lower()
            if value in ["true", "1"]:
                value = True
            elif value in ["false", "0"]:
                value = False
            else:
                raise ValueError(f"Expected {value!r} to be one of ['True', 'False', '1', '0']")
        elif not isinstance(value, bool):
            raise TypeError(f"Expected {value!r} to be an bool")

        return value

    def get_range_iterator(self):
        return [True, False]

    def __repr__(self) -> str:
        return f"Bool({self.private_name=} {self.default=} {self.hidden=})"

    def json(self):
        return {
            "type": bool.__name__,
            "default": self.default,
            "tooltip": self.tooltip,
        }


class Int(Validator):
    def __init__(self, default, min=None, max=None, step=1, hidden=False, tooltip=None):
        self.min = min
        self.max = max
        self.default = default
        self.step = step
        self.hidden = hidden
        self.tooltip = tooltip

    def validate(self, value):
        if isinstance(value, str):
            value = int(value)
        elif not isinstance(value, int):
            raise TypeError(f"Expected {value!r} to be an int")

        if self.min is not None and value < self.min:
            raise ValueError(f"Expected {value!r} to be at least {self.min!r}")
        if self.max is not None and value > self.max:
            raise ValueError(f"Expected {value!r} to be no more than {self.max!r}")
        return value

    def get_range_iterator(self):
        iter_min = self.min if self.min is not None else self.default
        iter_max = self.max if self.max is not None else self.default
        return itertools.takewhile(lambda x: x <= iter_max, itertools.count(iter_min, self.step))

    def __repr__(self) -> str:
        return f"Int({self.private_name=} {self.default=}, {self.min=}, {self.max=} {self.hidden=})"

    def json(self):
        return {
            "type": int.__name__,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "tooltip": self.tooltip,
        }


class Float(Validator):
    def __init__(self, default=0.0, min=None, max=None, step=0.5, hidden=False, tooltip=None):
        self.min = min
        self.max = max
        self.default = default
        self.step = step
        self.hidden = hidden
        self.tooltip = tooltip

    def validate(self, value):
        if isinstance(value, str) or isinstance(value, int):
            value = float(value)
        elif not isinstance(value, float):
            raise TypeError(f"Expected {value!r} to be float")

        if self.min is not None and value < self.min:
            raise ValueError(f"Expected {value!r} to be at least {self.min!r}")
        if self.max is not None and value > self.max:
            raise ValueError(f"Expected {value!r} to be no more than {self.max!r}")
        return value

    def get_range_iterator(self):
        iter_min = self.min if self.min is not None else self.default
        iter_max = self.max if self.max is not None else self.default
        return itertools.takewhile(lambda x: x <= iter_max, itertools.count(iter_min, self.step))

    def __repr__(self) -> str:
        return f"Float({self.private_name=} {self.default=}, {self.min=}, {self.max=} {self.hidden=})"

    def json(self):
        return {
            "type": float.__name__,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "step": self.step,
            "tooltip": self.tooltip,
        }


class String(Validator):
    def __init__(self, default="", min=None, max=None, predicate=None, hidden=False, tooltip=None):
        self.min = min
        self.max = max
        self.predicate = predicate
        self.default = default
        self.hidden = hidden
        self.tooltip = tooltip

    def validate(self, value):
        if not isinstance(value, str):
            raise TypeError(f"Expected {value!r} to be an str")
        if self.min is not None and len(value) < self.min:
            raise ValueError(f"Expected {value!r} to be no smaller than {self.min!r}")
        if self.max is not None and len(value) > self.max:
            raise ValueError(f"Expected {value!r} to be no bigger than {self.max!r}")
        if self.predicate is not None and not self.predicate(value):
            raise ValueError(f"Expected {self.predicate} to be true for {value!r}")
        return value

    def get_range_iterator(self):
        return iter([self.default])

    def __repr__(self) -> str:
        return f"String({self.private_name=} {self.default=}, {self.min=}, {self.max=} {self.hidden=})"

    def json(self):
        return {
            "type": str.__name__,
            "default": self.default,
            "tooltip": self.tooltip,
        }


class Path(Validator):
    def __init__(self, default="", hidden=False, tooltip=None):
        self.default = default
        self.hidden = hidden
        self.tooltip = tooltip

    def validate(self, value):
        if not isinstance(value, str):
            raise TypeError(f"Expected {value!r} to be an str")
        if not os.path.exists(value):
            raise ValueError(f"Expected {value!r} to be a valid path")

        return value

    def get_range_iterator(self):
        return iter([self.default])

    def __repr__(self) -> str:
        return f"String({self.private_name=} {self.default=}, {self.hidden=})"


class InputImage(Validator):
    def __init__(self, default="", hidden=False, tooltip=None):
        self.default = default
        self.hidden = hidden
        self.tooltip = tooltip

    valid_formats = {
        "JPEG": ["jpeg", "jpg"],
        "JPEG2000": ["jp2"],
        "PNG": ["png"],
        "GIF": ["gif"],
        "BMP": ["bmp"],
    }

    valid_extensions = {vi: k for k, v in valid_formats.items() for vi in v}

    def validate(self, value):
        _, ext = os.path.splitext(value).lower()
        image_format = InputImage.valid_extensions[ext]

        if not isinstance(value, str):
            raise TypeError(f"Expected {value!r} to be an str")
        if not os.path.exists(value):
            raise ValueError(f"Expected {value!r} to be a valid path")
        return value

    def get_range_iterator(self):
        return iter([self.default])

    def __repr__(self) -> str:
        return f"String({self.private_name=} {self.default=} {self.hidden=})"

    def json(self):
        return {
            "type": InputImage.__name__,
            "default": self.default,
            "values": self.valid_formats,
            "tooltip": self.tooltip,
        }


class MeshFormat(Validator):
    """
    Validator class for mesh formats. Valid inputs are either:
    - single valid format such as "glb", "obj"
    - or a list of valid formats such as "[obj, ply, usdz]"
    """

    valid_formats = {"glb", "usdz", "obj", "ply"}

    def __init__(self, default="glb", hidden=False, tooltip=None):
        self.default = default
        self.hidden = hidden
        self.tooltip = tooltip

    def validate(self, value: str) -> Union[str, List[str]]:
        try:
            # Attempt to parse the input as a Python list
            if value.startswith("[") and value.endswith("]"):
                formats = ast.literal_eval(value)
                if not all(fmt in MeshFormat.valid_formats for fmt in formats):
                    raise ValueError(f"Each item must be one of {MeshFormat.valid_formats}")
                return formats
            elif value in MeshFormat.valid_formats:
                return value
            else:
                raise ValueError(f"Expected {value!r} to be one of {MeshFormat.valid_formats} or a list of them")
        except (SyntaxError, ValueError) as e:
            # Handle case where the input is neither a valid single format nor a list of valid formats
            raise ValueError(f"Invalid format specification: {value}. Error: {str(e)}")

    def __repr__(self) -> str:
        return f"MeshFormat(default={self.default}, hidden={self.hidden})"

    def json(self):
        return {
            "type": MeshFormat.__name__,
            "default": self.default,
            "values": self.valid_formats,
            "tooltip": self.tooltip,
        }


class JsonDict(Validator):
    """
    JSON stringified version of a python dict.
    Example: '{"ema_customization_iter.pt": "ema_customization_iter.pt"}'
    """

    def __init__(self, default="", hidden=False):
        self.default = default
        self.hidden = hidden

    def validate(self, value):
        if not value:
            return {}
        try:
            dict = json.loads(value)
            return dict
        except json.JSONDecodeError as e:
            raise ValueError(f"Expected {value!r} to be json  stringified dict. Error: {str(e)}")

    def __repr__(self) -> str:
        return f"Dict({self.default=} {self.hidden=})"


class BytesIOType(Validator):
    """
    Validator class for BytesIO. Valid inputs are either:
    - bytes
    - objects of class BytesIO
    - str which can be successfully  decoded into BytesIO
    """

    def __init__(self, default=None, hidden=False, tooltip=None):
        self.default = default
        self.hidden = hidden
        self.tooltip = tooltip

    def validate(self, value: Any) -> BytesIO:
        if isinstance(value, str):
            try:
                # Decode the Base64 string
                decoded_bytes = base64.b64decode(value)
                # Create a BytesIO stream from the decoded bytes
                return BytesIO(decoded_bytes)
            except (base64.binascii.Error, ValueError) as e:
                raise ValueError(f"Invalid Base64 encoded string: {e}")
        elif isinstance(value, bytes):
            return BytesIO(value)
        elif isinstance(value, BytesIO):
            return value
        else:
            raise TypeError(f"Expected {value!r} to be a Base64 encoded string, bytes, or BytesIO")

    def __repr__(self) -> str:
        return f"BytesIOValidator({self.default=}, {self.hidden=})"

    def json(self):
        return {
            "type": BytesIO.__name__,
            "default": self.default,
            "tooltip": self.tooltip,
        }
