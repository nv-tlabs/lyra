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

import contextlib
import json
from typing import Any, Optional

from lyra_2._ext.imaginaire.utils import log
from lyra_2._ext.imaginaire.utils.env_parsers.cred_env_parser import CRED_ENVS, CRED_ENVS_DICT

DEPLOYMENT_ENVS = ["prod", "dev", "stg"]


# context manger to open a file or read from env variable
@contextlib.contextmanager
def open_auth(s3_credential_path: Optional[Any], mode: str):
    if not s3_credential_path:
        log.info(f"No credential file provided {s3_credential_path}.")
        yield None
        return

    name = s3_credential_path.split("/")[-1].split(".")[0]
    if not name:
        raise ValueError(f"Could not parse into env var: {s3_credential_path}")
    cred_env_name = f"PROD_{name.upper()}"

    if CRED_ENVS.APP_ENV in DEPLOYMENT_ENVS and cred_env_name in CRED_ENVS_DICT:
        object_storage_config = get_creds_from_env(cred_env_name)
        log.info(f"using ENV vars for {cred_env_name}")
        yield object_storage_config
    else:
        log.info(f"using credential file: {s3_credential_path}")
        with open(s3_credential_path, mode) as f:
            yield f


def get_creds_from_env(cred_env_name: str) -> dict[str, str]:
    try:
        object_storage_config = CRED_ENVS_DICT[cred_env_name]
    except KeyError:
        raise ValueError(f"Could not find {cred_env_name} in CRED_ENVS")
    empty_args = {key.upper() for key in object_storage_config if object_storage_config[key] == ""}
    if empty_args:
        raise ValueError(f"Some required environment variable(s) were not provided for {cred_env_name}", empty_args)
    return object_storage_config


def json_load_auth(f):
    if CRED_ENVS.APP_ENV in DEPLOYMENT_ENVS:
        return f if f else {}
    else:
        return json.load(f)
