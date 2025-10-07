from configuration.utils import PydanticBaseModelWithOptionalDefaultsPath as PBMwODP
from configuration import Config

import logging
from copy import deepcopy
from typing import Any, Dict, List, Tuple, no_type_check

import yaml  # type: ignore
from pydantic import ValidationError


def index_dict(d: Dict[str, Any], key: Tuple[str]) -> Any:
    """Given a dict and a key that indexes into a leaf , return the value at that leaf.

    Args:
        d (Dict): _description_
        key (Tuple[str]): _description_

    Returns:
        Any: _description_
    """
    if len(key) == 1:
        return d[key[0]]
    else:
        return index_dict(d[key[0]], key[1:])


def set_dict(d: Dict[str, Any], key: Tuple[str], value: Any) -> None:
    """Given a dict and a key that indexes into a leaf , set the value at that leaf.

    Args:
        d (Dict): _description_
        key (Tuple[str]): _description_
        value (Any): _description_
    """
    if len(key) == 1:
        d[key[0]] = value
    else:
        set_dict(d[key[0]], key[1:], value)


def expand_dict_over_key(d: Dict[str, Any], key: Tuple[str]) -> List[Dict]:
    """Given a dict and a key that indexes into a leaf , return a list of dicts
    where each dict has the same keys as the original dict, but the values are spread across the
    different dicts.

    Args:
        d (Dict): _description_
        key (Tuple[str]): _description_

    Returns:
        List[Dict]: _description_
    """

    values_to_expand = index_dict(d, key)
    if not isinstance(values_to_expand, list):
        raise ValueError(f"Expected list at key {key}, got {values_to_expand}")

    # Deep copy the dict
    new_dicts = [deepcopy(d) for _ in range(len(values_to_expand))]

    for i, value in enumerate(values_to_expand):
        set_dict(new_dicts[i], key, value)

    return new_dicts


def expand_dicts_over_key(dicts: List[Dict[str, Any]], key: Tuple[str]) -> List[Dict]:
    """Given a list of dicts and a key that indexes into a leaf for all dicts, return a list of dicts
    where each dict has the same keys as the original list of dicts, but the values are spread across the
    different dicts.

    Args:
        dicts (List[Dict]): _description_
        key (Tuple[str]): _description_

    Returns:
        List[Dict]: _description_
    """

    resulting_dicts = []

    for dict_ in dicts:
        if not isinstance(dict_, dict):
            raise ValueError(f"Expected dict, got {dict_}")
        resulting_dicts.extend(expand_dict_over_key(dict_, key))

    return resulting_dicts


@no_type_check
def get_pydantic_models_from_dict(cfg: Dict[str, Any]) -> List[Config]:
    try:
        model = Config(**cfg)
        return [model]
    except ValidationError as e:
        ers = e.errors()
        # Extract all keys that need to be expanded
        # Get all subconfig
        subs = set([err["loc"][1] for err in ers])
        keys_to_expand = []
        find_correct_sub = [
            {
                "sub": sub,
                "real": all(
                    [
                        err["type"] not in ["missing", "extra_forbidden"]
                        for err in ers
                        if err["loc"][1] == sub
                    ]
                ),
            }
            for sub in subs
        ]
        valid_configs = [sub["sub"] for sub in find_correct_sub if sub["real"]]
        if len(valid_configs) != 1:  # There should be only a single valid config.
            raise ValidationError

        for err in ers:
            # Modify this to make it work for Multiconfig
            if err["loc"][1] == valid_configs[0]:
                err["loc"] = list(err["loc"])
                if (
                    err["type"] == "value_error" and "cfg_path_exp" in err["msg"]
                ):  # Custom handling for cfg_path expansion
                    # Extract key from error message
                    keys_to_expand.append(tuple(err["loc"]) + ("cfg_path",))
                elif isinstance(err["input"], list):
                    keys_to_expand.append(tuple(err["loc"]))

        if len(keys_to_expand) == 0:
            raise e

    cfg = [cfg]

    for key in keys_to_expand:
        cfg = expand_dicts_over_key(cfg, key)

    ret_models: List[Config] = []
    for cfg_ in cfg:
        ret_models.extend(get_pydantic_models_from_dict(cfg_))

    def recursive_set_root_config(model: Config, root_config: Config) -> None:
        model._root_config = root_config
        for key, val in model.__dict__.items():
            if isinstance(val, PBMwODP):
                recursive_set_root_config(val, root_config)

    for model in ret_models:
        recursive_set_root_config(model, model)

    return ret_models


def get_pydantic_models_from_path(path: str) -> List[Config]:
    # Load yaml as dict
    cfg = yaml.safe_load(open(path, "r"))

    return get_pydantic_models_from_dict(cfg)
