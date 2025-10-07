# From SRI-lab
from typing import TYPE_CHECKING, Any, Dict, List

import yaml  # type: ignore
from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from config import Config


def load_yaml(cfg_path: str | List[str]) -> Dict:
    if isinstance(cfg_path, list):
        raise ValueError(f"Expected str, got list {cfg_path=} - cfg_path_exp")

    with open(cfg_path, "r") as stream:
        try:
            yaml_obj = yaml.safe_load(stream)
            return yaml_obj
        except (yaml.YAMLError, ValidationError) as exc:
            print(exc)
            raise exc


class PydanticBaseModelWithOptionalDefaultsPath(BaseModel):
    _root_config: "Config"

    def __init__(self, **kwargs: Any) -> None:
        if "cfg_path" in kwargs:
            cfg = load_yaml(kwargs["cfg_path"])
            del kwargs["cfg_path"]
            cfg |= kwargs
        else:
            cfg = kwargs
        super().__init__(**cfg)

    def __str__(self) -> str:
        # For all fields, if the field is not a BaseModel then print it
        ret_str = ""

        keys, vals = zip(*self.__dict__.items())
        # sort by keys
        svals = zip(*sorted(zip(keys, vals)))
        sval_list = list(zip(*[list(x) for x in svals]))

        for key, field in sval_list:
            ret_str += f"{key}=[{str(getattr(self, key))}],"
        if len(ret_str) > 0:
            ret_str = ret_str[:-1]
        return ret_str
