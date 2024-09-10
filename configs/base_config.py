import json
from typing import Dict, Any, Set

class BaseConfig:
    required_arguments: Set[str] = set()

    @classmethod
    def check_required(cls, d: Dict[str, Any]) -> None:
        if (d is None or len(d) == 0) and len(cls.required_arguments) > 0:
            raise ValueError("Missing required arguments !")

        for arg in cls.required_arguments:
            if arg not in d:
                raise ValueError("Could not find required argument !")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BaseConfig":
        raise NotImplementedError()

    @classmethod
    def from_json(cls, filepath: str) -> "BaseConfig":
        with open(filepath, "r") as config_file:
            parsed_json = json.load(config_file)
        assert isinstance(parsed_json, dict)
        config = cls.from_dict(parsed_json)
        return config

    def get_parsed_vars(self) -> Dict[str, Any]:
        var = vars(self)
        parsed_var: Dict[str, Any] = {}
        for k, v in sorted(var.items()):
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], BaseConfig):
                parsed_var[k] = [v_i.get_parsed_vars() for v_i in v]
            elif isinstance(v, BaseConfig):
                parsed_var[k] = v.get_parsed_vars()
            else:
                parsed_var[k] = v
        return parsed_var

    def __repr__(self) -> str:
        parsed_var = self.get_parsed_vars()
        return json.dumps(parsed_var, indent=4, ensure_ascii=False)

