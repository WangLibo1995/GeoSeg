import pydoc
import sys
from importlib import import_module
from pathlib import Path
from typing import Union

from addict import Dict


class ConfigDict(Dict):
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        else:
            return value
        raise ex


def py2dict(file_path: Union[str, Path]) -> dict:
    """Convert python file to dictionary.
    The main use - config parser.
    file:
    ```
    a = 1
    b = 3
    c = range(10)
    ```
    will be converted to
    {'a':1,
     'b':3,
     'c': range(10)
    }
    Args:
        file_path: path to the original python file.
    Returns: {key: value}, where key - all variables defined in the file and value is their value.
    """
    file_path = Path(file_path).absolute()

    if file_path.suffix != ".py":
        raise TypeError(f"Only Py file can be parsed, but got {file_path.name} instead.")

    if not file_path.exists():
        raise FileExistsError(f"There is no file at the path {file_path}")

    module_name = file_path.stem

    if "." in module_name:
        raise ValueError("Dots are not allowed in config file path.")

    config_dir = str(file_path.parent)

    sys.path.insert(0, config_dir)

    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = {name: value for name, value in mod.__dict__.items() if not name.startswith("__")}

    return cfg_dict


def py2cfg(file_path: Union[str, Path]) -> ConfigDict:
    cfg_dict = py2dict(file_path)

    return ConfigDict(cfg_dict)


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034

    return pydoc.locate(object_type)(**kwargs)