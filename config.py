from enum import StrEnum
from pydantic import BaseModel, PositiveInt
import yaml


class OutputDensity(StrEnum):
    NORMAL = "normal"
    DENSE = "dense"
    WIDE = "wide"


class Config(BaseModel):
    stream_prefix: str
    max_decimals: PositiveInt = 4
    comments: bool = True
    output_density: OutputDensity = OutputDensity.NORMAL
    ignore_streams: list[str] = []
    ignore_chemicals: list[str] = []
    ees_aliases: dict[str, str]
    short_aliases: dict[str, str] = {}
    only_prefixed_streams: bool = False


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as file:
        data = yaml.safe_load(file)
    return Config(**data)
