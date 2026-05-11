from pydantic import BaseModel
from typing import Dict, Union


class Parameter(BaseModel):
    type: Union[str, int]


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Parameter]
    returns: Parameter


class FunctionCall(BaseModel):
    prompt: str
    # name: str
    # parameters: Dict[str, Union[str, float, int]]


class output_format(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Union[str, float, int]]
