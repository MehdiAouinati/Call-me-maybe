import argparse
from .loader import Parse
import numpy as np
from pydantic import ValidationError


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--functions_definition", default="data/input/functions_definition.json")
    parse.add_argument("--input", default="data/input/function_calling_tests.json")
    parse.add_argument("--output")
    args = parse.parse_args()
    load = Parse()
    try:
        prompts_list = load.load_prompts(args.input)
        funcs_list = load.load_functions(args.functions_definition)
        func = [item.model_dump() for item in funcs_list]
        promts = [item.model_dump() for item in prompts_list]
    except ValidationError



