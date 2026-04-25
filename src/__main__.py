import argparse
from .loader import Parse
import numpy as np
from pydantic import ValidationError
from .buildPrompt import BuildPrompt
from .decoder import Prediction
from llm_sdk import Small_LLM_Model
import torch


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
        funcs = [item.model_dump() for item in funcs_list]
        prompts = [item.model_dump() for item in prompts_list]
    except ValidationError as e:
        print(e)
        exit(1)
    
    model = Small_LLM_Model()

    fun = []
    fun.append(["fn_add_numbers"])
    fun.append(["fn_greet"])
    fun.append(["fn_reverse_string"])
    fun.append(["fn_get_square_root"])
    fun.append(["fn_substitute_string_with_regex"])

    user_input = "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"
    createPrompt = BuildPrompt(prompts, funcs)
    prompt = createPrompt.build_prompt(user_input)

    predict = Prediction(fun, model, prompt)
    predict.predict_prompt(user_input)
    predict.predict_name()





