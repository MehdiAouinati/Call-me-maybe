import argparse
from .loader import Parse
import numpy as np
from pydantic import ValidationError
from .buildPrompt import BuildPrompt
from .decodernew import Decoder
from llm_sdk import Small_LLM_Model
import torch
import json


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
    with open(args.functions_definition, "r") as file:
        data = json.load(file)

    fun = []
    for a in data:
        fun.append([a["name"]])

    # fun = []
    # fun.append(['fn_add_numbers'])
    # fun.append(['fn_greet'])
    # fun.append(['fn_reverse_string'])
    # fun.append(['fn_get_square_root'])
    # fun.append(['fn_substitute_string_with_regex'])
    # fun.append(['fn_no_valid_tool_found'])

    user_input = "Replace all vowels in 'Programming is fun' with asterisks"
    createPrompt = BuildPrompt(prompts, funcs)
    prompt = createPrompt.build_prompt(user_input)

    predict = Decoder(fun, model, prompt)
    # num_tokens = predict.number_tokens()
    # predict.predict_prompt(user_input)
    # name = predict.predict_name()
    predict.predict_param("fn_substitute_string_with_regex", data)





