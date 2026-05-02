from llm_sdk import Small_LLM_Model
from .buildPrompt import BuildPrompt
from .decoder import Decoder
from .loader import Parse
from pydantic import ValidationError
import numpy as np
import argparse
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

    funs = []
    for fun_name in funcs_list:
        funs.append(fun_name.name)

    createPrompt = BuildPrompt(prompts, funcs)
    predictor = Decoder(funs, model)

    for p in prompts_list:
        output = {
            "prompt": None,
            "name": None,
            "parameters" : None
        }

        prompt = createPrompt.build_prompt(p.prompt)
        function_name = predictor.predict_name(prompt)
        params = predictor.predict_param(function_name, funcs_list)

        output["prompt"] = p.prompt
        output["name"] = function_name
        output["parameters"] = params


    # user_input = "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"
    # prompt = createPrompt.build_prompt(user_input)
    # predict = Decoder(fun, model, prompt)
    # # num_tokens = predict.number_tokens()
    # # predict.predict_prompt(user_input)
    # # name = predict.predict_name()
    # predict.predict_param("fn_substitute_string_with_regex", data)





