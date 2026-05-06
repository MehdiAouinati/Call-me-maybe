from llm_sdk import Small_LLM_Model
from .buildPrompt import BuildPrompt
from .decoder import Decoder
from .loader import Parse
from pydantic import ValidationError
import numpy as np
import argparse
import torch
import json
import sys


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
        sys.exit(1)

    model = Small_LLM_Model()

    funs = []
    for fun_name in funcs_list:
        funs.append(fun_name.name)

    createPrompt = BuildPrompt(prompts, funcs)
    predictor = Decoder(funs, model)

    all_of = []
    # for p in prompts_list:
    output = {
        "prompt": None,
        "name": None,
        "parameters" : None
    }

    n = {
        "prompt": None,
        "name": None,
        "parameters" : None
    }

    # prompt = createPrompt.build_prompt("Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS")
    prompt_param = createPrompt.build_param_prompt("What is the sum of 2 and 3?", funcs_list[0])
    # print(prompt_param)
    # function_name = predictor.predict_name(prompt)
    # print(function_name)
    # if function_name == "fn_no_valid_tool_found":
    #     sys.exit("ERROR: the function doesn't exist")
    param = predictor.predict_param("fn_add_numbers", funcs_list, prompt_param)
    print(param)
    try:
        json.dumps(param)
        output["prompt"] = "What is the sum of 2 and 3?"
        output["name"] = "fn_add_numbers"
        output["parameters"] = param
        all_of.append(output)
    except (TypeError, OverflowError) as e:
        print("❌ Invalid JSON:", e)

    # new = json.dumps(all_of, indent=2)
    # print(new)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # n["prompt"] = "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"
    # n["name"] = "fn_substitute_string_with_regex"
    # n["parameters"] = {"s": "string"},
    # all_of.append(output)
    # all_of.append(n)
    # param = '{"a":  5}'
    # print(all_of)
    # try:
    #     data = json.loads(output)
    #     data = json.loads(n)
    #     print("Valid JSON")
    #     all_of.append(output)
    #     all_of.append(n)
    #     new = json.dumps(all_of, indent=2)
    #     print(new)
    # except json.JSONDecodeError as e:
    #     print("Invalid JSON:")






