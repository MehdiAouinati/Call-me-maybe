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
import time


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

    #all available functions name.
    available_function_names = []
    for fun_name in funcs_list:
        available_function_names.append(fun_name.name)
    available_function_names.append("null")

    #to search for spesific function def
    function_lookup = {f.name: f for f in funcs_list}

    #initialize
    createPrompt = BuildPrompt(prompts, funcs)
    predictor = Decoder(available_function_names, model)

    #build prompts
    prompt_name = createPrompt.build_prompt()
    prompt_param = createPrompt.build_param_prompt()
    name_prompt = model.encode(prompt_name).tolist()[0]
    param_prompt = model.encode(prompt_param).tolist()[0]

    #save the data
    all_of_dict = []
    user_inp = "What is the sum of 2 and 3?"


    for p in prompts_list:
        output = {
            "prompt": None,
            "name": None,
            "parameters" : None
        }

        #predict the name of function
        addition = model.encode(f"Request: {user_inp}\nAnswer:\n").tolist()[0]
        function_name = predictor.predict_name(name_prompt + addition)
        if function_name == "null":
            sys.exit("ERROR: the function doesn't exist")

        #predict - param
        param_addition = f"Function: {function_lookup["fn_add_numbers"].name}("
        param_addition += ", ".join(f"{name}: {schema.type}" for name, schema in function_lookup["fn_add_numbers"].parameters.items())
        param_addition += ")\n "
        param_addition += f"Request: {user_inp}\n"
        param_addition += "Answer:{\n"
        param_addition = model.encode(param_addition).tolist()[0]
        param = predictor.predict_param("fn_add_numbers", funcs_list, param_prompt + param_addition)


    # measure time
    # start = time.time()
    # end = time.time()
    # second = end - start
    # minutes = second / 60
    # print(minutes)


    #convert to json
    # try:
    #     json.dumps(param)
    #     output["prompt"] = "What is the sum of 2 and 3?"
    #     output["name"] = "fn_add_numbers"
    #     output["parameters"] = param
    #     all_of.append(output)
    # except (TypeError, OverflowError) as e:
    #     print("❌ Invalid JSON:", e)
        

    # new = json.dumps(all_of, indent=2)
    # print(new)
    






