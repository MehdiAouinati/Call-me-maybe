from llm_sdk import Small_LLM_Model
from .buildPrompt import BuildPrompt
from .decoder import Decoder
from .loader import Parse
from pydantic import ValidationError
import argparse
import json
import sys
import time


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json")
    parse.add_argument(
        "--input",
        default="data/input/function_calling_tests.json")
    # parse.add_argument("--output")
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
        if not fun_name.name.startswith("fn_"):
            sys.exit(
                f"error : the function {fun_name.name} should start with 'fn_'"
                )
        available_function_names.append(fun_name.name)
    available_function_names.append("null")

    #to search for spesific function def
    function_lookup = {f.name: f for f in funcs_list}

    #initialize
    createPrompt = BuildPrompt(prompts, funcs)
    predictor = Decoder(available_function_names, model, function_lookup)

    #build prompts
    prompt_name = createPrompt.build_prompt()
    prompt_param = createPrompt.build_param_prompt()
    name_prompt = model.encode(prompt_name).tolist()[0]
    param_prompt = model.encode(prompt_param).tolist()[0]

    #save the data
    all_of_dict = []
    user_inp = "Greet shrek"

    start = time.time()
    for i, p in enumerate(prompts_list, start=1):

        # predict the name of function
        addition = model.encode(f"Request: {p.prompt}\nAnswer:\n").tolist()[0]
        function_name = predictor.predict_name(name_prompt + addition)
        if function_name == "null":
            sys.exit("ERROR: the function doesn't exist")
        print(function_name)

        # predict - param
        param_addition = f"Function: {function_lookup[function_name].name}("
        param_addition += ", ".join(f"{name}: {schema.type}" for name, schema in function_lookup[function_name].parameters.items())
        param_addition += ")\n "
        param_addition += f"Request: {p.prompt}\n"
        param_addition += "Answer:{\n"
        param_addition = model.encode(param_addition).tolist()[0]
        param = predictor.predict_param(function_name, funcs_list, param_prompt + param_addition)
        print(param)

        # animation
        print(f"\nfinish prompt {i} : {p.prompt}\n")

        # add to json
        try:
            json.dumps(param)
            all_of_dict.append({
            "prompt":     p.prompt,
            "name":       function_name,
            "parameters": param
            })
        except (TypeError, OverflowError) as e:
            print("❌ Invalid JSON:", e)

    json_string = json.dumps(my_dict, indent=4)
    with open(json, 'r') as file:
        json.dump(my_dict, file, indent=4)


    end = time.time()
    second = end - start
    minutes = second / 60



    

    # measure time
    # start = time.time()
    # second = end - start
    # minutes = second / 60
    # print(minutes)

    #convert to json

    # new = json.dumps(all_of, indent=2)
    # print(new)
