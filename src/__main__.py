from llm_sdk import Small_LLM_Model
from .buildPrompt import BuildPrompt
from .decoder import Decoder
from .loader import Parse
from pydantic import ValidationError
from .tokenizer import Toknizer
from pathlib import Path
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
    parse.add_argument("--output", default="data/output/function_calling_results.json")
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

    t = Toknizer(model)

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
    predictor = Decoder(available_function_names, model, function_lookup, t)

    #build prompts
    prompt_name = createPrompt.build_prompt()
    prompt_param = createPrompt.build_param_prompt()
    name_prompt = t.encode(prompt_name)
    param_prompt = t.encode(prompt_param)


    #save the data
    all_of_dict = []
    user_inp = "What is the sum of 2 and 3?"

    start = time.time()
    for i, p in enumerate(prompts_list, start=1):

        # predict the name of function
        addition = t.encode(f"Request: {p.prompt}\nAnswer:\n")
        function_name = predictor.predict_name(name_prompt + addition)
        if function_name == "null":
            sys.exit("ERROR: the function doesn't exist")

        # predict - param
        param_addition = f"Function: {function_lookup[function_name].name}("
        param_addition += ", ".join(f"{name}: {schema.type}" for name, schema in function_lookup[function_name].parameters.items())
        param_addition += ")\n "
        param_addition += f"Request: {p.prompt}\n"
        param_addition += "Answer:{\n"
        param_addition = t.encode(param_addition)
        param = predictor.predict_param(function_name, funcs_list, param_prompt + param_addition)

        # animation
        print(f"prompt {i}: finished\n")

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


    end = time.time()
    second = end - start
    minutes = second / 60
    print(minutes)

    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(all_of_dict, file, indent=4)

