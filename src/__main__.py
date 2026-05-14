from llm_sdk import Small_LLM_Model
from .buildPrompt import BuildPrompt
from .decoder import Decoder
from .loader import Parse
from pydantic import ValidationError
from .tokenizer import Tokenizer
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
    parse.add_argument(
        "--output",
        default="data/output/function_calling_results.json")
    args = parse.parse_args()

    loader = Parse()
    try:
        prompts_list = loader.load_prompts(args.input)
        funcs_list = loader.load_functions(args.functions_definition)
    except ValidationError as e:
        print(e)
        sys.exit(1)

    prompts = [item.model_dump() for item in prompts_list]
    funcs = [item.model_dump() for item in funcs_list]

    model = Small_LLM_Model()
    converting = Tokenizer(model)

    available_function_names = [fn.name for fn in funcs_list] + ["null"]

    function_lookup = {f.name: f for f in funcs_list}

    #initialize
    createPrompt = BuildPrompt(prompts, funcs)
    predictor = Decoder(
        available_function_names, model, function_lookup, converting
            )

    name_prompt = converting.encode(createPrompt.build_prompt())
    param_prompt = converting.encode(createPrompt.build_param_prompt())

    #save the data
    all_of_dict = []

    start = time.time()
    for i, p in enumerate(prompts_list, start=1):

        lenght = len(p.prompt.strip())
        if lenght == 0:
            sys.exit("Error: empty prompt")
        elif lenght > 100:
            sys.exit("Error: prompt too much big")

        print(f"Loading...", end="\r")
        # predict the name of function
        addition = converting.encode(f"Request: {p.prompt}\nAnswer:\n")
        function_name = predictor.predict_name(name_prompt + addition)

        if function_name == "null":
            sys.exit("ERROR: the function doesn't exist")

        # predict - param
        fn = function_lookup[function_name]
        fn_signature  = ", ".join(
            f"{name}: {schema.type}" for name, schema in fn.parameters.items()
                )

        param_prompt_text = (
            f"Function: {fn.name}({fn_signature})\n"
            f"Request: {p.prompt}\n"
            f"Answer:{{\n"
            )
        
        param_tokens = converting.encode(param_prompt_text)
        param = predictor.predict_param(
            function_name, funcs_list, param_prompt + param_tokens)

        # animation
        print(f"prompt {i}: finished\n")

        # add to json
        try:
            json.dumps(param)
            all_of_dict.append({
                "prompt": p.prompt,
                "name": function_name,
                "parameters": param
            })
        except (TypeError, OverflowError) as e:
            print("Invalid JSON:", e)

    end = time.time()
    second = end - start
    minutes = second / 60
    print(minutes)

    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as file:
        json.dump(all_of_dict, file, indent=4)
