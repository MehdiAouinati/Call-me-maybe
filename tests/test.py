from llm_sdk import Small_LLM_Model
from src.tokenizer import Tokenizer
from src.decoder import Decoder
from src.buildPrompt import BuildPrompt
from src.loader import Parse
from typing import Dict, Any


def predictor_setup() -> Dict[str, Any]:
    """Set up predictor with model, tokenizer, and function definitions.

    Returns:
        Dict[str, Any]: Dictionary containing predictor, converter, prompts,
                       function lookup, and function list.
    """
    model = Small_LLM_Model()
    converting = Tokenizer(model)
    loader = Parse()
    funcs_list = loader.load_functions(
        "data/input/functions_definition.json"
    )
    funcs = [f.model_dump() for f in funcs_list]
    available_function_names = [fn.name for fn in funcs_list] + ["null"]
    function_lookup = {f.name: f for f in funcs_list}
    create_prompt = BuildPrompt([], funcs)
    predictor = Decoder(
        available_function_names,
        model,
        function_lookup,
        converting
    )
    name_prompt = converting.encode(create_prompt.build_prompt())
    param_prompt = converting.encode(create_prompt.build_param_prompt())
    return {
        "predictor": predictor,
        "converting": converting,
        "name_prompt": name_prompt,
        "param_prompt": param_prompt,
        "function_lookup": function_lookup,
        "funcs_list": funcs_list
    }


def test_empty_prompt(predictor_setup: Dict[str, Any]) -> None:
    """Test that empty prompt returns 'null' function.

    Args:
        predictor_setup: Dictionary containing predictor and converter.
    """
    predictor = predictor_setup["predictor"]
    converting = predictor_setup["converting"]
    name_prompt = predictor_setup["name_prompt"]

    prompt = ""
    addition = converting.encode(f"Request: {prompt}\nAnswer:\n")
    function_name = predictor.predict_name(name_prompt + addition)

    if function_name != "null":
        return print("valid")
    else:
        return print("Error: empty prompt")


def test_spaces_prompt(predictor_setup: Dict[str, Any]) -> None:
    """Test that spaces-only prompt is handled gracefully.

    Args:
        predictor_setup: Dictionary containing predictor and converter.
    """
    predictor = predictor_setup["predictor"]
    converting = predictor_setup["converting"]
    name_prompt = predictor_setup["name_prompt"]

    prompt = "          "
    addition = converting.encode(f"Request: {prompt}\nAnswer:\n")
    function_name = predictor.predict_name(name_prompt + addition)

    if function_name != "null":
        return print("Valid")
    else:
        return print("Error: invalid prompt")


def test_large_number(predictor_setup: Dict[str, Any]) -> None:
    """Test parameter prediction with large numbers.

    Args:
        predictor_setup: Dictionary containing predictor, function lookup,
                        and other prediction components.
    """
    predictor = predictor_setup["predictor"]
    converting = predictor_setup["converting"]
    param_prompt = predictor_setup["param_prompt"]
    function_lookup = predictor_setup["function_lookup"]
    funcs_list = predictor_setup["funcs_list"]

    prompt = "What is the sum of 9999911111111 and 8?"

    fn = function_lookup["fn_add_numbers"]
    fn_signature = ", ".join(
        f"{name}: {schema.type}"
        for name, schema in fn.parameters.items()
    )
    param_prompt_text = (
        f"Function: {fn.name}({fn_signature})\n"
        f"Request: {prompt}\n"
        f"Answer:{{\n"
    )
    param_tokens = converting.encode(param_prompt_text)
    params = predictor.predict_param(
        "fn_add_numbers", funcs_list, param_prompt + param_tokens
        )

    if params:
        print("valid")

    return


tools: Dict[str, Any] = predictor_setup()
