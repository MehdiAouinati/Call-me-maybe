import json
import sys
from . import parsing
from pydantic import ValidationError
from typing import List


class Parse:
    def load_functions(self, path: str) -> list[parsing.FunctionDef]:
        try:
            with open(path, "r") as file:
                jsn = json.load(file)
        except FileNotFoundError:
            print(f"Error: Could not find the file at {path}.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: The file {path} contains invalid JSON.")
            sys.exit(1)

        functions = []
        for item in jsn:
            try:
                if not all(key in item.keys() for key in ["name", "description", "parameters", "returns"]):
                    sys.exit("Error: missing keys")
                func = parsing.FunctionDef.model_validate(item)
                functions.append(func)
            except ValidationError as e:
                raise ValidationError(f"Validation error for item {item}: {e}")

        return functions

    def load_prompts(self, path: str) -> List[parsing.FunctionCall]:
        """Loads the prompts from a JSON file and returns a list of strings."""
        try:
            with open(path, 'r') as file:
                data = json.load(file)

            prompts = []
            for item in data:
                try:
                    prompt = parsing.FunctionCall.model_validate(item)
                    prompts.append(prompt)
                except ValidationError as e:
                    raise ValidationError(
                        f"Validation error for item {item}: {e}")
            return prompts
        except FileNotFoundError:
            print(f"Error: Could not find the file at {path}.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: The file {path} contains invalid JSON.")
            sys.exit(1)
