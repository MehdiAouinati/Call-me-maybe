import json
import sys
from . import parsing
from pydantic import ValidationError
from typing import List


class Parse:
    """Utility for loading and validating JSON-based definitions.

    The `Parse` class provides helpers to load JSON files that define
    function metadata (`FunctionDef`) and function-call prompts
    (`FunctionCall`) and validate them using the `parsing` module's
    Pydantic models.

    The methods will exit the program with an error message if the file
    cannot be found or is not valid JSON. Validation errors raised by
    the Pydantic models are re-raised to preserve details.
    """

    def load_functions(self, path: str) -> list[parsing.FunctionDef]:
        """Load and validate function definitions from a JSON file.

        This method opens the JSON file at `path`, parses it as a list of
        mappings and validates each item using
        `parsing.FunctionDef.model_validate`.

        Parameters
        ----------
        path : str
            Filesystem path to a JSON file containing a list of function
            definition objects. Each object must include the keys
            ``name``, ``description``, ``parameters`` and ``returns``.

        Returns
        -------
        list[parsing.FunctionDef]
            A list of validated `FunctionDef` instances.

        Raises
        ------
        ValidationError
            If any item in the JSON does not validate against the
            `FunctionDef` model. The function will call ``sys.exit`` with
            an error message if the file is missing or contains invalid
            JSON.
        """
        try:
            with open(path, "r") as file:
                jsn = json.load(file)
        except FileNotFoundError:
            sys.exit(f"Error: Could not find the file at {path}.")
        except json.JSONDecodeError:
            sys.exit(f"Error: The file {path} contains invalid JSON.")

        functions: list[parsing.FunctionDef] = []
        for item in jsn:
            try:
                if not all(key in item.keys() for key in [
                        "name", "description", "parameters", "returns"
                        ]):
                    sys.exit(f"Error: missing keys in json file {path}")
                func = parsing.FunctionDef.model_validate(item)
                functions.append(func)
            except ValidationError as e:
                raise ValidationError(f"Validation error for item {item}: {e}")

        return functions

    def load_prompts(self, path: str) -> List[parsing.FunctionCall]:
        """Load and validate function-call prompts from a JSON file.

        Parameters
        ----------
        path : str
            Path to a JSON file containing a list of prompt objects that
            must conform to the `parsing.FunctionCall` model.

        Returns
        -------
        List[parsing.FunctionCall]
            A list of validated `FunctionCall` instances.

        Raises
        ------
        ValidationError
            If any prompt item fails Pydantic validation. The method will
            call ``sys.exit`` if the file is absent or contains invalid
            JSON.
        """
        try:
            with open(path, 'r') as file:
                data = json.load(file)

            prompts: List[parsing.FunctionCall] = []
            for item in data:
                try:
                    prompt = parsing.FunctionCall.model_validate(item)
                    prompts.append(prompt)
                except ValidationError as e:
                    sys.exit(f"Validation error for item {item}: {e}")
            return prompts
        except FileNotFoundError:
            sys.exit(f"Error: Could not find the file at {path}.")
        except json.JSONDecodeError:
            sys.exit(f"Error: The file {path} contains invalid JSON.")
