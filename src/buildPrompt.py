import numpy as np
import json

class BuildPrompt:
    def __init__(self, prompts, funcs):
        self.prompts = prompts
        self.funcs = funcs

    def build_prompt(self, user_prompt):
        text = "You select the correct function name based on a user request.\n"

        text += "\nSTRICT RULES:\n"
        text += "- Output ONLY the function name as plain text.\n"
        text += "- Do NOT return JSON.\n"
        text += "- Do NOT add quotes.\n"
        text += "- Do NOT explain.\n"

        text += "\nFUNCTION RULES:\n"
        text += "- Choose ONLY from the available functions.\n"
        text += "- Do NOT invent new function names.\n"

        text += "\nEDGE CASES:\n"
        text += "- If the request is empty → return: null\n"
        text += "- If no function matches → return: null\n"

        text += "\nAVAILABLE FUNCTIONS:\n"

        for fn in self.funcs:
            text += f"- {fn["name"]}: {fn["description"]}\n"

        text += "\nEXAMPLES:\n"

        text += "Request: What is the sum of 2 and 3?\n"
        text += "Answer:\nfn_add_numbers\n\n"

        text += "Request: Greet John\n"
        text += "Answer:\nfn_greet\n\n"

        text += "Request: Reverse the string \"hello\"\n"
        text += "Answer:\nfn_reverse_string\n\n"

        text += "Request: What is the square root of 16?\n"
        text += "Answer:\nfn_get_square_root\n\n"

        text += "Request: Replace numbers in text\n"
        text += "Answer:\nfn_substitute_string_with_regex\n\n"

        text += "Request: Tell me a joke\n"
        text += "Answer:\nnull\n\n"

        text += "Request:\n"
        text += "Answer:\nnull\n\n"

        text += "\nNow select the function name.\n"
        text += f"Request: {user_prompt}\n"
        text += "Answer:\n"
        return text


    def build_param_prompt(self, user_prompt, fn_def):
        prompt = "You extract ONLY function parameters from a user request.\n"

        prompt += "\nSTRICT RULES:\n"
        prompt += "- Return ONLY a valid JSON object.\n"
        prompt += "- Do NOT explain anything.\n"
        prompt += "- Do NOT execute the function.\n"
        prompt += "- Do NOT infer results, only extract inputs.\n"

        prompt += "\nTYPE RULES:\n"
        prompt += "- Numbers must be numeric (no quotes).\n"
        prompt += "- Strings must be valid JSON strings.\n"

        prompt += "\nREGEX RULES (VERY IMPORTANT):\n"
        prompt += "- Regex must ALWAYS be a valid JSON string.\n"
        prompt += "- Escape backslashes correctly.\n"
        prompt += "- Use EXACT patterns, no variation allowed.\n"

        prompt += "- For 'all numbers' → use EXACTLY \"\\d+\"\n"
        prompt += "- For 'all vowels' → use EXACTLY \"[aeiouAEIOU]\"\n"

        prompt += "- DO NOT use alternatives like:\n"
        prompt += "  ❌ \"d+\"\n"
        prompt += "  ❌ \"\\\\\\\\d+\"\n"
        prompt += "  ❌ \"aeiou\"\n"
        prompt += "  ❌ \"(a|e|i|o|u)\"\n"

        prompt += "- ONLY use the exact allowed regex.\n"

        prompt += "\nLANGUAGE RULES:\n"
        prompt += "- 'half of X' → X / 2\n"
        prompt += "- Convert words to numbers: one=1, two=2, three=3\n"

        prompt += "\nEXAMPLES:\n"

        prompt += "Function: fn_add_numbers(a: number, b: number)\n"
        prompt += "Request: What is the sum of 2 and 3?\n"
        prompt += 'Answer:\n{"a": 2.0, "b": 3.0}\n'

        prompt += "Function: fn_greet(name: string)\n"
        prompt += "Request: Greet shrek\n"
        prompt += 'Answer:\n{"name": "shrek"}\n'

        prompt += "Function: fn_reverse_string(s: string)\n"
        prompt += "Request: Reverse the string 'hello'\n"
        prompt += 'Answer:\n{"s": "hello"}\n'

        prompt += "Function: fn_get_square_root(a: number)\n"
        prompt += "Request: What is the square root of 16?\n"
        prompt += 'Answer:\n{"a": 16.0}\n'

        prompt += "Function: fn_substitute_string_with_regex(source_string: string, regex: string, replacement: string)\n"
        prompt += 'Request: Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS\n'
        prompt += 'Answer:\n{"source_string": "Hello 34 I\'m 233 years old", "regex": "\\d+", "replacement": "NUMBERS"}\n'

        prompt += "Function: fn_substitute_string_with_regex(source_string: string, regex: string, replacement: string)\n"
        prompt += 'Request: Replace all vowels in "Programming is fun" with "*"\n'
        prompt += 'Answer:\n{"source_string": "Programming is fun", "regex": "[aeiouAEIOU]", "replacement": "*"}\n'

        prompt += "\nNow extract parameters for this request.\n"

        prompt += f"Function: {fn_def.name}("
        prompt += ", ".join(f"{name}: {schema.type}" for name, schema in fn_def.parameters.items())
        prompt += ")\n"

        prompt += f"Request: {user_prompt}\n"
        prompt += "Answer:{\n"
        return prompt 