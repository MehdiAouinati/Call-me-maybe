class BuildPrompt:
    def __init__(self, prompts, funcs):
        self.prompts = prompts
        self.funcs = funcs

    def build_prompt(self):
        prompt = "You select the correct function name based on a user request"
        prompt += ".\n"

        prompt += "\nSTRICT RULES:\n"
        prompt += "- Output ONLY the function name as plain text.\n"
        prompt += "- Do NOT return JSON.\n"
        prompt += "- Do NOT add quotes.\n"
        prompt += "- Do NOT explain.\n"

        prompt += "\nFUNCTION RULES:\n"
        prompt += "- Choose ONLY from the available functions.\n"
        prompt += "- Do NOT invent new function names.\n"

        prompt += "\nEDGE CASES:\n"
        prompt += "- If the request is empty → return: null\n"
        prompt += "- If no function matches → return: null\n"

        prompt += "\nAVAILABLE FUNCTIONS:\n"

        for fn in self.funcs:
            prompt += f"- {fn["name"]}: {fn["description"]}\n"

        prompt += "\nEXAMPLES:\n"

        prompt += "Request: What is the sum of 2 and 3?\n"
        prompt += "Answer:\nfn_add_numbers\n\n"

        prompt += "Request: Greet John\n"
        prompt += "Answer:\nfn_greet\n\n"

        prompt += "Request: Reverse the string \"hello\"\n"
        prompt += "Answer:\nfn_reverse_string\n\n"

        prompt += "Request: What is the square root of 16?\n"
        prompt += "Answer:\nfn_get_square_root\n\n"

        prompt += "Request: Replace numbers in text\n"
        prompt += "Answer:\nfn_substitute_string_with_regex\n\n"

        prompt += "Request: Tell me a joke\n"
        prompt += "Answer:\nnull\n\n"

        prompt += "Request:\n"
        prompt += "Answer:\nnull\n\n"

        prompt += "\nNow select the function name.\n"

        return prompt

    def build_param_prompt(self):
        prompt = "You extract ONLY function parameters from a user request.\n"

        prompt += "\nSTRICT RULES:\n"
        prompt += "- Return ONLY a valid JSON object.\n"
        prompt += "- Do NOT explain anything.\n"
        prompt += "- Do NOT execute the function.\n"
        prompt += "- Do NOT infer results, only extract inputs.\n"

        prompt += "\nCRITICAL — NEVER EXECUTE THE FUNCTION:\n"
        prompt += "- Extract ONLY what the user provided as raw input.\n"
        prompt += "- NEVER compute, reverse, sort, add, transform, square, or process the value.\n"
        prompt += "- The function will be called separately — your job is ONLY to extract.\n"
        prompt += "- 'square root of X' -> extract X as-is, do NOT compute √X\n"
        prompt += "- 'asterisks' or 'an asterisk' -> use EXACTLY \"*\" (the character, not the word)\n"
        prompt += "- 'reverse string X' -> extract X as-is, do NOT compute reversing\n"


        prompt += "\nTYPE RULES:\n"
        prompt += "- Numbers must be numeric (no quotes).\n"
        prompt += "- Strings must be valid JSON strings.\n"

        prompt += "\nREGEX RULES (VERY IMPORTANT):\n"
        prompt += "- Regex must ALWAYS be a valid JSON string.\n"
        prompt += "- Escape backslashes correctly.\n"
        prompt += "- Use EXACT patterns, no variation allowed.\n"

        prompt += "- For 'all numbers' → use EXACTLY \"\\d+\"\n"
        prompt += "- For 'all vowels' → use EXACTLY \"[aeiouAEIOU]\"\n"

        prompt += "- ONLY use the exact allowed regex.\n"

        prompt += "\nLANGUAGE RULES:\n"
        prompt += "- 'half of X' → X / 2\n"
        prompt += "- Convert words to numbers: one=1, two=2, three=3\n"

        prompt += "\nEXAMPLES:\n"

        # prompt += "Function: fn_add_numbers(a: number, b: number)\n"
        # prompt += "Request: What is the sum of 2 and 3?\n"
        # prompt += 'Answer:\n{"a": 2.0, "b": 3.0}\n'

        # prompt += "Function: fn_greet(name: string)\n"
        # prompt += "Request: Greet shrek\n"
        # prompt += 'Answer:\n{"name": "shrek"}\n'

        prompt += "Function: fn_reverse_string(s: string)\n"
        prompt += "Request: Reverse the string 'hello'\n"
        prompt += "# WARNING: Do NOT reverse the string. Extract 'hello' as-is.\n"
        prompt += 'Answer:\n{"s": "hello"}\n'

        # prompt += "Function: fn_get_square_root(a: number)\n"
        # prompt += "Request: What is the square root of 16?\n"
        # prompt += 'Answer:\n{"a": 16.0}\n'

        # prompt += "Function: fn_get_square_root(a: number)\n"
        # prompt += "Request: Calculate the square root of 144?\n"
        # prompt += 'Answer:\n{"a": 144.0}\n'

        prompt += "Function: fn_substitute_string_with_regex(source_string:"
        prompt += " string, regex: string, replacement: string)\n"
        prompt += 'Request: Replace all numbers in '
        prompt += '"Hello 34 I\'m 233 years old" with NUMBERS\n'
        prompt += 'Answer:\n{"source_string": "Hello 34 I\'m 233 years old",'
        prompt += ' "regex": "\\d+", "replacement": "NUMBERS"}\n'

        prompt += "Function: fn_substitute_string_with_regex(source_string: "
        prompt += "string, regex: string, replacement: string)\n"
        prompt += 'Request: '
        prompt += 'Replace all vowels in "Programming is fun" with "*"\n'
        prompt += 'Answer:\n{"source_string": "Programming is fun",'
        prompt += ' "regex": "[aeiouAEIOU]", "replacement": "*"}\n'

        prompt += "\nNow extract parameters for this request.\n"

        return prompt
