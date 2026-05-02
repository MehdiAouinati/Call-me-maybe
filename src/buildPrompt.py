import numpy as np
import json

class BuildPrompt:
    def __init__(self, prompts, funcs):
        self.prompts = prompts
        self.funcs = funcs
    
    def build_prompt(self, user_prompt):
        lines = []
        lines.append("Your task:")
        lines.append("You are a function-calling assistant. Based on the User Question, select the most appropriate function from the list below and provide its arguments in JSON format.")

        lines.append("\nRules:")
        lines.append("- Identify the best function from the list below")
        lines.append("- You must respond ONLY in valid JSON")
        lines.append("- Do NOT add explanations")
        lines.append("- Do NOT add extra text")
        lines.append("- Output MUST be a single JSON object.")
        lines.append("- For regex patterns: digits/numbers → \\d+, vowels → [aeiou], specific word like 'cat' → cat, spaces → \\s+, letters → [a-zA-Z]+")
        lines.append("- Extract arguments exactly as they appear in the user request.")

        lines.append("\nAvailable functions:")
        for fun in self.funcs:
            param = {}
            for k, v in fun["parameters"].items():
                for kk, vv in v.items():
                    param[k] = vv

            lines.append(f"- {fun['name']}: Parameters {param}: {fun['description']}")
        
        lines.append("""- fn_no_valid_tool_found: Use this function ONLY when 
            the user's request cannot be fulfilled by any other available tool.
            This includes general conversation, questions about topics not
            covered by other functions(like weather, news, or history), or
            when the user's intent is unclear. Selecting this prevents
            incorrect tool usage.""")

        lines.append("\nEXAMPLES:")
        lines.append("User: What is the sum of 2 and 3?")
        lines.append('Result:{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2, "b": 3}}')

        lines.append("\nuser: Greet john")
        lines.append('Result:{"prompt": "Greet mehdi", "name": "fn_greet", "parameters": {"name": "mehdi"}}')

        lines.append("\nuser: Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'")
        lines.append('Result:{"prompt": "Substitute the word "cat" with "dog" in "The cat sat on the mat with another cat"",') 
        lines.append('"name": "fn_substitute_string_with_regex", "parameters": {"source_string": "The cat sat on the mat with another cat", "regex": "cat", "replacement": "dog"}}')
        lines.append("\nuser: Replace all numbers in \"Hello 34 I\'m 233 years old\" with NUMBERS")
        lines.append('Result:{"prompt": "Replace all numbers in \"Hello 34 I\'m 233 years old\" with NUMBERS",') 
        lines.append('"name": "fn_substitute_string_with_regex", "parameters": {"source_string": "\"Hello 34 I\'m 233 years old\"", "regex": "\\d+", "replacement": "NUMBERS"}}')

        lines.append(f"\nUser request:\n\"{user_prompt}\"")


        lines.append("\nResult:")
        lines.append(f'{{"prompt": "{user_prompt}", "name": ')

        return "\n".join(lines)
