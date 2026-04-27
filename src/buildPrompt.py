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
        lines.append('Result:{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 10, "b": 20}}')

        lines.append("\nuser: Greet john")
        lines.append('Result:{"prompt": "Greet john", "name": "fn_greet", "parameters": {"name": "Alice"}}')

        lines.append(f"\nUser request:\n\"{user_prompt}\"")

        lines.append("\nResult:")
        lines.append('"name": "')

        return "\n".join(lines)


### EXAMPLES
# User: What is 10 plus 20?
# Result: {"name": "fn_add_numbers", "parameters": {"a": 10, "b": 20}}

# User: Hello to Alice!
# Result: {"name": "fn_greet", "parameters": {"name": "Alice"}}


# ### Task
# You are a function-calling assistant. Based on the User Question, select the most appropriate function from the list below and provide its arguments in JSON format.

# ### Available Functions
# - fn_add_numbers(a: number, b: number): Add two numbers together and return their sum. [cite: 168]
# - fn_greet(name: string): Generate a greeting message for a person by name. [cite: 168]
# - fn_reverse_string(s: string): Reverse a string and return the reversed result. [cite: 168]
# - fn_get_square_root(a: number): Calculate the square root of a number.
# - fn_substitute_string_with_regex(source_string: string, regex: string, replacement: string): Replace all occurrences matching a regex pattern in a string.

# ### User Question
# "What is the sum of 2 and 3?" [cite: 164]

# ### Result
# {"name":