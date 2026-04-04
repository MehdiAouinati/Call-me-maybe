class BuildPrompt:
    def __init__(self, prompts, funcs)
        self.prompts = prompts
        self.funcs = funcs
    
    def build_prompt(self, user_prompt):
        lines = []

        lines.append("You are a function calling assistant.")
        lines.append("Your task:")
        lines.append("Select the best function based on the user request.")

        lines.append("\nAvailable functions:")
        for fun in self.funcs:
            lines.append(f"- {fun['name']}: {fun['description']}")

        lines.append(f"\nUser request:\n\"{user_prompt}\"")

        lines.append("\nRules:")
        lines.append("- You must choose exactly ONE function")
        lines.append("- You must respond ONLY in valid JSON")
        lines.append("- Do NOT add explanations")
        lines.append("- Do NOT add extra text")

        return "\n".join(lines)
