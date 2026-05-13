from llm_sdk import Small_LLM_Model
import json

# llm = Small_LLM_Model()

class Toknizer:
    def __init__(self, model):
        self.model = model
        self.path = model.get_path_to_vocab_file()
        self.ids = {}
        with open(self.path, 'r') as file:
            self.vocab = json.load(file)
        
        for key, v in self.vocab.items():
            self.ids[v] = key

    def encode(self, word: str) -> list[int]:
        tokens = []
        word = word.replace(" ", "Ġ")
        word = word.replace("\n", "Ċ")
        while word:
            matched = False
            for end in range(len(word), 0, -1):
                token_id = self.vocab.get(word[:end])
                if token_id is not None:
                    tokens.append(token_id)
                    word = word[end:]
                    matched = True
                    break
            if not matched:
                word = word[1:]

        return tokens

    def decode(self, tokenz):
        if isinstance(tokenz, int):
            tokenz = [tokenz]
        
        string = "".join(self.ids[t] for t in tokenz)  # join is faster than +=
        
        return (
            string
            .replace("Ċ", "\n")
            .replace("Ġ", " ")
            .replace("ĉ", "\t")  # tab, consistent with encode
        )

# tt = Toknizer(llm)
# var = """STRICT RULES:
# - Output ONLY the function name as plain text.
# - Do NOT return JSON.
# - Do NOT add quotes.
# - Do NOT explain.
# """
# c = tt.encode(var)
# n = tt.decode(c)
# print(n)