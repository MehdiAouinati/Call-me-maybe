from llm_sdk import Small_LLM_Model
import json

llm = Small_LLM_Model()

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
                print("not found")
                word = word[1:]

        return tokens

    def decode(self, tokenz):
        string  = ""
        for t in tokenz:
            string += self.ids[t]
        return string.replace("Ġ", " ")

tt = Toknizer(llm)
c = tt.encode("hello     world")
n = tt.decode(c)
print(len(n))