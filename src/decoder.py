import numpy as np


class Creating:
    def __init__(self, functions, model):
        self.model = model
        self.funcs = functions
        self.all_fun = []
        self.convert()


    def convert(self):
        for f in self.funcs:
            check = self.model.encode(f).tolist()[0][0]
            self.all_fun.append(check)


    def valid_tokens_fn(self, generated_text):
        if not generated_text:
            res = []
            for fun in self.all_fun:
                if len(fun) > 0:
                    res.append(fun[0])
            return list(set(res))
        else:
            # current_gen_ids = model.encode(generated_text).tolist()[0]
            pos = len(generated_text)
            
            res = []
            for fun in self.all_fun:
                if fun[:pos] == generated_text:
                    if pos < len(fun):
                        res.append(fun[pos])
            
            return list(set(res))


    def generate_name(self, prompt: str) -> str:#handle if prompt doesn't exist .
        prefix_ids = self.model.encode('"name": ').tolist()[0]
        current_tokens = self.model.encode(prompt).tolist()[0]
        generated_tokens = []

        for i in range(10):
            logits = self.model.get_logits_from_input_ids(current_tokens)
            mask = np.full_like(logits, float("-inf"))

            if i < len(prefix_ids):
                mask[prefix_ids[i]] = 0 
            else:
                allowed = self.valid_tokens_fn(generated_tokens[len(prefix_ids):])
                if allowed:
                    for a in allowed:
                        mask[a] = 0
                else:
                    break 

            next_token = int(np.argmax(logits + mask))
            current_tokens.append(next_token)
            generated_tokens.append(next_token)
            
        print(self.model.decode(generated_tokens))