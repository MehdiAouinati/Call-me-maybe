import numpy as np
import json


class Decoder:
    def __init__(self, functions, model):
        self.model = model
        self.funcs = functions
        self.all_fun = []
        self.convert()


    def convert(self):
        for f in self.funcs:
            check = self.model.encode(f).tolist()[0]
            self.all_fun.append(check)


    def valid_tokens_fn_name(self, generated_text):
        if not generated_text:
            res = []
            for fun in self.all_fun:
                if len(fun) > 0:
                    res.append(fun[0])
            return list(set(res))
        else:
            pos = len(generated_text)
            
            res = []
            for fun in self.all_fun:
                if fun[:pos] == generated_text:
                    if pos < len(fun):
                        res.append(fun[pos])
            
            return list(set(res))


    def predict_name(self, prompt) -> str:
        current_tokens = self.model.encode(prompt).tolist()[0]
        generated_tokens = []

        for i in range(20):
            logits = self.model.get_logits_from_input_ids(current_tokens)
            mask = np.full_like(logits, float("-inf"))
            allowed = self.valid_tokens_fn_name(generated_tokens)
            if allowed:
                for a in allowed:
                    mask[a] = 0
            else:
                break

            next_token = int(np.argmax(logits + mask))
            current_tokens.append(next_token)
            generated_tokens.append(next_token)
        
        return self.model.decode(generated_tokens)