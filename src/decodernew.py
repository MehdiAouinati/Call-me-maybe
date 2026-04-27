import numpy as np
import json


class Decoder:

    def __init__(self, functions, model, prompt):
        self.model = model
        self.funcs = functions
        self.prompt = prompt
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


    def predict_name(self) -> str:#handle if prompt doesn't exist .
        current_tokens = self.model.encode(self.prompt).tolist()[0]
        generated_tokens = []
        for i in range(10):
            logits = self.model.get_logits_from_input_ids(current_tokens)
            mask = np.full_like(logits, float("-inf"))
            allowed = self.valid_tokens_fn(generated_tokens)
            # print(self.model.decode(allowed))
            if allowed:
                for a in allowed:
                    mask[a] = 0
            else:
                break

            next_token = int(np.argmax(logits + mask))
            current_tokens.append(next_token)
            generated_tokens.append(next_token)
        
        return self.model.decode(generated_tokens)

    def choose_tokens(self, name):
        arr = "0123456789" 
        if name in ["fn_add_numbers"]:
            num = []
            for n in arr:
                num.append(self.model.encode(n).tolist()[0])
            return num + [11, 92] 

    def predict_param(self, fn_name):
        special_prompt = self.prompt  + fn_name  + '", ' + 'parameters: { "a": '
        current_tokens = self.model.encode(special_prompt).tolist()[0]
        generated_tokens = []

        for i in range(5):
            logits = self.model.get_logits_from_input_ids(current_tokens)
            mask = np.full_like(logits, float("-inf"))
            if fn_name in ["fn_add_numbers"]:
                if 92 in generated_tokens:
                    break
                allowed = self.choose_tokens(fn_name)
                for a in allowed:
                    mask[a] = 0
                # if 11 in allowed:
                #     current_tokens.append(self.model.encode("b :").tolist()[0])

            next_token = int(np.argmax(logits + mask))
            current_tokens.append(next_token)
            generated_tokens.append(next_token)
        
        print(self.model.decode(generated_tokens))
        # print()
