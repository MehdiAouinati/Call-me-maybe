import numpy as np
import json


class Prediction:
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
        prefix_ids = self.model.encode('"name": "').tolist()[0]
        current_tokens = self.model.encode(self.prompt).tolist()[0]
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
        return self.model.decode(generated_tokens)


    def predict_prompt(self, user_input):
        prefix_ids = self.model.encode('{"prompt": "').tolist()[0]
        current_tokens = self.model.encode(self.prompt).tolist()[0]
        generated_tokens = []

        for i in range(len(prefix_ids)):
            logits = self.model.get_logits_from_input_ids(current_tokens)
            mask = np.full_like(logits, float("-inf"))#read about it 
            mask[prefix_ids[i]] = 0

            next_token = int(np.argmax(logits + mask))#read about it
            current_tokens.append(next_token)
            generated_tokens.append(next_token)

        res = self.model.decode(generated_tokens)
        res += user_input + '",'
        print(res)
        return res

        # " = 1
        # a , b = 64, 65
        # , = 11
        # : = 25
        # } = 92

    def number_tokens(self):
        path = self.model.get_path_to_vocab_file()
        with open(path, "r") as file:
            data = json.load(file)
        digit = []
        for key, value in data.items():
            clean_key = key.replace('Ġ', '').replace(' ', '').strip()
            try:
                if clean_key != "":
                    n = float(clean_key)
                    digit.append(value)
            except (ValueError, TypeError):
                pass
        return digit

    def param_valid(self, fn_name, number_tokens, generated_text):
        last_char = generated_text[-1] if generated_text else ""

        if fn_name == '"fn_add_numbers",':
            if last_char == "{" or last_char == ",":
                return [1] 

            elif last_char == '"':
                if generated_text[-2] == 'a' or generated_text[-2] == 'b':
                    return [25]
                else:
                    return [64, 65]

            elif last_char == 'a' or last_char == 'b':
                return [1]

            elif last_char == ':':
                return number_tokens

            elif last_char.isdigit() or last_char == ".":
                return number_tokens + [11, 92]
                
        return []
            


    def predict_param(self, fn_name, number_tokens):
        key, name = fn_name.split(" ")
        prefix_ids = self.model.encode('"parameters": {').tolist()[0]
        current_tokens = self.model.encode(self.prompt).tolist()[0]
        generated_tokens = []

        for i in range(11):
            logits = self.model.get_logits_from_input_ids(current_tokens)
            mask = np.full_like(logits, float("-inf"))

            if i < len(prefix_ids):
                mask[prefix_ids[i]] = 0
            else:
                allowed = self.param_valid(name,number_tokens, self.model.decode(generated_tokens))
                for a in allowed:
                    mask[a] = 0
                if allowed:
                else:
                    break

            next_token = int(np.argmax(logits + mask))#read about it
            current_tokens.append(next_token)
            generated_tokens.append(next_token)
        
        print(self.model.decode(generated_tokens))