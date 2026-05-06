import numpy as np
import json


class Decoder:
    def __init__(self, functions, model):
        self.model = model
        self.funcs = functions
        self.all_fun = []

        self.comma = self.model.encode(',').tolist()[0][0]
        self.quote = self.model.encode('"').tolist()[0][0]
        self.close = self.model.encode('}').tolist()[0][0]
        self.number_tokens_ids = self.build_number_tokens()
        self.convert()


    def convert(self):
        for f in self.funcs:
            check = self.model.encode(f).tolist()[0]
            self.all_fun.append(check)


    def build_number_tokens(self):
        arr = "0123456789.-,}" 
        num = []
        for n in arr:
            num.append(self.model.encode(n).tolist()[0])
        return num


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


    def get_function_def(self, fn_name, functions):
        for fun in functions:
            if fun.name == fn_name:
                return fun
        return []


    def generate_value(self, ids, param_type):
        if param_type == "number":
            return self.generate_number(ids)
        elif param_type == "string":
            return self.generate_string(ids)


    def generate_number(self, ids):
        generated_tokens = ""
        
        for i in range(20):
            logits = self.model.get_logits_from_input_ids(ids)
            mask = np.full_like(logits, float("-inf"))

            for num in self.number_tokens_ids:
                mask[num] = logits[num[0]]

            next_token = int(np.argmax(mask))
            if self.comma == next_token or self.close == next_token:
                break
            ids.append(next_token)
            generated_tokens += self.model.decode(next_token)
        
        return float(generated_tokens), ids


    def generate_string(self, ids):
        generated_tokens = ""
        
        for i in range(50):
            logits = self.model.get_logits_from_input_ids(ids)
            next_token = int(np.argmax(logits))
            tokens = self.model.decode(next_token)
            # print(tokens)
            if '}' in tokens or ',' in tokens:
                if '}' in tokens:
                    before_quote = tokens.split('}')[0]
                else:
                    before_quote = tokens.split(',')[0]
                generated_tokens += before_quote
                ids.append(next_token)
                break
            ids.append(next_token)
            generated_tokens += self.model.decode(next_token)
        
        return generated_tokens, ids


    def predict_param(self, fn_name, functions, prompt):
        # full_prompt = prompt + f'"{fn_name}", ' + '"parameters": { '
        curr_ids = self.model.encode(prompt).tolist()[0]

        fn_def = self.get_function_def(fn_name, functions)
        params = {}

        for i, (param_name, param_type) in enumerate(fn_def.parameters.items()):
            curr_ids += self.model.encode(f'"{param_name}": ').tolist()[0]
            if param_type.type == "string":
                # curr_ids += self.model.encode('"').tolist()[0]
                value, curr_ids = self.generate_string(curr_ids)
                value = value.strip().strip('"')
            elif param_type.type == "number":
                value, curr_ids = self.generate_number(curr_ids)
            # value, curr_ids = self.generate_value(curr_ids, param_type.type)
            # print(value)
            params[param_name] = value
            # print

        return params