# noqa
# import numpy as np
# import json


# class Decoder:
#     def __init__(self, functions, model, prompt):
#         self.model = model
#         self.funcs = functions
#         self.prompt = prompt
#         self.all_fun = []
#         self.convert()


#     def convert(self):
#         for f in self.funcs:
#             check = self.model.encode(f).tolist()[0][0]
#             self.all_fun.append(check)


#     def valid_tokens_fn(self, generated_text):
#         if not generated_text:
#             res = []
#             for fun in self.all_fun:
#                 if len(fun) > 0:
#                     res.append(fun[0])
#             return list(set(res))
#         else:
#             pos = len(generated_text)
            
#             res = []
#             for fun in self.all_fun:
#                 if fun[:pos] == generated_text:
#                     if pos < len(fun):
#                         res.append(fun[pos])
            
#             return list(set(res))


#     def predict_name(self) -> str:#handle if prompt doesn't exist .
#         current_tokens = self.model.encode(self.prompt).tolist()[0]
#         generated_tokens = []
#         for i in range(10):
#             logits = self.model.get_logits_from_input_ids(current_tokens)
#             mask = np.full_like(logits, float("-inf"))
#             allowed = self.valid_tokens_fn(generated_tokens)
#             # print(self.model.decode(allowed))
#             if allowed:
#                 for a in allowed:
#                     mask[a] = 0
#             else:
#                 break

#             next_token = int(np.argmax(logits + mask))
#             current_tokens.append(next_token)
#             generated_tokens.append(next_token)
        
#         return self.model.decode(generated_tokens)

#     def get_function_def(self, fn_name, functions):
#         for fun in functions:
#             if fun["name"] == fn_name:
#                 return fun
#         return []

#     def generate_number(self, ids):
#         arr = "0123456789.-,}" 
#         num = []
#         for n in arr:
#             num.append(self.model.encode(n).tolist()[0])
        
#         generated_tokens = ""
        
#         for i in range(20):
#             logits = self.model.get_logits_from_input_ids(ids)
#             mask = np.full_like(logits, float("-inf"))

#             for n in num:
#                 mask[n] = logits[n[0]]

#             next_token = int(np.argmax(mask))
#             if 11 == next_token or 92 == next_token:
#                 break
#             ids.append(next_token)
#             generated_tokens += self.model.decode(next_token)
        
#         return float(generated_tokens), ids

#     def generate_string(self, ids):
#         generated_tokens = ""
        
#         for i in range(50):
#             logits = self.model.get_logits_from_input_ids(ids)
#             next_token = int(np.argmax(logits))
#             tokens = self.model.decode(next_token)
#             if "}" in tokens or "," in tokens:
#                 break
#             ids.append(next_token)
#             generated_tokens += self.model.decode(next_token)
        
#         return generated_tokens, ids


#     def generate_value(self, ids, param_type):
#         if param_type == "number":
#             return self.generate_number(ids)
#         elif param_type == "string":
#             return self.generate_string(ids)


#     def predict_param(self, fn_name, functions):
#         full_prompt = self.prompt + f'"{fn_name}", ' + '"parameters": { '
#         current_ids = self.model.encode(full_prompt).tolist()[0]

#         fn_def = self.get_function_def(fn_name, functions)
#         params = {}

#         for i, (param_name, param_type) in enumerate(fn_def["parameters"].items()):
#             # print(param_name)
#             # print(param_type["type"])
#             current_ids += self.model.encode(f'"{param_name}": ').tolist()[0]
#             value, current_ids = self.generate_value(current_ids, param_type["type"])
#             params[param_name] = value
#             print(value)
#         print(params)

#             # params[param_name] = value



#     # def get_word(self, user_input):
#     #     i = 0 
#     #     j = 0
#     #     first = 0
#     #     for index, char in enumerate(user_input):
#     #         if not first and char == "'":
#     #             i = index + 1
#     #             first = 1
#     #         elif first and char == "'":
#     #             j = index
#     #     return user_input[i : j]


#     # def choose_tokens(self, name, user_input):
#     #     if name in ["fn_add_numbers", "fn_get_square_root"]:
#     #         arr = "0123456789,}" 
#     #         num = []
#     #         for n in arr:
#     #             num.append(self.model.encode(n).tolist()[0])
#     #         return num
#     #     elif name in ["fn_greet"]:
#     #         arr = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'\"@#$%^&*()_+-=[]{}|;:,./<>~`'"
#     #         s = []
#     #         input_token = self.model.encode(arr).tolist()[0]
#     #         for token in input_token:
#     #             s.append([token])
#     #         s.append([1])
#     #         return s
#     #     elif name == "fn_reverse_string":
#     #         s = []
#     #         word = self.get_word(user_input)
#     #         input_token = self.model.encode(word).tolist()[0]
#     #         for token in input_token:
#     #             s.append([token])
#     #         s.append([1])
#     #         return s


#     # def first_key(self, fn_name):
#     #     if fn_name in ["fn_add_numbers", "fn_get_square_root"]:
#     #         return self.prompt  + fn_name  + '", ' + 'parameters: { "a": '
#     #     elif fn_name == "fn_greet":
#     #         return self.prompt  + fn_name  + '", ' + '"parameters": {"name": "'
#     #     elif fn_name == "fn_reverse_string":
#     #         return self.prompt  + fn_name  + '", ' + 'parameters: { "s": '
#     #     elif fn_name == "fn_substitute_string_with_regex":
#     #         return self.prompt  + fn_name  + '", ' + 'parameters: { "source_string": '
#     #     else:
#     #         return [] 

#     # def predict_param(self, fn_name, user_input):
#     #     special_prompt = self.first_key(fn_name)
#     #     print(special_prompt)
#     #     current_tokens = self.model.encode(special_prompt).tolist()[0]
#     #     generated_tokens = []
#     #     second = 0

#     #     for i in range(5):
#     #         logits = self.model.get_logits_from_input_ids(current_tokens)
#     #         mask = np.full_like(logits, float("-inf"))

#     #         if fn_name in ["fn_add_numbers"]:
#     #             if 11 in generated_tokens and not second:
#     #                 second_key = self.model.encode("b :").tolist()[0]
#     #                 for token in second_key:
#     #                     current_tokens.append(token)
#     #                 logits = self.model.get_logits_from_input_ids(current_tokens)
#     #                 second = 1

#     #             allowed = self.choose_tokens(fn_name, user_input)
#     #             for a in allowed:
#     #                 mask[a] = 0

#     #         elif fn_name == "fn_get_square_root":
#     #             allowed = self.choose_tokens(fn_name, user_input)
#     #             for a in allowed:
#     #                 mask[a] = 0
            
#     #         elif fn_name == "fn_greet":
#     #             pass
#     #             # if "}" in self.model.decode(generated_tokens):
#     #             #     break
#     #             # allowed = self.choose_tokens(fn_name, user_input)
#     #             # for a in allowed:
#     #             #     mask[a] = 0
            
#     #         # elif fn_name ==  "fn_reverse_string":
#     #         #     if 1 in generated_tokens:
#     #         #         break
#     #         #     allowed = self.choose_tokens(fn_name, user_input)
#     #         #     for a in allowed:
#     #         #         mask[a] = 0
#     #         next_token = int(np.argmax(logits))
#     #         if "}" in self.model.decode(next_token):
#     #             break
#     #         current_tokens.append(next_token)
#     #         generated_tokens.append(next_token)
        
#     #     print(generated_tokens)
#     #     print(self.model.decode(generated_tokens))
