from typing import Any, Dict, List, Tuple, Union
import numpy as np
import sys


class Decoder:
    def __init__(
        self, functions: List[Any], model: Any, function_lookup: Any,
        tokenizer: Any
            ) -> None:
        self.model = model
        self.funcs = functions
        self.all_fun: List[List[int]] = []
        self.convert()
        self.number_tokens_ids: List[List[int]] = self.build_number_tokens()

        self.tokenizer = tokenizer
        self._fn_map: Dict[str, Any] = function_lookup

        self.close = self.model.encode('}').tolist()[0][0]
        self.comma = self.model.encode(',').tolist()[0][0]

    def convert(self) -> None:
        for f in self.funcs:
            check = self.model.encode(f).tolist()[0]
            self.all_fun.append(check)

    def build_number_tokens(self) -> List[List[int]]:
        return [self.model.encode(n).tolist()[0] for n in "0123456789.-,}"]

    def valid_tokens_fn_name(self, generated_text: List[int]) -> List[int]:
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

    def predict_name(self, tokens: List[int]) -> str:
        current_tokens = tokens
        generated_tokens: list[int] = []

        for i in range(25):
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

        return self.tokenizer.decode(generated_tokens)

    def generate_number(self, ids: List[int]) -> Tuple[float, List[int]]:
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
            generated_tokens += self.tokenizer.decode(next_token)
            if len(generated_tokens) >= 13:
                sys.exit("Error :the number too much big")

        return float(generated_tokens), ids

    def generate_string(self, ids: List[int]) -> Tuple[str, List[int]]:
        generated_tokens = ""

        for i in range(60):
            logits = self.model.get_logits_from_input_ids(ids)
            next_token = int(np.argmax(logits))
            tokens = self.tokenizer.decode(next_token)
            if '}' in tokens or ',' in tokens:
                if '}' in tokens:
                    before_quote = tokens.split('}')[0]
                else:
                    before_quote = tokens.split(',')[0]
                generated_tokens += before_quote
                ids.append(next_token)
                break
            ids.append(next_token)
            generated_tokens += self.tokenizer.decode(next_token)

        return generated_tokens, ids

    def predict_param(
            self, fn_name: str, functions: List[Any], tokens: List[int]
            ) -> Dict[str, Any]:

        curr_ids = tokens

        fn_def = self._fn_map.get(fn_name)

        if fn_def is None:
            sys.exit(f"Unknown function: {fn_name}")

        params: Dict[str, Any] = {}

        for i, (p_name, p_type) in enumerate(fn_def.parameters.items()):
            curr_ids += self.tokenizer.encode(f'"{p_name}": ')
            value: Union[str, float]
            if p_type.type == "number":
                value, curr_ids = self.generate_number(curr_ids)
            else:
                value, curr_ids = self.generate_string(curr_ids)
                value = value.strip().strip('"')
            params[p_name] = value

        return params
