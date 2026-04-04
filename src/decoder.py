from llm_sdk import Small_LLM_Model
import numpy as np
import torch

# class Creating:
#     def generate(self, prompt: str, valid_tokens_fn) -> str:
#         # 1. encode the prompt
#         model = Small_LLM_Model()
#         tokens = model.encode(prompt)
#         # 2. get logits
#         tokens_list = tokens.tolist()[0]
#         logits = model.get_logits_from_input_ids(tokens_list)
        # 3. call valid_tokens_fn to get the mask

        # 4. apply mask
        # 5. argmax → winner token id
        # 6. look up winner in vocab
        # 7. append to output string
        # 8. re-encode prompt + output so far
        # 9. repeat from step 2
        # 10. stop when output string ends with "}}"
        # return the complete JSON string

class Creating:
    def generate(self, prompt: str, valid_tokens_fn) -> str:
        model = Small_LLM_Model()
        current_tokens = model.encode(prompt).tolist()[0]
        generated_tokens = [] 

        for i in range(5):
            logits = model.get_logits_from_input_ids(current_tokens)
            vocab_size = len(logits)
            # Pick the winner
            if i == 0:
                mask = np.full(vocab_size, -1e9)
                mask[90] = 1
                next_token = int(np.argmax(mask))
            elif i == 1:
                mask = np.full(vocab_size, -1e9)
                mask[1] = 1
                next_token = int(np.argmax(mask))
            elif i == 2:
                mask = np.full(vocab_size, -1e9)
                mask[606] = 1
                next_token = int(np.argmax(mask))
            elif i == 3:
                mask = np.full(vocab_size, -1e9)
                mask[1] = 1
                next_token = int(np.argmax(mask))
            elif i == 4:
                mask = np.full(vocab_size, -1e9)
                mask[25] = 1
                next_token = int(np.argmax(mask))
            else:
                next_token = int(np.argmax(logits))
            # ADD TO BOTH: 
            # 1. Add to history so AI can 'remember' it for the next loop
            current_tokens.append(next_token)
            # 2. Add to your basket so you can see it later
            generated_tokens.append(next_token)
            
            # OPTIONAL: Print it immediately to see it happening live
            # print(f"AI chose token ID: {next_token} -> '{model.decode([next_token])}'")

        # Return ONLY what the AI created
        # print(model.decode(generated_tokens))
        return model.decode(generated_tokens)