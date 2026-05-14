from typing import Dict, Iterable, List, Union
from llm_sdk import Small_LLM_Model
import json


class Tokenizer:
    """Simple tokenizer wrapper around a Small_LLM_Model vocabulary.

    This class loads a vocabulary file from the provided model and provides
    simple `encode` and `decode` helpers that map between strings and
    token id sequences. The tokenizer uses special marker characters
    found in the model vocabulary: `Ġ` for space, `Ċ` for newline and
    `ĉ` for tab.

    Args:
        model: An object providing `get_path_to_vocab_file()` which returns
            the filesystem path to a JSON vocabulary mapping string->int.
    """

    def __init__(self, model: Small_LLM_Model) -> None:
        """Load the vocabulary from the model and build reverse mapping.

        Parameters
        ----------
        model : object
            An object with a `get_path_to_vocab_file()` method returning the
            path to a JSON file containing a dict mapping token strings to
            integer ids.
        """
        self.model: Small_LLM_Model = model
        self.path: str = model.get_path_to_vocab_file()

        with open(self.path, 'r') as file:
            self.vocab: Dict[str, int] = json.load(file)

        self.ids: Dict[int, str] = {v: k for k, v in self.vocab.items()}

    def encode(self, word: str) -> List[int]:
        """Encode a string into a list of token ids.

        The method applies simple preprocessing replacing spaces and newlines
        with the model's special markers before performing a greedy longest-
        match tokenization using the loaded vocabulary.

        Parameters
        ----------
        word : str
            The input string to tokenize.

        Returns
        -------
        list[int]
            A list of integer token ids corresponding to the input string.
        """
        tokens: List[int] = []
        word = word.replace(" ", "Ġ")
        word = word.replace("\n", "Ċ")
        word = word.replace("\t", 'ĉ')
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

    def decode(self, tokenz: Union[int, Iterable[int]]) -> str:
        """Decode token id(s) back into a string.

        Parameters
        ----------
        tokenz : int or Iterable[int]
            A single token id or an iterable of token ids to decode.

        Returns
        -------
        str
            The decoded string where special markers are converted back to
            their original whitespace characters (newline, space, tab).
        """
        if isinstance(tokenz, int):
            tokenz = [tokenz]

        string = "".join(self.ids[int(t)] for t in tokenz)

        return (
            string
            .replace("Ċ", "\n")
            .replace("Ġ", " ")
            .replace("ĉ", "\t")
        )
