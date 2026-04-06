import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

import transformers

transformers.logging.set_verbosity_error()

"""
   Modified based on  https://github.com/D3Mlab/rir/blob/main/prefernce_matching/LM.py
"""


class BERT_model:
    _bert_name: str
    _tokenizer: AutoTokenizer
    _bert_model: torch.nn.Module
    _device: torch.device

    def __init__(self, bert_name: str, tokenizer_name: str, from_pt: bool = True):
        """
        :param bert_name: name or address of language prefernce_matching
        :param tokenizer_name: name or address of the tokenizer
        """
        self._bert_name = bert_name
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._bert_model = self._create_model(bert_name, from_pt)
        self._bert_model.to(self._device)
        self._bert_model.eval()

    def embed(self, texts: list[str], strategy=None, bs=48, verbose=0) -> np.ndarray:
        """
        Embed the batch of texts.

        :param texts: list of strings to be embedded
        :param strategy: Defaults to None.
        :param bs: Defaults to 48.
        :param verbose: Defaults to 0.
        :return: embeddings of texts
        """
        embeddings = []

        for start_index in range(0, len(texts), bs):
            batch_texts = texts[start_index:start_index + bs]
            batch = self._tokenizer(
                batch_texts,
                max_length=512,
                add_special_tokens=True,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )

            batch = {key: value.to(self._device) for key, value in batch.items()}

            with torch.inference_mode():
                outputs = self._bert_model(**batch)

            last_hidden_state = outputs.last_hidden_state
            attention_mask = batch['attention_mask'].unsqueeze(-1).type_as(last_hidden_state)
            summed = (last_hidden_state * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts
            embeddings.append(pooled.detach().cpu())

        return torch.cat(embeddings, dim=0).numpy()

    def get_tensor_embedding(self, query: str) -> torch.Tensor:
        """
        Get a tensor embedding of a string.

        :param query: string to be embedded
        :return: tensor embedding of query
        """
        query_embedding = self.embed([query])
        query_embedding = torch.tensor(query_embedding).to(self._device)
        query_embedding = query_embedding.squeeze(0)

        return query_embedding

    def _create_model(self, bert_name: str, from_pt: bool = True) -> torch.nn.Module:
        """
        Load a Hugging Face encoder model with PyTorch.

        The from_pt flag is kept for compatibility with the old API.
        For Hugging Face checkpoints it should remain True.
        """
        if from_pt:
            model = AutoModel.from_pretrained(
                bert_name,
                trust_remote_code=True
            )
        else:
            model = AutoModel.from_pretrained(
                bert_name,
                from_tf=True,
                trust_remote_code=True
            )
        return model
