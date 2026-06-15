import os
import json
import torch
from torch.utils.data import Dataset
from typing import List, Optional
import sentencepiece as spm

from data.tokenizer import load_tokenizer, encode, BOS_ID, EOS_ID, PAD_ID


class BtwGPTDataset(Dataset):
    """
    Dataset for BtwGPT-1 training.

    Supports two data formats:
    1. Plain text files (.txt) - one document per line or paragraph-separated
    2. JSON/JSONL files (.json/.jsonl) - conversational format:
       {"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    All data is tokenized and packed into fixed-length sequences.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        max_seq_length: int = 4096,
        split: str = "train",
        system_prompt_path: Optional[str] = None,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.sp = load_tokenizer(tokenizer_path)
        self.samples: List[List[int]] = []
        self.system_prompt = self._load_system_prompt(system_prompt_path)

        self._load_data(data_path)

    def _load_system_prompt(self, path: Optional[str]) -> str:
        """Load system prompt from file, or use default BtwGPT personality."""
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return (
            "You are BtwGPT, a chill and honest friend. "
            "You talk casually, like texting a buddy. "
            "You're funny, direct, and you say what you think. "
            "You speak whatever language the user speaks to you."
        )

    def _load_data(self, data_path: str):
        """Load and tokenize data from the given path."""
        if os.path.isdir(data_path):
            files = [
                os.path.join(data_path, f)
                for f in os.listdir(data_path)
                if f.endswith((".txt", ".json", ".jsonl"))
            ]
        else:
            files = [data_path]

        all_tokens: List[int] = []

        for filepath in sorted(files):
            if filepath.endswith(".txt"):
                all_tokens.extend(self._process_text_file(filepath))
            elif filepath.endswith((".json", ".jsonl")):
                all_tokens.extend(self._process_json_file(filepath))

        self._pack_tokens(all_tokens)

    def _process_text_file(self, filepath: str) -> List[int]:
        """Process a plain text file."""
        tokens = []
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()
            if paragraph:
                ids = encode(self.sp, paragraph, add_bos=True, add_eos=True)
                tokens.extend(ids)
        return tokens

    def _process_json_file(self, filepath: str) -> List[int]:
        """Process a JSON/JSONL conversational file."""
        tokens = []
        with open(filepath, "r", encoding="utf-8") as f:
            if filepath.endswith(".jsonl"):
                entries = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
                entries = data if isinstance(data, list) else [data]

        for entry in entries:
            conv_tokens = self._format_conversation(entry)
            tokens.extend(conv_tokens)

        return tokens

    def _format_conversation(self, entry: dict) -> List[int]:
        """Format a conversation entry into tokens (Mixtral chat template)."""
        tokens = [BOS_ID]

        conversations = entry.get("conversations", entry.get("messages", []))

        has_system = any(
            msg.get("role", msg.get("from", "")) == "system" for msg in conversations
        )
        if not has_system and self.system_prompt:
            sys_text = f"[SYS] {self.system_prompt} [/SYS]\n"
            tokens.extend(self.sp.encode(sys_text, out_type=int))

        for msg in conversations:
            role = msg.get("role", msg.get("from", ""))
            content = msg.get("content", msg.get("value", ""))

            if role in ("user", "human"):
                text = f"[INST] {content} [/INST]"
            elif role in ("assistant", "gpt", "bot"):
                text = f" {content}"
            elif role == "system":
                text = f"[SYS] {content} [/SYS]\n"
            else:
                continue

            ids = self.sp.encode(text, out_type=int)
            tokens.extend(ids)

        tokens.append(EOS_ID)
        return tokens

    def _pack_tokens(self, all_tokens: List[int]):
        """Pack tokens into fixed-length sequences."""
        for i in range(0, len(all_tokens) - self.max_seq_length, self.max_seq_length):
            chunk = all_tokens[i : i + self.max_seq_length + 1]
            if len(chunk) == self.max_seq_length + 1:
                self.samples.append(chunk)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        tokens = self.samples[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
