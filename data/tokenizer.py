import os
import sentencepiece as spm
from typing import Optional, List


SPECIAL_TOKENS = ["[INST]", "[/INST]", "[SYS]", "[/SYS]"]

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def train_tokenizer(
    input_files: str,
    model_prefix: str = "tokenizer/btwgpt",
    vocab_size: int = 32000,
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    num_threads: int = 8,
):
    """
    Train a SentencePiece BPE tokenizer on the provided text data.

    Args:
        input_files: Comma-separated paths to text files for training.
        model_prefix: Output path prefix for the tokenizer model.
        vocab_size: Target vocabulary size.
        model_type: Tokenizer type ('bpe' or 'unigram').
        character_coverage: Character coverage for training (high for French).
        num_threads: Number of threads for training.
    """
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=input_files,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        num_threads=num_threads,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        user_defined_symbols=SPECIAL_TOKENS,
        normalization_rule_name="nfkc",
    )
    print(f"Tokenizer trained and saved to {model_prefix}.model")


def load_tokenizer(model_path: str = "tokenizer/btwgpt.model") -> spm.SentencePieceProcessor:
    """Load a trained SentencePiece tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def encode(
    sp: spm.SentencePieceProcessor,
    text: str,
    add_bos: bool = True,
    add_eos: bool = True,
) -> List[int]:
    """Encode text to token IDs."""
    ids = sp.encode(text, out_type=int)
    if add_bos:
        ids = [BOS_ID] + ids
    if add_eos:
        ids = ids + [EOS_ID]
    return ids


def decode(sp: spm.SentencePieceProcessor, ids: List[int]) -> str:
    """Decode token IDs back to text."""
    return sp.decode(ids)
