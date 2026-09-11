import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from lsx import LM

CORPUS = [
    "the thesis is that a particle has a definite position and momentum",
    "the antithesis is that measurement disturbs what it measures",
    "the synthesis is that position and momentum are complementary descriptions",
    "a child is embedded in its impulses; later the impulses become objects it can hold",
    "the muppets take manhattan while star wars unfolds two hundred years earlier",
] * 4


@pytest.fixture(scope="session")
def tiny_lm() -> LM:
    torch.manual_seed(0)
    tk = Tokenizer(models.BPE(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(CORPUS, trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]", "[PAD]", "[EOS]"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    cfg = Qwen2Config(vocab_size=tok.vocab_size + 8, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    return LM(Qwen2ForCausalLM(cfg), tok)
