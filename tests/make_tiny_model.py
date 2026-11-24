"""Builds a tiny random causal LM + tokenizer on disk, for CPU tests of the RL loop (no download).

    python tests/make_tiny_model.py /tmp/tiny-lm
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def main(out):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from cpbench import CorruptPoliceEnv, GameConfig
    from cpbench.agents.llm import build_prompt
    env = CorruptPoliceEnv(GameConfig.mini(seed=0))
    obs = env.reset()
    corpus = [build_prompt(o) for o in obs.values()] + ['{"type": "investigate", "station": 12, "thought": "x"}'] * 50
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=2000, special_tokens=["<unk>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                 unk_token="<unk>")
    hf.chat_template = CHAT_TEMPLATE
    hf.save_pretrained(out)
    cfg = Qwen3Config(vocab_size=len(hf), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=8192,
                      eos_token_id=hf.eos_token_id, pad_token_id=hf.pad_token_id, tie_word_embeddings=True)
    Qwen3ForCausalLM(cfg).save_pretrained(out)
    print("tiny model written to", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tiny-lm")
