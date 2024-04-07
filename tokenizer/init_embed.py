import os
import torch

from tqdm import tqdm
# from pathlib import Path
# from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def init_embeddings_average(
        old_tokenizer,
        new_tokenizer,
        old_embeddings,
        old_lm_head,
        new_embeddings,
        new_lm_head,
):
    old_vocab_size = old_tokenizer.vocab_size
    new_vocab_size = new_tokenizer.vocab_size

    for id in tqdm(range(old_vocab_size, new_vocab_size), desc='initing'):
        zh_token = new_tokenizer.decode([id])

        zh_token_old_ids = old_tokenizer(zh_token)["input_ids"]
        if len(zh_token_old_ids) == 0:
            print(f"WARNING: id = {id} zh_token = `{zh_token}`, cannot be tokenized by old tokenizer, using <unk> id")
            zh_token_old_ids = [0]  # unk
        zh_token_old_embeddings_avg = sum([old_embeddings[oid] for oid in zh_token_old_ids]) / len(zh_token_old_ids)
        zh_token_old_lm_head_avg = sum([old_lm_head[oid] for oid in zh_token_old_ids]) / len(zh_token_old_ids)
        new_embeddings[id] = zh_token_old_embeddings_avg
        new_lm_head[id] = zh_token_old_lm_head_avg

        
def main(
        old_tokenizer_path: str,
        new_tokenizer_path: str,
        old_model_path: str,
        new_model_path: str
):
    # load tokenizers
    old_tokenizer = AutoTokenizer.from_pretrained(old_tokenizer_path)
    new_tokenizer = AutoTokenizer.from_pretrained(new_tokenizer_path)
    new_vocab_size = len(new_tokenizer)  # __len__ = vocab_size + num_added_tokens

    # load old model
    old_model = AutoModelForCausalLM.from_pretrained(old_model_path)
    model_dict = {}
    for key in old_model.state_dict():
        model_dict[key] = old_model.state_dict()[key]
    del old_model

    # shape:
    #   old_embeddings: (vocab_size, d_model)
    #   old_lm_head:    (vocab_size, d_model)
    old_embeddings = model_dict["model.embed_tokens.weight"]
    old_lm_head = model_dict["lm_head.weight"]

    # create new embeddings and lm_head
    #   en: copy from old
    #   zh: init with zero
    new_embeddings = torch.zeros((new_vocab_size, old_embeddings.shape[1]), dtype=old_embeddings.dtype)
    new_lm_head = torch.zeros((new_vocab_size, old_lm_head.shape[1]), dtype=old_lm_head.dtype)
    new_embeddings[: old_embeddings.shape[0]] = old_embeddings.clone()
    new_lm_head[: old_lm_head.shape[0]] = old_lm_head.clone()

    init_embeddings_average(
        old_tokenizer,
        new_tokenizer,
        old_embeddings,
        old_lm_head,
        new_embeddings,
        new_lm_head,
    )

    model_dict["model.embed_tokens.weight"] = new_embeddings
    model_dict["lm_head.weight"] = new_lm_head

    save_tmp = new_model_path + '/tmp'
    if not os.path.exists(save_tmp):
        os.makedirs(save_tmp)

    # save draft
    print('Saving draft model...')
    torch.save(model_dict, f"{save_tmp}/pytorch_model.bin")
    config = AutoConfig.from_pretrained(f'{old_model_path}/config.json')
    config.vocab_size = new_vocab_size
    config.save_pretrained(save_tmp)
    new_tokenizer.save_pretrained(save_tmp)
    del model_dict

    # save model
    print('Saving final model...')
    model = AutoModelForCausalLM.from_pretrained(save_tmp, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(save_tmp)
    model.save_pretrained(new_model_path)
    tokenizer.save_pretrained(new_model_path)
    os.system(f"rm -rf {save_tmp}")

    print(f"Done! `new_vocab_size` = {new_vocab_size}, please update `config.json` manually.")


if __name__ == "__main__":
    old_model = '/path/to/old_tokenizer'
    new_tokenizer = '/path/to/new_tokenizer'
    new_model = '/path/to/new_model'

    main(
        old_tokenizer_path=old_model,
        new_tokenizer_path=new_tokenizer,
        old_model_path=old_model,
        new_model_path=new_model
    )