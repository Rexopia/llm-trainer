import os
import sentencepiece as spm

from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from tqdm import tqdm


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    return '\u4e00' <= uchar <= '\u9fa5'

def is_chinese_string(string):
    """判断是否全为汉字"""
    return all(is_chinese(c) for c in string)

def main(
    base_tokenizer_dir, 
    domain_sp_model_file
    output_sp_dir = './merged_tokenizer_sp'
    output_hf_dir = './merged_tokenizer_hf'
):
    # load
    llama_tokenizer = LlamaTokenizer.from_pretrained(base_tokenizer_dir)
    chinese_sp_model = spm.SentencePieceProcessor()
    chinese_sp_model.Load(domain_sp_model_file)

    llama_spm = sp_pb2_model.ModelProto()
    llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
    chinese_spm = sp_pb2_model.ModelProto()
    chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())

    # print number of tokens
    print(len(llama_tokenizer), len(chinese_sp_model))
    print(llama_tokenizer.all_special_tokens)
    print(llama_tokenizer.all_special_ids)
    print(llama_tokenizer.special_tokens_map)
    
    # Add Chinese tokens to LLaMA tokenizer
    llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)

    print(len(llama_spm_tokens_set))
    print(f"Before:{len(llama_spm_tokens_set)}")
    added_set = set()
    for p in chinese_spm.pieces:
        piece = p.piece
        if piece not in llama_spm_tokens_set and is_chinese_string(piece):
            # print('picec', piece)
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            llama_spm.pieces.append(new_p)
            added_set.add(piece)
    print(f"[add domain tokens]New model pieces: {len(llama_spm.pieces)}")

    # Save
    os.makedirs(output_sp_dir, exist_ok=True)
    with open(output_sp_dir + '/chinese_llama.model', 'wb') as f:
        f.write(llama_spm.SerializeToString())
    tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + '/chinese_llama.model')

    tokenizer.save_pretrained(output_hf_dir)
    print(f"Chinese-LLaMA tokenizer has been saved to {output_hf_dir}")
    
    

    
if __name__ == '__main__':
    
    base_tokenizer_dir = '/path/to/base_tokenizer'
    domain_sp_model_file = '/path/to/domain_sp/tokenizer.model'
    output_sp_dir = './merged_tokenizer_sp'
    output_hf_dir = './merged_tokenizer_hf'
    
    llama_tokenizer = LlamaTokenizer.from_pretrained(base_tokenizer_dir)
    chinese_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
    print(chinese_llama_tokenizer.all_special_tokens)
    print(chinese_llama_tokenizer.all_special_ids)
    print(chinese_llama_tokenizer.special_tokens_map)
    print('old len:', len(llama_tokenizer), ' new len:', len(chinese_llama_tokenizer))
    text = '''this is a test, hello world. thisisatesthelloworld, 
    慕容复来到河边，姑苏慕容氏在外面丢了人。
    1号店一周岁了，我们一古脑儿买了10斤零食。
    巴塞罗那足球俱乐部简称巴萨（Barça），是一家位于西班牙加泰罗尼亚巴塞罗那的足球俱乐部，于1899年由瑞士企业家胡安·甘伯所创立，世界球坛顶级足球俱乐部之一。俱乐部主场可容纳接近十万名观众，是全欧洲最大及世界第二大的足球场。
    白日依山尽，黄河入海流。欲穷千里目，更上一层楼。'''
    print(f"Test text:\n{text}")
    print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
    print(f"Tokenized by Chinese-LLaMA tokenizer:{chinese_llama_tokenizer.tokenize(text)}")