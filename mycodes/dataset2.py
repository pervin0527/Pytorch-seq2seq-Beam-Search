import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import re
import torch
import random
import unicodedata


from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
MAX_LENGTH = 50


def split_data(total_path, train_path, valid_path, test_path, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1):
    with open(total_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    random.shuffle(lines)

    total_size = len(lines)
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    test_size = total_size - train_size - valid_size

    train_data = lines[:train_size]
    valid_data = lines[train_size:train_size + valid_size]
    test_data = lines[train_size + valid_size:]

    with open(train_path, 'w', encoding='utf-8') as train_file:
        train_file.writelines(train_data)

    with open(valid_path, 'w', encoding='utf-8') as valid_file:
        valid_file.writelines(valid_data)

    with open(test_path, 'w', encoding='utf-8') as test_file:
        test_file.writelines(test_data)


"""
유니코드 문자열을 아스키 문자열로 변환. 이 과정을 통해 텍스트 데이터의 일관성을 높인다.
"""
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

"""
텍스트를 소문자로 변환하고, 불필요한 공백이나 문자가 아닌 문자를 제거해 모델의 학습 데이터 품질을 향상시킨다.
"""
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def tokenize_and_build_vocab(lang, pairs):
    if lang == 'eng':
        tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    elif lang == 'fra':
        tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
    else:
        raise ValueError(f"Unsupported language: {lang}")

    vocab = build_vocab_from_iterator(tokenizer(pair[0]) if lang == 'eng' else tokenizer(pair[1]) for pair in pairs)
    vocab.insert_token('<pad>', PAD_TOKEN)
    vocab.insert_token('<sos>', SOS_TOKEN)
    vocab.insert_token('<eos>', EOS_TOKEN)
    vocab.insert_token('<unk>', UNK_TOKEN)
    vocab.set_default_index(vocab['<unk>'])

    return vocab, tokenizer

def build_vocab(data_dir, src_lang='eng', trg_lang='fra', save_dir=None):
    lines = open(data_dir, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = filterPairs(pairs)

    src_vocab, src_tokenizer = tokenize_and_build_vocab(src_lang, pairs)
    trg_vocab, trg_tokenizer = tokenize_and_build_vocab(trg_lang, pairs)

    if save_dir:
        torch.save(src_vocab, os.path.join(save_dir, f'src_vocab_{src_lang}.pth'))
        torch.save(trg_vocab, os.path.join(save_dir, f'trg_vocab_{trg_lang}.pth'))

    return src_vocab, src_tokenizer, trg_vocab, trg_tokenizer


class TranslationDataset(Dataset):
    def __init__(self, data_dir, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, src_lang='eng'):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        
        lines = open(data_dir, encoding='utf-8').read().strip().split('\n')
        self.pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
        self.pairs = filterPairs(self.pairs)
        
        if src_lang == 'fra':
            self.pairs = [list(reversed(p)) for p in self.pairs]
            
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        input_sentence, output_sentence = self.pairs[idx]
        
        input_tokens = self.src_tokenizer(input_sentence)
        output_tokens = self.trg_tokenizer(output_sentence)
        
        input_tensor = [self.src_vocab['<sos>']] + [self.src_vocab[token] if token in self.src_vocab else self.src_vocab['<unk>'] for token in input_tokens] + [self.src_vocab['<eos>']]
        output_tensor = [self.trg_vocab['<sos>']] + [self.trg_vocab[token] if token in self.trg_vocab else self.trg_vocab['<unk>'] for token in output_tokens] + [self.trg_vocab['<eos>']]
        
        return torch.tensor(input_tensor, dtype=torch.long), torch.tensor(output_tensor, dtype=torch.long)
    

def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_sample, trg_sample in batch:
        src_batch.append(src_sample)
        trg_batch.append(trg_sample)

    src_batch = pad_sequence(src_batch, padding_value=PAD_TOKEN)
    trg_batch = pad_sequence(trg_batch, padding_value=PAD_TOKEN)

    return src_batch, trg_batch