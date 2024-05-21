import re
import spacy
import torch
import torchtext.transforms as transforms

from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def yield_tokens(data_iter, tokenizer, index):
    for data_sample in data_iter:
        yield tokenizer(data_sample[index])

def load_dataset(batch_size):
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
    url = re.compile('(<url>.*</url>)')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

    de_tokenizer = get_tokenizer(tokenize_de)
    en_tokenizer = get_tokenizer(tokenize_en)

    train_iter, val_iter, test_iter = Multi30k(split=('train', 'valid', 'test'))

    de_vocab = build_vocab_from_iterator(yield_tokens(train_iter, de_tokenizer, index=0), specials=["<sos>", "<eos>", "<unk>", "<pad>"], min_freq=2)
    en_vocab = build_vocab_from_iterator(yield_tokens(train_iter, en_tokenizer, index=1), specials=["<sos>", "<eos>", "<unk>", "<pad>"], max_size=10000)

    de_vocab.set_default_index(de_vocab["<unk>"])
    en_vocab.set_default_index(en_vocab["<unk>"])

    text_transform_de = transforms.Sequential(
        transforms.VocabTransform(de_vocab),
        transforms.AddToken(token="<sos>", begin=True),
        transforms.AddToken(token="<eos>", end=True)
    )

    text_transform_en = transforms.Sequential(
        transforms.VocabTransform(en_vocab),
        transforms.AddToken(token="<sos>", begin=True),
        transforms.AddToken(token="<eos>", end=True)
    )

    def collate_fn(batch):
        de_batch, en_batch = [], []
        for (de_item, en_item) in batch:
            de_batch.append(text_transform_de(de_tokenizer(de_item)))
            en_batch.append(text_transform_en(en_tokenizer(en_item)))
        de_batch = pad_sequence(de_batch, padding_value=de_vocab["<pad>"])
        en_batch = pad_sequence(en_batch, padding_value=en_vocab["<pad>"])
        return de_batch, en_batch

    train_dataloader = DataLoader(list(train_iter), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(list(val_iter), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(list(test_iter), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader, de_vocab, en_vocab