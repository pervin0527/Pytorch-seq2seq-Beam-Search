import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torch.utils.tensorboard import SummaryWriter

from model import Encoder, AttentionDecoder, Seq2Seq
from dataset import TranslationDataset, split_data, build_vocab, collate_fn, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN

def train(model, dataloader, optimizer, criterion, vocab_size, grad_clip, device, epoch, writer):
    model.train()
    total_loss = 0
    num_batches = 0
    for src, trg in tqdm(dataloader, desc='Train', leave=False):
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg)
        output = output[1:].view(-1, vocab_size)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    train_loss = total_loss / num_batches
    train_perplexity = math.exp(train_loss)

    writer.add_scalar('Train/Loss', train_loss, epoch)
    writer.add_scalar('Train/Perplexity', train_perplexity, epoch)

    return train_loss, train_perplexity

def valid(model, dataloader, criterion, vocab_size, trg_vocab, device, epoch, writer):
    model.eval()
    total_loss = 0
    num_batches = 0
    decoded_batch_list = []
    with torch.no_grad():
        for src, trg in tqdm(dataloader, desc='Valid', leave=False):
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg)
            output = output[1:].view(-1, vocab_size)
            loss = criterion(output, trg[1:].contiguous().view(-1))
            total_loss += loss.item()
            num_batches += 1
            decoded_batch = model.decode(src, trg, method='beam-search')
            decoded_batch_list.append(decoded_batch)
    
    valid_loss = total_loss / num_batches
    valid_perplexity = math.exp(valid_loss)

    writer.add_scalar('Valid/Loss', valid_loss, epoch)
    writer.add_scalar('Valid/Perplexity', valid_perplexity, epoch)

    return valid_loss, valid_perplexity

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = './runs'
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    data_dir = '/home/pervinco/Desktop/en-fr/data'

    total_data_dir = f'{data_dir}/eng-fra.txt'
    train_data_dir = f'{data_dir}/train.txt'
    valid_data_dir = f'{data_dir}/valid.txt'
    test_data_dir = f'{data_dir}/test.txt'
    src_lang, trg_lang = 'eng', 'fra'

    if not os.path.exists(train_data_dir) or not os.path.exists(valid_data_dir):
        split_data(total_data_dir, train_data_dir, valid_data_dir, test_data_dir)

    if not os.path.exists(f'{data_dir}/src_vocab_{src_lang}.pth') and not os.path.exists(f'{data_dir}/trg_vocab_{trg_lang}.pth'):
        src_vocab, src_tokenizer, trg_vocab, trg_tokenizer = build_vocab(total_data_dir, src_lang, trg_lang, data_dir)
    else:
        src_vocab = torch.load(f'{data_dir}/src_vocab_{src_lang}.pth')
        trg_vocab = torch.load(f'{data_dir}/trg_vocab_{trg_lang}.pth')

        if src_lang == 'eng':
            src_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
            trg_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
        elif src_lang == 'fra':
            src_tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
            trg_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

    src_vocab_size = len(src_vocab)
    trg_vocab_size = len(trg_vocab)
    print(src_vocab_size, trg_vocab_size)
    
    epochs = 100
    batch_size = 80
    learning_rate = 0.0001
    grad_clip = 1
    max_len = 50
    
    embed_dim = 620
    hidden_dim = 1000
    encoder_layers = 1
    encoder_dropout = 0.0
    decoder_dropout = 0.0

    train_dataset = TranslationDataset(train_data_dir, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, src_lang)
    valid_dataset = TranslationDataset(valid_data_dir, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, src_lang)
    test_dataset = TranslationDataset(test_data_dir, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, src_lang)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, n_layers=encoder_layers, dropout=encoder_dropout).to(device)
    decoder = AttentionDecoder(embed_dim, hidden_dim, trg_vocab_size, n_layers=1).to(device)
    seq2seq = Seq2Seq(encoder, decoder, SOS_TOKEN, EOS_TOKEN, max_len, device).to(device)
    # optimizer = torch.optim.Adam(seq2seq.parameters(), lr=learning_rate)

    optimizer = torch.optim.Adadelta(seq2seq.parameters(), rho=0.95, eps=1e-6)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    
    best_val_loss = float('inf')
    for epoch in range(1, epochs+1):
        print(f'\nEpoch : {epoch}')
        train_loss, train_perplexity = train(seq2seq, train_dataloader, optimizer, criterion, trg_vocab_size, grad_clip, device, epoch, writer)
        valid_loss, valid_perplexity = valid(seq2seq, valid_dataloader, criterion, trg_vocab_size, trg_vocab, device, epoch, writer)
        print(f'Train Loss : {train_loss:.4f}, Train Perplexity : {train_perplexity:.4f}')
        print(f'Valid Loss : {valid_loss:.4f}, Valid Perplexity : {valid_perplexity:.4f}')
        
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(seq2seq.state_dict(), os.path.join(save_dir, 'best.pth'))
    
    torch.save(seq2seq.state_dict(), os.path.join(save_dir, 'last.pth'))
    test_loss, test_perplexity = valid(seq2seq, test_dataloader, criterion, trg_vocab_size, trg_vocab, device, epoch, writer)
    writer.close()
