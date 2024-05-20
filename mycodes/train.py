import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Multi30kDataset
from model import Encoder, AttentionDecoder, Seq2Seq

def train(model, dataloader, optimizer, criterion, vocab_size, grad_clip, device, epoch, writer):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch in tqdm(dataloader, desc='Train', leave=False):
        src = batch["en_ids"].to(device)
        trg = batch["de_ids"].to(device)
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
    for batch in tqdm(dataloader, desc='Valid', leave=False):
        src = batch["en_ids"].to(device)
        trg = batch["de_ids"].to(device)
        with torch.no_grad():
            output = model(src, trg)
            output = output[1:].view(-1, vocab_size)
            loss = criterion(output, trg[1:].contiguous().view(-1))
            total_loss += loss.item()
            num_batches += 1
            decoded_batch = model.decode(src, trg, method='beam-search')
            decoded_batch_list.append(decoded_batch)
    for sentence_index in decoded_batch_list[0]:
        decode_text_arr = [trg_vocab.get_itos()[i] for i in sentence_index[0]]
        decode_sentence = " ".join(decode_text_arr[1:-1])
        print(f"Pred target : {decode_sentence}")
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
    
    epochs = 100
    batch_size = 32
    learning_rate = 0.0001
    grad_clip = 10.0
    max_len = 50
    hidden_size = 512
    embed_size = 256
    encoder_layers = 2
    encoder_dropout = 0.5
    decoder_dropout = 0.2
    dataset = Multi30kDataset(max_length=max_len, lower=True, min_freq=2)
    src_vocab = dataset.en_vocab
    trg_vocab = dataset.de_vocab
    src_vocab_size = len(src_vocab)
    trg_vocab_size = len(trg_vocab)
    sos_token_idx = src_vocab.get_stoi()['<sos>']
    eos_token_idx = src_vocab.get_stoi()['<eos>']
    pad_token_idx = src_vocab.get_stoi()['<pad>']
    train_dataset, valid_dataset, test_dataset = dataset.get_datasets()
    collate_fn = dataset.get_collate_fn(pad_token_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    encoder = Encoder(src_vocab_size, embed_size, hidden_size, n_layers=encoder_layers, dropout=encoder_dropout).to(device)
    decoder = AttentionDecoder(embed_size, hidden_size, trg_vocab_size, n_layers=1).to(device)
    seq2seq = Seq2Seq(encoder, decoder, sos_token_idx, eos_token_idx, max_len, device).to(device)
    optimizer = torch.optim.Adam(seq2seq.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_idx)
    
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