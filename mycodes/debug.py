import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
from dataset import load_dataset
from model import Encoder, AttentionDecoder, Seq2Seq

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    max_len = 50
    hidden_size = 512
    embed_size = 256
    n_layers = 2
    dropout = 0.5

    train_dataloader, valid_dataloader, test_dataloader, src_vocab, trg_vocab = load_dataset(batch_size=batch_size)
    sos_token_idx = src_vocab.get_stoi()['<bos>']
    eos_token_idx = src_vocab.get_stoi()['<eos>']
    src_vocab_size = len(src_vocab)
    trg_vocab_size = len(trg_vocab)
    
    encoder = Encoder(src_vocab_size, embed_size, hidden_size, n_layers=n_layers, dropout=dropout).to(device)
    decoder = AttentionDecoder(embed_size, hidden_size, trg_vocab_size, n_layers=1, dropout=0.5).to(device)
    seq2seq = Seq2Seq(encoder, decoder, sos_token_idx, eos_token_idx, max_len, device).to(device)
    print(seq2seq)

    

    print(src_vocab_size, trg_vocab_size)
    for src, trg in train_dataloader:
        src = src.to(device)
        trg = trg.to(device)
        print("Source shape:", src.shape, "Target shape:", trg.shape)
        
        encoder_output, (hidden, cell) = encoder(src)
        ## [src_len, batch_size] => [src_len, batch_size, hidden_size], ([num_layers * num_directions, batch_size, hidden_size], [num_layers * num_directions, batch_size, hidden_size])
        
        decoder_hidden = (hidden[:decoder.n_layers], cell[:decoder.n_layers])
        ## ([num_layers * num_directions, batch_size, hidden_size], [num_layers * num_directions, batch_size, hidden_size]) => ([num_layers, batch_size, hidden_size], [num_layers, batch_size, hidden_size])
        decoder_input = trg.data[0, :] # sos
        
        max_len = trg.size(0)
        decoder_outputs = []
        for t in range(1, max_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)
            top1 = decoder_output.data.max(1)[1]  # 확률이 가장 높은 단어의 인덱스
            decoder_input = Variable(top1).to(device)  # 이전 출력을 다음 입력으로 사용
        
        print("Decoder outputs length:", len(decoder_outputs))
        print("Decoder output shape:", decoder_outputs[0].shape)

        decoder_outputs = []
        decoder_attentions = []
        for t in range(1, max_len):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_output)
            decoder_outputs.append(decoder_output)
            decoder_attentions.append(decoder_attention)
            top1 = decoder_output.data.max(1)[1]  # 확률이 가장 높은 단어의 인덱스
            decoder_input = Variable(top1).cuda()  # 이전 출력을 다음 입력으로 사용
        
        print("Decoder outputs length:", len(decoder_outputs))
        print("Decoder output shape:", decoder_outputs[0].shape)
        print("Decoder attentions length:", len(decoder_attentions))
        print("Decoder attention shape:", decoder_attentions[0].shape)


        break