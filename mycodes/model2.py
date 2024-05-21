import math
import torch
import random
import operator

from queue import PriorityQueue

from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super().__init__()
        self.input_size = input_size  # input_size: 입력 단어 집합의 크기
        self.hidden_size = hidden_size  # hidden_size: 인코더의 히든 상태 크기
        self.embed_size = embed_size  # embed_size: 단어 임베딩 벡터의 크기
        self.embed = nn.Embedding(input_size, embed_size)

        ## [num_layers * num_directions, batch_size, hidden_size]
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, src, hidden=None):
        # src: [src_len, batch_size]
        embedded = self.embed(src)  # embedded: [src_len, batch_size, embed_size]
        outputs, (hidden, cell) = self.lstm(embedded, hidden)
        # outputs: [src_len, batch_size, hidden_size * num_directions]
        # hidden: [num_layers * num_directions, batch_size, hidden_size]
        # cell: [num_layers * num_directions, batch_size, hidden_size]

        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:])
        # outputs: [src_len, batch_size, hidden_size]

        return outputs, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, n_layers=1, dropout=0.2):
        super().__init__()
        self.embed_size = embed_size  # embed_size: 단어 임베딩 벡터의 크기
        self.hidden_size = hidden_size  # hidden_size: 디코더의 히든 상태 크기
        self.output_size = output_size  # output_size: 출력 단어 집합의 크기
        self.n_layers = n_layers  # n_layers: LSTM 레이어의 개수
        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, last_hidden):
        # input: [batch_size]
        # last_hidden: (hidden: [num_layers, batch_size, hidden_size], cell: [num_layers, batch_size, hidden_size])
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # embedded: [1, batch_size, embed_size]
        embedded = self.dropout(embedded)

        # Pass through the LSTM layer
        output, hidden = self.lstm(embedded, last_hidden)
        # output: [1, batch_size, hidden_size]
        # hidden: (hidden: [num_layers, batch_size, hidden_size], cell: [num_layers, batch_size, hidden_size])

        output = output.squeeze(0)  # output: [batch_size, hidden_size]
        output = self.out(output)  # output: [batch_size, output_size]
        output = F.log_softmax(output, dim=1)
        return output, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size  # hidden_size: 히든 상태의 크기
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [src_len, batch_size, hidden_size]
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)  # h: [batch_size, src_len, hidden_size]
        encoder_outputs = encoder_outputs.transpose(0, 1)  # encoder_outputs: [batch_size, src_len, hidden_size]
        attn_energies = self.score(h, encoder_outputs)  # attn_energies: [batch_size, src_len]
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # [batch_size, 1, src_len]

    def score(self, hidden, encoder_outputs):
        # hidden: [batch_size, src_len, hidden_size]
        # encoder_outputs: [batch_size, src_len, hidden_size]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))  # energy: [batch_size, src_len, hidden_size]
        energy = energy.transpose(1, 2)  # energy: [batch_size, hidden_size, src_len]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # v: [batch_size, 1, hidden_size]
        energy = torch.bmm(v, energy)  # energy: [batch_size, 1, src_len]
        return energy.squeeze(1)  # [batch_size, src_len]


class AttentionDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, n_layers=1, dropout=0.0):
        super().__init__()
        self.embed_size = embed_size  # embed_size: 단어 임베딩 벡터의 크기
        self.hidden_size = hidden_size  # hidden_size: 디코더의 히든 상태 크기
        self.output_size = output_size  # output_size: 출력 단어 집합의 크기
        self.n_layers = n_layers  # n_layers: LSTM 레이어의 개수
        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # input: [batch_size]
        # last_hidden: (hidden: [num_layers, batch_size, hidden_size], cell: [num_layers, batch_size, hidden_size])
        # encoder_outputs: [src_len, batch_size, hidden_size]
        # Get the embedding of the current input word (last output word)
        embedded = self.embed(input).unsqueeze(0)  # embedded: [1, batch_size, embed_size]
        embedded = self.dropout(embedded)
        
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[0][-1], encoder_outputs)  # attn_weights: [batch_size, 1, src_len]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # context: [batch_size, 1, hidden_size]
        context = context.transpose(0, 1)  # context: [1, batch_size, hidden_size]
        
        # Combine embedded input word and attended context, run through LSTM
        rnn_input = torch.cat([embedded, context], 2)  # rnn_input: [1, batch_size, embed_size + hidden_size]
        output, (hidden, cell) = self.lstm(rnn_input, last_hidden)
        # output: [1, batch_size, hidden_size]
        # hidden: [num_layers, batch_size, hidden_size]
        # cell: [num_layers, batch_size, hidden_size]
        
        output = output.squeeze(0)  # output: [batch_size, hidden_size]
        context = context.squeeze(0)  # context: [batch_size, hidden_size]
        output = self.out(torch.cat([output, context], 1))  # output: [batch_size, output_size]
        output = F.log_softmax(output, dim=1)
        return output, (hidden, cell), attn_weights  # output: [batch_size, output_size], (hidden, cell): ([num_layers, batch_size, hidden_size], [num_layers, batch_size, hidden_size]), attn_weights: [batch_size, 1, src_len]
    

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, sos_token, eos_token, max_len, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_token = sos_token  # 시작 토큰 (Start of Sentence)
        self.eos_token = eos_token  # 종료 토큰 (End of Sentence)
        self.max_len = max_len  # 최대 시퀀스 길이

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = torch.zeros(max_len, batch_size, vocab_size).to(self.device)

        encoder_output, (hidden, cell) = self.encoder(src)  # src: [src_len, batch_size], encoder_output: [src_len, batch_size, hidden_size], hidden: [num_layers * num_directions, batch_size, hidden_size]
        decoder_hidden = (hidden[:self.decoder.n_layers], cell[:self.decoder.n_layers])
        decoder_input = trg.data[0, :] # sos

        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(decoder_input, decoder_hidden, encoder_output)  # output: [batch_size, vocab_size], hidden: [num_layers, batch_size, hidden_size], attn_weights: [batch_size, 1, src_len]
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]  # top1: [batch_size]
            output = trg.data[t] if is_teacher else top1

        return outputs

    def decode(self, src, trg, method='beam-search'):
        encoder_output, (hidden, cell) = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        cell = cell[:self.decoder.n_layers]
        if method == 'beam-search':
            return self.beam_decode(trg, (hidden, cell), encoder_output)
        else:
            return self.greedy_decode(trg, (hidden, cell), encoder_output)
        
    def greedy_decode(self, trg, hidden, encoder_output):
        max_len = self.max_len
        batch_size = trg.size(1)
        vocab_size = self.decoder.output_size
        
        decoder_input = trg.data[0, :]  # sos
        outputs = torch.zeros(max_len, batch_size, vocab_size).to(self.device)

        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(decoder_input, hidden, encoder_output)
            outputs[t] = output
            top1 = output.data.max(1)[1]
            decoder_input = top1

        return outputs

    def beam_decode(self, trg, hidden, encoder_output, beam_width=3):
        max_len = self.max_len
        batch_size = trg.size(1)
        vocab_size = self.decoder.output_size

        # Initialization
        sequences = []
        decoder_input = trg.data[0, :].unsqueeze(1)  # sos, [batch_size] -> [batch_size, 1]
        for batch_idx in range(batch_size):
            sequences.append((0, [decoder_input[batch_idx]], (hidden[0][:, batch_idx:batch_idx+1, :], hidden[1][:, batch_idx:batch_idx+1, :])))

        final_sequences = []

        for _ in range(max_len):
            all_candidates = []

            for score, seq, hidden in sequences:
                decoder_input = seq[-1]  # [1, 1]

                output, hidden, attn_weights = self.decoder(decoder_input, hidden, encoder_output[:, seq[-1].shape[0]-1, :].unsqueeze(1))
                topk_scores, topk_indices = output.data.topk(beam_width, dim=1)

                for i in range(beam_width):
                    candidate_score = score - topk_scores[0, i].item()
                    candidate_seq = seq + [topk_indices[0, i].unsqueeze(0)]
                    candidate_hidden = hidden
                    candidate = (candidate_score, candidate_seq, candidate_hidden)
                    all_candidates.append(candidate)

            # Select beam_width candidates
            ordered = sorted(all_candidates, key=lambda tup: tup[0])
            sequences = []

            for i in range(min(beam_width, len(ordered))):
                sequences.append(ordered[i])
                if ordered[i][1][-1].item() == self.eos_token:
                    final_sequences.append(ordered[i])

            if len(final_sequences) >= beam_width:
                break

        # Select the best sequence
        if len(final_sequences) == 0:
            final_sequences = [sequences[i] for i in range(min(beam_width, len(sequences)))]

        best_sequence = sorted(final_sequences, key=lambda tup: tup[0])[0]
        output_sequence = torch.stack([t for t in best_sequence[1][1:]], dim=0)

        return output_sequence

