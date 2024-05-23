import math
import torch
import random
import operator

from torch import nn
from torch.nn import functional as F
from queue import PriorityQueue

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
UNK_TOKEN = 3
MAX_LENGTH = 50

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.001)

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim, n_layers=1, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=True)
        self.apply(weight_init)  # 초기화 적용

    def forward(self, src, hidden=None):
        # src: (seq_len, batch_size)
        embedded = self.embed(src)
        # embedded: (seq_len, batch_size, embed_dim)
        
        outputs, (hidden, cell) = self.lstm(embedded, hidden)
        # outputs: (seq_len, batch_size, hidden_dim * 2)
        
        # 순방향과 역방향의 출력을 합침
        outputs = outputs[:, :, :self.hidden_dim] + outputs[:, :, self.hidden_dim:]
        # outputs: (seq_len, batch_size, hidden_dim)
        
        return outputs, (hidden, cell)

class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, trg_vocab_size, n_layers=1, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(trg_vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_dim, trg_vocab_size)
        self.trg_vocab_size = trg_vocab_size
        self.apply(weight_init)  # 초기화 적용

    def forward(self, input, last_hidden):
        # input: (batch_size), last_hidden: (n_layers, batch_size, hidden_dim)
        embedded = self.embed(input).unsqueeze(0)
        embedded = self.dropout(embedded)
        # embedded: (1, batch_size, embed_dim)
        
        output, hidden = self.lstm(embedded, last_hidden)
        # output: (1, batch_size, hidden_dim)
        
        output = output.squeeze(0)
        # output: (batch_size, hidden_dim)
        
        output = self.out(output)
        # output: (batch_size, trg_vocab_size)
        
        output = F.log_softmax(output, dim=1)
        # output: (batch_size, trg_vocab_size)
        
        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
        self.apply(weight_init)  # 초기화 적용

    def forward(self, hidden, encoder_outputs):
        # hidden: (batch_size, hidden_dim)
        # encoder_outputs: (seq_len, batch_size, hidden_dim * 2)
        
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        # h: (batch_size, seq_len, hidden_dim)
        
        encoder_outputs = encoder_outputs.transpose(0, 1)
        # encoder_outputs: (batch_size, seq_len, hidden_dim * 2)
        
        attn_energies = self.score(h, encoder_outputs)
        # attn_energies: (batch_size, seq_len)
        
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
        # return: (batch_size, 1, seq_len)

    def score(self, hidden, encoder_outputs):
        # hidden: (batch_size, seq_len, hidden_dim)
        # encoder_outputs: (batch_size, seq_len, hidden_dim * 2)
        
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        # energy: (batch_size, seq_len, hidden_dim)
        
        energy = energy.transpose(1, 2)
        # energy: (batch_size, hidden_dim, seq_len)
        
        v = self.v.unsqueeze(0).expand(encoder_outputs.size(0), -1).unsqueeze(1)
        # v: (batch_size, 1, hidden_dim)
        
        energy = torch.bmm(v, energy)
        # energy: (batch_size, 1, seq_len)
        
        return energy.squeeze(1)
        # return: (batch_size, seq_len)
        

class AttentionDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, trg_vocab_size, n_layers=1, dropout=0.0):
        super().__init__()
        self.n_layers = n_layers
        self.embed = nn.Embedding(trg_vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + embed_dim, hidden_dim, n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_dim * 2, trg_vocab_size)
        self.trg_vocab_size = trg_vocab_size
        self.apply(weight_init)  # 초기화 적용

    def forward(self, input, last_hidden, encoder_outputs):
        # input: (batch_size)
        # last_hidden: ((n_layers, batch_size, hidden_dim), (n_layers, batch_size, hidden_dim))
        # encoder_outputs: (seq_len, batch_size, hidden_dim * 2)
        
        embedded = self.embed(input).unsqueeze(0)
        # embedded: (1, batch_size, embed_dim)
        
        embedded = self.dropout(embedded)
        # embedded: (1, batch_size, embed_dim)
        
        attn_weights = self.attention(last_hidden[0][-1], encoder_outputs)
        # attn_weights: (batch_size, 1, seq_len)
        
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # context: (batch_size, 1, hidden_dim * 2)
        
        context = context.transpose(0, 1)
        # context: (1, batch_size, hidden_dim * 2)
        
        rnn_input = torch.cat([embedded, context], 2)
        # rnn_input: (1, batch_size, hidden_dim + embed_dim)
        
        output, (hidden, cell) = self.lstm(rnn_input, last_hidden)
        # output: (1, batch_size, hidden_dim)
        # hidden: (n_layers, batch_size, hidden_dim)
        # cell: (n_layers, batch_size, hidden_dim)
        
        output = output.squeeze(0)
        # output: (batch_size, hidden_dim)
        
        context = context.squeeze(0)
        # context: (batch_size, hidden_dim * 2)
        
        output = self.out(torch.cat([output, context], 1))
        # output: (batch_size, trg_vocab_size)
        
        output = F.log_softmax(output, dim=1)
        # output: (batch_size, trg_vocab_size)
        
        return output, (hidden, cell), attn_weights
        # return: (batch_size, trg_vocab_size), ((n_layers, batch_size, hidden_dim), (n_layers, batch_size, hidden_dim)), (batch_size, 1, seq_len)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, sos_token, eos_token, max_len, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.max_len = max_len

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: (seq_len, batch_size)
        # trg: (seq_len, batch_size)
        
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.trg_vocab_size
        
        # outputs: (max_len, batch_size, vocab_size)
        outputs = torch.zeros(max_len, batch_size, vocab_size).to(self.device)

        encoder_output, (hidden, cell) = self.encoder(src)
        # encoder_output: (seq_len, batch_size, hidden_dim * 2)
        # hidden, cell: (n_layers * 2, batch_size, hidden_dim)
        
        decoder_hidden = (hidden[:self.decoder.n_layers], cell[:self.decoder.n_layers])
        # decoder_hidden: (n_layers, batch_size, hidden_dim)
        
        decoder_input = trg.data[0, :]
        # decoder_input: (batch_size)

        for t in range(1, max_len):
            output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)
            # output: (batch_size, vocab_size)
            # decoder_hidden: (n_layers, batch_size, hidden_dim)
            
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            # top1: (batch_size)
            
            decoder_input = trg[t] if is_teacher else top1
            # decoder_input: (batch_size)

        return outputs
        # outputs: (max_len, batch_size, vocab_size)

    def decode(self, src, trg, method='beam-search'):
        encoder_output, (hidden, cell) = self.encoder(src)
        # encoder_output: (seq_len, batch_size, hidden_dim * 2)
        # hidden, cell: (n_layers * 2, batch_size, hidden_dim)
        
        hidden = hidden[:self.decoder.n_layers]
        cell = cell[:self.decoder.n_layers]
        # hidden, cell: (n_layers, batch_size, hidden_dim)
        
        if method == 'beam-search':
            return self.beam_decode(trg, (hidden, cell), encoder_output)
        else:
            return self.greedy_decode(trg, (hidden, cell), encoder_output)
        
    def greedy_decode(self, trg, decoder_hidden, encoder_outputs):
        # trg: (seq_len, batch_size)
        seq_len, batch_size = trg.size()
        
        decoded_batch = torch.zeros((batch_size, seq_len), dtype=torch.long).to(self.device)
        # decoded_batch: (batch_size, seq_len)
        
        decoder_input = trg.data[0, :]
        # decoder_input: (batch_size)

        for t in range(seq_len):
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # decoder_output: (batch_size, vocab_size)
            # decoder_hidden: (n_layers, batch_size, hidden_dim)
            
            top1 = decoder_output.argmax(1)
            # top1: (batch_size)
            
            decoded_batch[:, t] = top1
            decoder_input = top1
            # decoder_input: (batch_size)

        return decoded_batch
        # decoded_batch: (batch_size, seq_len)

    def beam_decode(self, target_tensor, decoder_hiddens, encoder_outputs=None):
        # target_tensor: (seq_len, batch_size)
        seq_len, batch_size = target_tensor.size()
        beam_width = 10
        topk = 1
        decoded_batch = []

        for idx in range(batch_size):
            decoder_hidden = (decoder_hiddens[0][:, idx, :].unsqueeze(1), decoder_hiddens[1][:, idx, :].unsqueeze(1))
            # decoder_hidden: (n_layers, 1, hidden_dim)
            
            encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)
            # encoder_output: (seq_len, 1, hidden_dim * 2)

            decoder_input = torch.LongTensor([self.sos_token]).to(self.device)
            # decoder_input: (1)

            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            nodes.put((-node.eval(), node))
            qsize = 1

            while True:
                if qsize > 2000:
                    break

                score, n = nodes.get()
                decoder_input = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == self.eos_token and n.prevNode is not None:
                    endnodes.append((score, n))
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)
                # decoder_output: (1, vocab_size)
                # decoder_hidden: (n_layers, 1, hidden_dim)

                log_prob, indexes = torch.topk(decoder_output, beam_width)
                # log_prob: (1, beam_width)
                # indexes: (1, beam_width)
                
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(-1)
                    # decoded_t: (1)
                    
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    qsize += len(nextnodes) - 1

            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                while n.prevNode is not None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)
            # decoded_batch: (batch_size, beam_width)

        return decoded_batch
        # return: List of decoded sequences
    
class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

    def __lt__(self, other):
        return self.leng < other.leng

    def __gt__(self, other):
        return self.leng > other.leng