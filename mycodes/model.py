import math
import torch
import random
import operator

from torch import nn
from torch.nn import functional as F
from queue import PriorityQueue


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
        
    def greedy_decode(self, trg, decoder_hidden, encoder_outputs):
        '''
        :param trg: target indexes tensor of shape [seq_len, batch_size]
        :param decoder_hidden: input tensor of shape [num_layers, batch_size, hidden_size]
        :param encoder_outputs: encoder outputs of shape [src_len, batch_size, hidden_size]
        :return: decoded_batch: decoded target tensor of shape [batch_size, seq_len]
        '''
        seq_len, batch_size = trg.size()
        decoded_batch = torch.zeros((batch_size, seq_len))
        decoder_input = trg.data[0, :]  # 첫 번째 입력으로 시작 토큰 사용

        for t in range(seq_len):
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)  # decoder_output: [batch_size, vocab_size], decoder_hidden: [num_layers, batch_size, hidden_size]
            topv, topi = decoder_output.data.topk(1)  # topv: [batch_size, 1], topi: [batch_size, 1]
            topi = topi.view(-1)
            decoded_batch[:, t] = topi
            decoder_input = topi.detach().view(-1)

        return decoded_batch

    def beam_decode(self, target_tensor, decoder_hiddens, encoder_outputs=None):
        '''
        :param target_tensor: target indexes tensor of shape [seq_len, batch_size]
        :param decoder_hiddens: input tensor of shape [num_layers, batch_size, hidden_size]
        :param encoder_outputs: encoder outputs of shape [src_len, batch_size, hidden_size]
        :return: decoded_batch: list of decoded target sequences
        '''
        seq_len, batch_size = target_tensor.size()
        beam_width = 10
        topk = 1  # 생성할 문장의 개수
        decoded_batch = []

        # 문장 단위로 디코딩
        for idx in range(batch_size):
            decoder_hidden = (decoder_hiddens[0][:, idx, :].unsqueeze(1), decoder_hiddens[1][:, idx, :].unsqueeze(1))
            encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)

            # 시작 토큰으로 시작
            decoder_input = torch.LongTensor([self.sos_token]).to(self.device)  # decoder_input: [1]

            # 문장 생성을 위한 노드 초기화
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # 시작 노드 생성 (hidden vector, previous node, word id, logp, length)
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # 큐 시작
            nodes.put((-node.eval(), node))
            qsize = 1

            # 빔 검색 시작
            while True:
                # 디코딩 시간이 너무 오래 걸리면 중단
                if qsize > 2000:
                    break

                # 최상의 노드 가져오기
                score, n = nodes.get()
                decoder_input = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == self.eos_token and n.prevNode is not None:
                    endnodes.append((score, n))
                    # 필요한 문장 개수에 도달한 경우 중단
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # 디코더를 사용하여 한 단계 디코딩
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)

                # 상위 beam_width개의 결과 선택
                log_prob, indexes = torch.topk(decoder_output, beam_width)
                nextnodes = []

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(-1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # 새로운 노드를 큐에 추가
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    qsize += len(nextnodes) - 1

            # nbest 경로 선택 및 역추적
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordid)
                # 역추적
                while n.prevNode is not None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_batch.append(utterances)

        return decoded_batch
    
class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate: 디코더의 히든 상태
        :param previousNode: 이전 노드
        :param wordId: 현재 단어 ID
        :param logProb: 로그 확률
        :param length: 시퀀스 길이
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # 보상 함수 추가 가능

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward  # 길이에 대한 보상 및 패널티 적용

    def __lt__(self, other):
        return self.leng < other.leng  # 길이를 기준으로 노드 비교

    def __gt__(self, other):
        return self.leng > other.leng