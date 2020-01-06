import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F


class Bahdanau_mechanism(nn.Module):
    def __init__(self, dim):
        super(Bahdanau_mechanism, self).__init__()
        self.dim = dim
        self.query_layer = nn.Linear(dim, dim, bias=False)
        self.tanh = nn.Tanh()
        self.v = nn.Linear(dim, 1, bias=False)
    
    def forward(self, query, processed_memory):
        if query.dim() == 2:
            query = query.unsqueeze(1)

        processed_query = self.query_layer(query)
        alignment = self.v(self.tanh(processed_query + processed_memory))

        return alignment.squeeze(-1)


class AttentionWrapper(nn.Module):
    def __init__(self,
                 rnn_cell,
                 attn_mechanism):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.attn_mechansim = attn_mechanism
    
    def forward(self, decoder_input, attn_value, attn_hidden, encoder_output):
        # concat decorder input and previous attention context vector
        rnn_cell_input = torch.cat((decoder_input, attn_value), -1)
        # feed input to RNN (GRU cell)
        rnn_cell_output = self.rnn_cell(rnn_cell_input, attn_hidden)

        # alignment
        alignment = self.attn_mechansim(rnn_cell_output, encoder_output)

        # normalize attention value
        alignment = F.softmax(alignment)

        alignment = alignment.unsqueeze(1)
        attention_value = torch.bmm(alignment, encoder_output)

        alignment = alignment.squeeze(1)
        attention_value = attention_value.squeeze(1)

        return rnn_cell_output, attention_value, alignment