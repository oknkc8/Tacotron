import torch
from torch import nn
from torch.autograd import Variable
from .attention import AttentionWrapper, Bahdanau_mechanism


class Prenet(nn.Module):
    '''
    Prenet
        FC-256-ReLU
        +
        Dropout(0.5)
        +
        FC-128-ReLU
        +
        Dropout(0.5)
    '''
    def __init__(self, 
                 input_size=256, 
                 hidden_size=256, 
                 output_size=128):
        super(Prenet, self).__init__()
        self.layers = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(hidden_size, output_size),
                        nn.ReLU(),
                        nn.Dropout(0.5))
    
    def forward(self, inputs):
        return self.layers(inputs)


class BatchNormConv1D(nn.Module):
    '''
    BatchNormConv1D
    '''
    def __init__(self,
                 input_size,
                 output_size,
                 kernel_size,
                 stride,
                 padding,
                 activation=None):
        super().__init__()
        self.conv1d = nn.Conv1D(input_size, output_size, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        self.batchnorm1d = nn.BatchNormConv1D(output_size)
        self.activation = activation
    
    def forward(self, inputs):
        x = self.conv1d(inputs)
        if self.activation is not None:
            x = self.activation(x)
        return self.batchnorm1d(x)


class Highway(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 N_layers=4):
        super().__init__()
        self.N_layers = N_layers
        self.H = nn.Linear(input_size, output_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(input_size, output_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = inputs
        for _ in range(self.N_layers):
            H = self.relu(self.H(x))
            T = self.sigmoid(self.T(x))
            x = H * T + x * (1. - T)
        return x


class CBHG(nn.Module):
    '''
    CBHG
        Conv1D Bank (conv-k-128-ReLU)
            K = 16
        +
        Max Pooling
            stride = 1
            width = 2
        +
        Conv1D Projections (conv-3-128-ReLU + conv3-128)
        +
        Highway Net (4 x FC-128-ReLU)
        +
        Bi-GRU (128 cells)
    '''
    def __init__(self,
                 input_size,
                 K,
                 projection_size_1=128,
                 projection_size_2=128):
        super(CBHG, self).__init__()
        self.input_size = input_size
        self.K = K
        self.relu = nn.ReLU()
        self.conv1d_bank = nn.ModuleList(
                            [BatchNormConv1D(input_size, input_size,
                                             kernel_size=k, stride=1, padding=k//2, activation=self.relu)
                             for k in range(1, K+1)])
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.conv1d_proj_1 = BatchNormConv1D(input_size, projection_size_1,
                                             kernel_size=3, stride=1, padding=1, activation=self.relu)
        self.conv1d_proj_2 = BatchNormConv1D(projection_size_1, projection_size_2,
                                             kernel_size=3, stride=1, padding=1)
        self.pre_highway = nn.Linear(projection_size_2, input_size, bias=False)
        self.highway = Highway(input_size, input_size, N_layers=4)
        self.gru = nn.GRU(input_size, input_size, batch_fisrt=True, bidirectional=True)
    
    def forward(self, inputs, input_lengths):
        x = inputs
        x = x.transpose(1, 2)

        T = x.size(-1)
        x = torch.cat(
                [conv1d(x)[:,:,T]
                for conv1d in self.conv1d_bank], dim=1)
        
        x = self.maxpool1d(x)[:,:,T]
        x = self.conv1d_proj_1(x)
        x = self.conv1d_proj_2(x)

        x = x.transpose(1, 2)
        x = self.pre_highway(x)

        # Residual connect
        x += inputs
        x = self.highway(x)

        outputs, _ = self.gru(x)

        return outputs


class Encoder(nn.Module):
    '''
    Encoder 
    '''
    def __init__(self, 
                 N_character, 
                 embedding_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(N_character, embedding_dim)
        self.prenet = Prenet(input_size=256, hidden_size=256, output_size=128)
        self.cbhg = CBHG(input_size=128, K=16, projection_size_1=128, projection_size_2=128)
    
    def forward(self, inputs, input_lengths):
        x = self.embedding(inputs)
        x = self.prenet(x)
        x = self.cbhg(x, input_lengths)
        return x
        

class Decoder(nn.Module):
    '''
    Decoder
    '''
    def __init__(self,
                 mel_dim,
                 r):
        super(Decoder, self).__init__()
        self.mel_dim = mel_dim
        self.r = r
        self.prenet = Prenet(input_size=256, hidden_size=256, output_size=128)
        self.attn_rnn = AttentionWrapper(
                            nn.GRUCell(256+128, 256),
                            Bahdanau_mechanism)
        #self.attention_rnn = 
        self.proj_to_decoder = nn.Linear(256+256, 256)
        self.decoder_rnn = nn.Sequential(
                             nn.GRUCell(256,256),
                             nn.GRUCell(256,256))
        self.memory_layer = nn.Linear(256, 256, bias=False)

        self.proj_to_mel = nn.Linear(256, self.mel_dim * self.r)

    def forward(self, encoder_outputs, targets):
        batch_size = encoder_outputs.size(0)
        
        # when syntheis
        greedy = targets is None
        
        # grouping multiple frames
        if targets is not None:
            if targets.size(-1) == self.mel_dim:
                targets = targets.view(batch_size, targets(1)//self.r, -1)
            T = targets.size(1)

        go_frames = Variable(
            encoder_outputs.data.new(batch_size, self.mel_dim * self.r)).zero()
        attn_rnn_hidden = Variable(
            encoder_outputs.data.new(batch_size, 256)).zero()
        decoder_rnn_hidden = Variable(
            encoder_outputs.data.new(batch_size, 256)).zero()
        now_attn_value = Variable(
            encoder_outputs.data.new(batch_size, 256)).zero()

        outputs = []
        alignments = []

        t = 0
        now_input_frames = go_frames
        while t < T:
            if t > 0:
                # when syntheis, input frames are end of previous output
                # if ont, input frames are tareget frames in time t (teacher forcing)
                now_input_frames = outputs[-1] if greedy else targets[t - 1]
            
            now_input_frames = self.prenet(now_input_frames)

            attn_rnn_hidden, now_attn_value, alignment = self.attn_rnn(
                                                            decoder_input = now_input_frames,
                                                            attn_value = now_attn_value,
                                                            attn_hidden = attn_rnn_hidden,
                                                            encoder_output = encoder_outputs)

            decoder_rnn_input = self.proj_to_decoder(
                                    torch.cat((attn_rnn_hidden, now_attn_value), -1))
            
            for i in range(len(self.decoder_rnn)):
                decoder_rnn_hidden[i] = self.decoder_rnn(
                                            decoder_rnn_input, decoder_rnn_hidden[i])
                decoder_rnn_input = decoder_rnn_hidden[i] + decoder_rnn_input
            
            output = decoder_rnn_input
            output = self.proj_to_mel(output)

            outputs += [output]
            alignments += [alignment]

            t += 1
        
        alignments = torch.stack(alignments).transpose(0, 1)
        outputs = torch.stack(alignments).transpose(0, 1).contiguous()

        return outputs, alignments
    

class Tacotron(nn.Module):
    '''
    Tacotron
        Encoder
        +
        Decoder
        +
        CBHG
        +
        Linear Layer
    '''
    def __init__(self, 
                 N_character, 
                 embedding_dim, 
                 mel_dim, 
                 final_output_dim, 
                 r):
        '''
        params
            N_character : number of characters (=number of symbols)
            embedding_dim : output dim of character embedding
            mel_dim : dim of mel-spectrogram
            final_output_dim : last dim of targets
            r : reduction factor
        '''

        super(Tacotron, self).__init__()
        self.mel_dim = mel_dim
        self.final_output_dim = final_output_dim
        self.encoder = Encoder(N_character, embedding_dim)
        self.decoder = Decoder(mel_dim, r)
        self.cbhg = CBHG(input_size=mel_dim, K=8, projection_size_1=256, projection_size_2=mel_dim)
        self.to_linear_mel = nn.Linear(mel_dim * 2, final_output_dim)

    def forward(self, inputs, targets=None, input_lengths=None):
        batch_size = inputs.size(0)

        inputs = self.embedding(inputs)

        encoder_outputs = self.encoder(inputs, input_lengths)

        mel_outputs, alignments = self.decoder(
                                    encoder_outputs, targets)

        mel_outputs = mel_outputs.view(batch_size, -1, self.mel_dim)

        linear_outputs = self.cbhg(mel_outputs)
        linear_outputs = self.to_linear_mel(linear_outputs)

        return mel_outputs, linear_outputs, alignments

