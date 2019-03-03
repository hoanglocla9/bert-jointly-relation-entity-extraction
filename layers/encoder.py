from torch import nn
import torch

class BertBiLSTMEncoder(nn.Module):
    def __init__(self, embeddings,
                 hidden_dim=128, rnn_layers=1, use_cuda=True):
        super(BertBiLSTMEncoder, self).__init__()
        self.embeddings = embeddings
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(
            self.embeddings.embedding_dim, hidden_dim // 2,
            rnn_layers, batch_first=True, bidirectional=True)
        self.hidden = None
        if use_cuda:
            self.cuda()
        self.init_weights()
        self.output_dim = hidden_dim
    
    def init_weights(self):
        #for p in self.lstm.parameters():
           # nn.init.xavier_normal(p)
        pass 
    def sort_lengths(self, inputs, input_lens):
        inputs_list = inputs.tolist()
        sorted_input_lens = sorted(input_lens, key=lambda l: l, reverse=True)
        sorted_input_len_ids = [input_lens.index(i) for i in sorted_input_lens]
        sorted_input_list = [inputs_list[i] for i in sorted_input_len_ids]
        return torch.tensor(sorted_input_list), sorted_input_lens
        
    def forward(self, batch):
        input, input_mask = batch[0], batch[1]
        output = self.embeddings(*batch)
        # output = self.dropout(output)
        lens = input_mask.sum(-1).tolist()
        output, lens = self.sort_lengths(output, lens)
        
        output = nn.utils.rnn.pack_padded_sequence(output, lens, batch_first=True)
        
        if self.use_cuda:
            output, self.hidden = self.lstm(output.to("cuda"))
        else:
            output, self.hidden = self.lstm(output.to("cpu"))
        
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        return output, self.hidden

    @classmethod
    def create(cls, embeddings, hidden_dim=128, rnn_layers=1, use_cuda=True):
        model = cls(
            embeddings=embeddings, hidden_dim=hidden_dim, rnn_layers=rnn_layers, use_cuda=use_cuda)
        return model
