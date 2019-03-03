from pytorch_pretrained_bert.modeling import *
from torch.nn import BCEWithLogitsLoss
from layers.embedding import BertEmbedder
from layers.encoder import BertBiLSTMEncoder
from layers.decoder import CRFDecoder
from torch.autograd import Variable
import torch.nn.functional as F


class BertForMultiHeadProblem(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForMultiHeadProblem, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        if labels is not None:
            loss_bce = BCEWithLogitsLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1, self.num_labels)[active_loss]
                loss = loss_bce(active_logits, active_labels)
            else:
                loss = loss_bce(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class MultiHeadLayers(nn.Module):
    def __init__(self, label_embedding_size, hidden_size, lstm_hidden_size, ner_size, rel_size, activation="tanh", dropout=0.1):
        super(MultiHeadLayers, self).__init__()
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        
        self.activation = activation
        self.rel_size = rel_size
        
        self.label_embedding_size = label_embedding_size
        
        self.u_a = nn.Parameter(torch.randn(lstm_hidden_size + label_embedding_size, hidden_size)) 
        self.w_a = nn.Parameter(torch.randn(lstm_hidden_size + label_embedding_size, hidden_size))
        self.v = nn.Parameter(torch.randn(hidden_size, rel_size))
        self.b_s = nn.Parameter(torch.randn(hidden_size))
        
        if self.label_embedding_size != 0:
            self.label_embedding = nn.Embedding(ner_size, label_embedding_size)
        
    
    def broadcasting(self, left, right):
        left = left.transpose(1, 0)
        left = left.unsqueeze(3)
        
        right = right.transpose(2, 1)
        right = right.unsqueeze(0)
        
        B = left + right
        B = B.transpose(1, 0).transpose(3, 2)
        
        return B
    
    def forward(self, lstm_output, pred_ner):
        lstm_output = self.dropout_1(lstm_output)
        if self.label_embedding_size != 0:
            embeded_label = self.label_embedding(pred_ner)
            z = torch.cat([lstm_output, embeded_label], dim=2)
        else:
            z = lstm_output
        left = torch.einsum('aij,jk->aik', z, self.u_a)  
        right = torch.einsum('aij,jk->aik', z, self.w_a) 
        
        outer_sum = self.broadcasting(left, right)
        
        outer_sum_bias = 1 * ( outer_sum + self.b_s )
        
        if self.activation=="tanh":
            output = torch.tanh(outer_sum_bias)
        elif self.activation=="relu":
            output = torch.relu(outer_sum_bias)
        
        output = self.dropout_2(output)
        
        g = torch.einsum('aijk,kp->aijp', output, self.v)
        
        g = g.view(g.size(0), g.size(1), g.size(2) * self.rel_size)
    
        sigmoid = torch.nn.Sigmoid()
        probas = sigmoid(g)
        predictedRel = torch.round(probas)
        
        return predictedRel
        
    def score(self, lstm_output, gold_ner_labels, gold_rel_labels):
        lstm_output = self.dropout_1(lstm_output)
        
        if self.label_embedding_size != 0:
            embeded_label = self.label_embedding(gold_ner_labels)[:, :lstm_output.size(1), :]
            z = torch.cat([lstm_output, embeded_label], dim=2)
        else:
            z = lstm_output
            
        left = torch.einsum('aij,jk->aik', z, self.u_a)  
        right = torch.einsum('aij,jk->aik', z, self.w_a) 
        
        outer_sum = self.broadcasting(left, right)
        
        outer_sum_bias = 1 * ( outer_sum + self.b_s)
        
        if self.activation == "tanh":
            output = torch.tanh(outer_sum_bias)
        elif self.activation == "relu":
            output = torch.relu(outer_sum_bias)
        
        output = self.dropout_2(output)
        
        g = torch.einsum('aijk,kp->aijp', output, self.v)
        
        g = g.view(g.size(0), g.size(1), g.size(2) * self.rel_size)
        
        loss_bce = BCEWithLogitsLoss(reduction="mean")
        
        active_rel_labels = gold_rel_labels[:, :g.size(1), :g.size(2)]
        
        loss = loss_bce(g, active_rel_labels)
 
        return loss
    
        
class BertBiLSTMCRF(nn.Module):
    def __init__(self, encoder, decoder, extra=None, use_cuda=True, use_extra=True):
        super(BertBiLSTMCRF, self).__init__()
        self.encoder = encoder
        self.extra = extra
        self.decoder = decoder
        self.use_cuda = use_cuda
        self.use_extra = use_extra
        if use_cuda:
            self.cuda()
        #print(list(self.parameters()))
        
    def forward(self, batch):
        output, hidden = self.encoder(batch)
        predictedNer = self.decoder(output, batch[-3])

        if self.use_extra:
            predictedRel = self.extra(output, self.decoder(output, batch[-3]))
            return predictedNer, predictedRel
        else:
            return predictedNer, None

    def score(self, batch):
        output, _ = self.encoder(batch)
        lossNER = self.decoder.score(output, batch[-3], batch[-2].long())
        if self.use_extra:
            lossREL = self.extra.score(output, batch[-2].long(), batch[-1])
            return lossNER + lossREL
        else:
            return lossNER

    @classmethod
    def create(cls,
               ner_size,
               rel_size,
               bert_pretrained_path, embedding_dim=768, bert_mode="weighted",
               freeze=True,
               enc_hidden_dim=128, rnn_layers=1,
               input_dropout=0.1,
               use_cuda=True,
               use_extra=True,
               meta_dim=None,
               hidden_size=64,
               label_embedding_size=64, 
               activation="tanh"):
	
        embedder = BertEmbedder.create(bert_pretrained_path, embedding_dim, use_cuda, bert_mode, freeze)
        encoder = BertBiLSTMEncoder.create(embedder, enc_hidden_dim, rnn_layers, use_cuda)
        
        extra = None
        if use_extra:
            extra = MultiHeadLayers(label_embedding_size, hidden_size, enc_hidden_dim, ner_size, rel_size, activation, input_dropout)
            
        decoder = CRFDecoder(ner_size, encoder.output_dim, input_dropout, activation=activation)
        
        return cls(encoder, decoder, extra, use_cuda, use_extra)
