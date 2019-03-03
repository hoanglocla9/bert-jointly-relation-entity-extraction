from torch import nn
import torch
from pytorch_pretrained_bert.modeling import BertModel, BertConfig

class BertEmbedder(nn.Module):
    def __init__(self, model, bert_pretrained_path,
                 freeze=True, embedding_dim=768, use_cuda=True, bert_mode="weighted",):
        super(BertEmbedder, self).__init__()
        self.bert_pretrained_path = bert_pretrained_path
        self.is_freeze = freeze
        self.embedding_dim = embedding_dim
        self.model = model
        self.use_cuda = use_cuda
        self.bert_mode = bert_mode
        if self.bert_mode == "weighted":
            self.bert_weights = nn.Parameter(torch.FloatTensor(12, 1))
            self.bert_gamma = nn.Parameter(torch.FloatTensor(1, 1))

        if use_cuda:
            self.cuda()

        self.init_weights()

    def init_weights(self):
        if self.bert_mode == "weighted":
            nn.init.xavier_normal(self.bert_gamma)
            nn.init.xavier_normal(self.bert_weights)

    def forward(self, *batch):
        input_ids, input_mask, input_type_ids = batch[:3]
        all_encoder_layers, _ = self.model(input_ids.long(), token_type_ids=input_type_ids.long(), attention_mask=input_mask.long())
        if self.bert_mode == "last":
            return all_encoder_layers[-1]
        elif self.bert_mode == "weighted":
            all_encoder_layers = torch.stack([a * b for a, b in zip(all_encoder_layers, self.bert_weights)])
            return self.bert_gamma * torch.sum(all_encoder_layers, dim=0)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_to(self, to=-1):
        idx = 0
        if to < 0:
            to = len(self.model.encoder.layer) + to + 1
        for idx in range(to):
            for param in self.model.encoder.layer[idx].parameters():
                param.requires_grad = False
        print("Embeddings freezed to {}".format(to))
        to = len(self.model.encoder.layer)
        for idx in range(idx, to):
            for param in self.model.encoder.layer[idx].parameters():
                param.requires_grad = True

    @classmethod
    def create(cls,
               bert_pretrained_path, embedding_dim=768, use_cuda=True, bert_mode="weighted",
               freeze=True):
        model = BertModel.from_pretrained(bert_pretrained_path)
        #if use_cuda:
           # device = torch.device("cuda")
          #  map_location = "cuda"
        #else:
           # map_location = "cpu"
          #  device = torch.device("cpu")
        #model = model.to(device)
        model = cls(model=model, embedding_dim=embedding_dim, use_cuda=use_cuda, bert_mode=bert_mode,
                    bert_pretrained_path=bert_pretrained_path, freeze=freeze)
        if freeze:
            model.freeze()
        return model
