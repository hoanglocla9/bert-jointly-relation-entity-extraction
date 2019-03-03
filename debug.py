import torch
from pytorch_pretrained_bert.modeling import *
from DataGen import DataGenerator
from model import BertBiLSTMCRF
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ ==  "__main__":
    BERT_PRETRAINED_PATH = "./multi_cased_L-12_H-768_A-12/"
    TRAIN_PATH = "/data/loclh2/QABot/data/train_analysis.txt"
    batch_size = 32
    shuffle = False
    
    data_gen = DataGenerator(model=BertModel, model_name=BERT_PRETRAINED_PATH)
        
    train_gen = data_gen.get_generator(TRAIN_PATH, batch_size, shuffle=shuffle)
    for batch in train_gen:
        pass
    
    