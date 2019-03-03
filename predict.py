import torch 
from pytorch_pretrained_bert.modeling import * 
from DataGen import DataGenerator 
from model import BertBiLSTMCRF 
import pickle

def convert_rel_matrix_to_str(predicted_rel, rel_list):
    index_rels = []
    label_rels = []
    
    for i, pos in enumerate(predicted_rel[0]):
        index_rel = []
        label_rel = []
        for j, val in enumerate(pos):
            if j % 5 == 0 and val == 1: #
                index_rel.append(int(j/5))
                label_rel.append("SP")
            elif j % 5 == 1 and val == 1: #
                index_rel.append(int(j/5))
                label_rel.append("N")
            elif j % 5 == 2 and val == 1:
                index_rel.append(int(j/5))
                label_rel.append("ST")
            elif j%5 == 3 and val == 1:
                index_rel.append(int(j/5))
                label_rel.append("PO")
            elif j%5 == 4 and val == 1:
                index_rel.append(int(j/5))
                label_rel.append("SC")
                
        if len(index_rel) == 0:
            index_rels.append([i])
            label_rels.append(["N"])
        else:
            index_rels.append(index_rel)
            label_rels.append(label_rel)
    
    return index_rels, label_rels 

def predict(model, featurized_sentence, rel_list, ner_list, use_extra=True):
    model.eval()
    input_tensor_1 = featurized_sentence[0].to("cuda")
    input_tensor_2 = featurized_sentence[1].to("cuda")
    input_tensor_3 = featurized_sentence[2].to("cuda")
    input_tensor_4 = featurized_sentence[3].to("cuda")
    input_tensor_5 = featurized_sentence[4].to("cuda")
    ner_logits, rel_logits = model([input_tensor_1, input_tensor_2, input_tensor_3, input_tensor_4, input_tensor_5])
    
    label_types = []
    ner_logits = ner_logits[0]
    for i in ner_logits:
        label_types.append(ner_list[i])
    
    if use_extra:
        rel_result = []
        ner_logits = ner_logits.tolist()
        print(rel_logits)
        index_rels, label_rels = convert_rel_matrix_to_str(rel_logits, rel_list)
        return label_types, index_rels, label_rels
    
    return label_types, None, None 

def load_model(save_path):
    with open(save_path,'rb') as f:
        data = pickle.load(f)
    return data["model"]
    
if __name__ == "__main__":
    PRETRAINED_MODEL = "models/"
    BERT_PRETRAINED_PATH = "./multi_cased_L-12_H-768_A-12/"
    #VALID_PATH = "/data/loclh2/QABot/data/test_analysis.txt"
    batch_size = 32
    shuffle = True
    use_cuda = True
    
    data_gen = DataGenerator(model=BertModel, model_name=BERT_PRETRAINED_PATH)
    
    #model = BertBiLSTMCRF.create(16,
                     #            len(data_gen.rel_list),
                      #           BERT_PRETRAINED_PATH,
                       #          freeze=True,
                        #         rnn_layers=2,
                         #        input_dropout=0.1,
                          #       use_cuda=use_cuda,
                           #      hidden_size=64,
                            #     label_embedding_size=64,
                             #    enc_hidden_dim=64,
                              #   activation="tanh")
    
   # model.load_state_dict(torch.load(PRETRAINED_MODEL))
   
    model = load_model("models/rel_ner_v1_adam/bert_ner_epoches=50_valid_loss=10.281595188638438.pickle")
    
    sentence = "Con trai của Ronald là ai"
    featurized_sentence = data_gen.get_featurized_sentence(sentence)
    
    label_types, index_rels, label_rels = predict(model,
                                                  featurized_sentence,
                                                  data_gen.rel_list,
                                                  data_gen.ner_list,
                                                  use_extra=True)
    print("label_types : " + str(label_types))
    print("index_rels : " + str(index_rels))
    print("label_rels : " + str(label_rels))
