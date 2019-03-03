from pathlib import Path
import torch
from torch import nn

import pytorch_pretrained_bert as _bert
from random import shuffle

import numpy as np

#from dougu import flatten, lines


_device = torch.device("cpu")

class DataGenerator:

    MASK = "[MASK]"
    CLS = "[CLS]"
    SEP = "[SEP]"
    ner_list = ['O', 'I-CAT', 'B-TIM', 'I-ETT', 'B-RLT', 'B-ETT', 'B-VAR', 'I-RLT', 'I-TIM', 'B-CAT', 'I-VAR', 'X', '[CLS]', '[SEP]'] 
    rel_list = ["SP", "N", "ST", "PO", "SC"]
    
    def __init__(self, model, model_name, device=None, half_precision=False, rel_max_len=32):
        self.model_name = model_name
        self.device = device or _device
        do_lower_case = "uncased" in model_name
        self.tokenizer = _bert.BertTokenizer.from_pretrained(self.model_name, do_lower_case=do_lower_case)
        
        maybe_model_wrapper = model.from_pretrained(model_name).to(device=self.device)
        try:
            self.model = maybe_model_wrapper.bert
        except AttributeError:
            self.model = maybe_model_wrapper
        if half_precision:
            self.model.half()
        self.max_len = \
            self.model.embeddings.position_embeddings.weight.size(0)
        self.dim = self.model.embeddings.position_embeddings.weight.size(1)
        self.rel_max_len = rel_max_len
    def tokenize(self, text, masked_idxs=None):
        tokenized_text = self.tokenizer.tokenize(text)
        if masked_idxs is not None:
            for idx in masked_idxs:
                tokenized_text[idx] = self.MASK
        tokenized = [self.CLS] + tokenized_text + [self.SEP]
        return tokenized

    def tokenize_to_ids(self, text, masked_idxs=None, pad=True):
        tokens = self.tokenize(text, masked_idxs)
        return self.convert_tokens_to_ids(tokens, pad=pad)

    def convert_tokens_to_ids(self, tokens, pad=True):
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = torch.tensor([token_ids]).to(device=self.device)
        assert ids.size(1) < self.max_len
        if pad:
            padded_ids = torch.zeros(1, self.max_len).to(ids)
            padded_ids[0, :ids.size(1)] = ids
            mask = torch.zeros(1, self.max_len).to(ids)
            mask[0, :ids.size(1)] = 1
            
            return padded_ids, mask
        else:
            return ids
            
    def padding_labels(self, true_labels, pad=True):
        true_labels = torch.tensor([true_labels]).to(device=self.device)
        assert true_labels.size(1) < self.max_len
        if pad:
            
            padded_labels = torch.zeros(1, self.max_len).to(true_labels)
            padded_labels[0, :true_labels.size(1)] = true_labels
            return padded_labels
        else:
            return_true_labels = torch.zeros(1, true_labels.size(1))
            return_true_labels[0,:] = true_labels
            return return_true_labels
            
    def parseData(self, file_name):
        data = {}
        with open(file_name, "r") as f:
            current_id = ""
            for line in f:
                if len(line) <= 2:
                    continue
                if "#" in line:
                    current_id = line.strip()
                    data[current_id] = {}
                    data[current_id]["tokens"] = []
                    data[current_id]["label_types"] = []
                    data[current_id]["index_rels"] = []
                    data[current_id]["label_rels"] = []
                else:
                    info = line.split("\t") 
                
                    data[current_id]["tokens"].append(info[1])
                    data[current_id]["label_types"].append(info[2])
                
                    index_rels_info =  info[4].strip().replace("[","").replace("]","").replace(" ", "").split(",")
                    data[current_id]["index_rels"].append([int(i) for i in index_rels_info])
                
                    label_rels_info =  info[3].strip().replace("[","").replace("]","").replace(" ", "").split(",")
                    data[current_id]["label_rels"].append([str(i).replace("'", "") for i in label_rels_info])
        return data

    def convertData(self, original_data, padding=True):
        ver_1_data = {}
        
        for id in original_data:
            ver_1_item = {}
            ver_1_item["tokens"] = []
            ver_1_item["label_types"] = original_data[id]["label_types"]
            ver_1_item["index_rels"] = original_data[id]["index_rels"]
            ver_1_item["label_rels"] = original_data[id]["label_rels"]
            shift_idx = 0
            for idx, orig_token in enumerate(original_data[id]["tokens"]):
                sub_orig_tokens = orig_token.split("_")
                ## tokens
                ver_1_item["tokens"] += sub_orig_tokens
                if len(sub_orig_tokens) > 1:
                    ## label_types
                    orig_item_label_type = ver_1_item["label_types"][idx + shift_idx].replace("B-","").replace("I-", "")
                    if ver_1_item["label_types"][idx + shift_idx] == "O":
                        ver_1_item["label_types"] = ver_1_item["label_types"][:idx + shift_idx + 1] + ["O"] * (len(sub_orig_tokens) - 1) + ver_1_item["label_types"][idx+shift_idx + 1:]
                    else:
                        ver_1_item["label_types"] = ver_1_item["label_types"][:idx + shift_idx + 1] + ["I-" + orig_item_label_type] * (len(sub_orig_tokens) - 1) + ver_1_item["label_types"][idx+shift_idx + 1:]
                    ## label_rels
                    ver_1_item["label_rels"] = ver_1_item["label_rels"][:idx + shift_idx] + [["N"]]  * (len(sub_orig_tokens) - 1)  + ver_1_item["label_rels"][idx + shift_idx:]

                    ## index_rels
                    prev_index_rels = [] # ver_1_item["label_rels"][:idx + shift_idx]
                    next_index_rels = [] # ver_1_item["label_rels"][idx+shift_idx:]
                
                    for index_rels in ver_1_item["index_rels"][:idx + shift_idx]:    
                        prev_index_rel = []
                        for index_rel in index_rels:
                            if index_rel >= idx + shift_idx:
                                prev_index_rel.append(index_rel + len(sub_orig_tokens) - 1)
                            else:
                                prev_index_rel.append(index_rel)
                        prev_index_rels.append(prev_index_rel)

                    for index_rels in ver_1_item["index_rels"][idx+shift_idx:]:
                        next_index_rel = []
                        for index_rel in index_rels:
                            if index_rel >= idx + shift_idx:
                                next_index_rel.append(index_rel + len(sub_orig_tokens) - 1)
                            else:
                                next_index_rel.append(index_rel)

                        next_index_rels.append(next_index_rel)

                    ver_1_item["index_rels"] = prev_index_rels + [[(idx + shift_idx + i)] for i in range(len(sub_orig_tokens) - 1)] + next_index_rels
                shift_idx += len(sub_orig_tokens) - 1
            ver_1_data[id] = ver_1_item
        
        
        bert_data = {}
        for id in ver_1_data:
            bert_item = {}
            bert_item["subword_ids"] = []
            bert_item["mask"] = []
            bert_item["token_starts"] = []

            bert_item["label_types"] = ver_1_data[id]["label_types"]
            bert_item["index_rels"] = ver_1_data[id]["index_rels"]
            bert_item["label_rels"] = ver_1_data[id]["label_rels"]
            

            shift_idx = 0
            
            bert_tokens = []
            bert_token_starts = []
            for idx, ver_1_token in enumerate(ver_1_data[id]["tokens"]):
                sub_ver_1_tokens = self.tokenizer.tokenize(ver_1_token)
                bert_token_starts.append(1 + len(sub_ver_1_tokens))
                ## tokens
                bert_tokens += sub_ver_1_tokens
                if len(sub_ver_1_tokens) > 1:
                    ## label_types
                    bert_item["label_types"] = bert_item["label_types"][:idx + shift_idx + 1] + ["X"] * (len(sub_ver_1_tokens) - 1) + bert_item["label_types"][idx+shift_idx + 1:]

                    ## label_rels
                    bert_item["label_rels"] = bert_item["label_rels"][:idx + shift_idx] + [["N"]]  * (len(sub_ver_1_tokens) - 1)  + bert_item["label_rels"][idx + shift_idx:]
                    ## index_rels
                    prev_index_rels = [] # bert_item["label_rels"][:idx + shift_idx]
                    next_index_rels = [] # bert_item["label_rels"][idx+shift_idx:]
                    for index_rels in bert_item["index_rels"][:idx + shift_idx]:    
                        prev_index_rel = []
                        for index_rel in index_rels:
                            if index_rel >= idx + shift_idx:
                                prev_index_rel.append(index_rel + len(sub_ver_1_tokens) - 1)
                            else:
                                prev_index_rel.append(index_rel)
                        prev_index_rels.append(prev_index_rel)

                    for index_rels in bert_item["index_rels"][idx+shift_idx:]:
                        next_index_rel = []
                        for index_rel in index_rels:
                            if index_rel >= idx + shift_idx:
                                next_index_rel.append(index_rel + len(sub_ver_1_tokens) - 1)
                            else:
                                next_index_rel.append(index_rel)

                        next_index_rels.append(next_index_rel)

                    bert_item["index_rels"] = prev_index_rels + [[(idx + shift_idx + i)] for i in range(len(sub_ver_1_tokens) - 1)] + next_index_rels
                shift_idx += len(sub_ver_1_tokens) - 1
                
            ######################################################
            bert_tokens = [self.CLS] + bert_tokens + [self.SEP]
            bert_item["subword_ids"], bert_item["mask"] = self.convert_tokens_to_ids(bert_tokens)

            bert_item["label_types"] = [self.CLS] + bert_item["label_types"] + [self.SEP]
            
            bert_item["label_rels"] = [["N"]] + bert_item["label_rels"]  + [["N"]]
            
            bert_item_index_rels_ = []
            for i in bert_item["index_rels"]:
                row = []
                for j in i:
                    row.append(j+1)
                bert_item_index_rels_.append(row)
            bert_item["index_rels"] = [[0]] + bert_item_index_rels_ + [[len(bert_item_index_rels_) + 1]]
            
            bert_item["token_starts"] = torch.zeros(1, self.max_len).to(bert_item["subword_ids"])
            bert_item["token_starts"][0, bert_item["token_starts"]] = 1

            bert_item_label_types = []
            for label_type in bert_item["label_types"]:
                bert_item_label_types.append(self.ner_list.index(label_type))
            
            bert_item["label_types"] = self.padding_labels(bert_item_label_types, padding)
            
            bert_item["rel_matrix"] = self.convert_rel_ids(bert_item["index_rels"], bert_item["label_rels"])
            
            bert_item["token_type_ids"] = torch.zeros(bert_item["mask"].size(0), bert_item["mask"].size(1))
            bert_item["token_type_ids"][:, :len(bert_tokens)] = 1
            bert_item["token_type_ids"] = bert_item["token_type_ids"].long()
            #################################################################
            bert_data[id] = bert_item
            
            
        return bert_data
    
    def convert_rel_ids(self, index_rels, label_rels):
        num_rels = len(self.rel_list)
        rel_matrix = torch.zeros(1, self.max_len, self.rel_max_len * num_rels)
        
        for i, (idxs, labels) in enumerate(zip(index_rels, label_rels)):
            for idx, label in zip(idxs, labels):
                rel_matrix[0, i, idx * num_rels + self.rel_list.index(label)] = 1
        return rel_matrix
        
    def subword_tokenize_to_ids(self, tokens):
        subwords, token_start_idxs = self.subword_tokenize(tokens)
        subword_ids, mask = self.convert_tokens_to_ids(subwords)
        token_starts = torch.zeros(1, self.max_len).to(subword_ids)
        token_starts[0, token_start_idxs] = 1
        return subword_ids, mask, token_starts

    def segment_ids(self, segment1_len, segment2_len):
        ids = [0] * segment1_len + [1] * segment2_len
        return torch.tensor([ids]).to(device=self.device)

    def get_featurized_sentence(self, sentence, padding=True):
        tokens = self.tokenize(sentence)
        if padding:
            input_ids, mask = self.convert_tokens_to_ids (tokens, padding)
        else:
            input_ids = self.convert_tokens_to_ids (tokens, padding)
        
        input_types = torch.zeros(input_ids.size(0), input_ids.size(1))
        input_types[:, :len(tokens)] = 1
        
        empty_ners = torch.zeros(input_ids.size(0), input_ids.size(1))
        empty_rels = torch.zeros(input_ids.size(0), input_ids.size(1), len(self.rel_list))
        return input_ids, mask, input_types, empty_ners, empty_rels
    
    def get_featurized_sentences(self, file_name, padding=True):
        original_data = self.parseData(file_name)
        bertData = self.convertData(original_data, padding)
        
        featurized_sentences = []
        for id in bertData:
            features = {}
            features["bert_ids"], features["bert_mask"], features["bert_token_starts"], features["ett_tags"], features["rel_matrix"], features["token_type_ids"] = \
                  bertData[id]["subword_ids"], bertData[id]["mask"], bertData[id]["token_starts"], bertData[id]["label_types"], bertData[id]["rel_matrix"], bertData[id]["token_type_ids"]
            featurized_sentences.append(features)
            
        self.num_samples = len(featurized_sentences)
        return featurized_sentences

    def get_generator(self, file_name, batch_size=32, padding=True, is_shuffle=True):
        batch_input_ids = []
        batch_mask_ids = []
        batch_ner_ids = []
        batch_input_type_ids = []
        batch_rel_matrix = []
        featurized_sentences = self.get_featurized_sentences(file_name, padding)
        if is_shuffle:
            shuffle(featurized_sentences)
        for idx, features in enumerate(featurized_sentences):
            batch_input_ids.append(features["bert_ids"]) 
            batch_input_type_ids.append(features["token_type_ids"]) 
            batch_mask_ids.append(features["bert_mask"])
            batch_ner_ids.append(features["ett_tags"])
            batch_rel_matrix.append(features["rel_matrix"])
            if idx % batch_size == batch_size-1:
                return_batch_input_ids = torch.cat(batch_input_ids, dim=0)
                return_batch_mask_ids = torch.cat(batch_mask_ids, dim=0)
                return_batch_input_type_ids = torch.cat(batch_input_type_ids, dim=0)
                return_batch_label_ids = torch.cat(batch_ner_ids, dim=0)
                return_batch_rel_matrix = torch.cat(batch_rel_matrix, dim=0)
	    
                batch_input_ids = []
                batch_mask_ids = []
                batch_ner_ids = []
                batch_input_type_ids = []
                batch_rel_matrix = []
                
                yield return_batch_input_ids, return_batch_mask_ids, return_batch_input_type_ids, return_batch_label_ids, return_batch_rel_matrix
        
        if len(batch_input_ids) != 0:
            yield torch.cat(batch_input_ids, dim=0), torch.cat(batch_mask_ids, dim=0), torch.cat(batch_input_type_ids, dim=0), torch.cat(batch_ner_ids, dim=0), torch.cat(batch_rel_matrix, dim=0)
