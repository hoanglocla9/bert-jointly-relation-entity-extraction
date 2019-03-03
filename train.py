import torch
from pytorch_pretrained_bert.modeling import *
from tqdm import tqdm
from DataGen import DataGenerator
from model import BertBiLSTMCRF
import matplotlib.pyplot as plt
import numpy as np
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

	    
def evaluate(model, epoch, valid_generator, device):
    #load valid data_generator
    num_valid = 0
    num_correct = total_loss = 0
    batch_size = 0
    idx = 0 
    
    for batch_input_ids, batch_mask_ids, batch_input_type_ids, batch_label_ids, batch_rel_matrix in valid_generator:
            #compute loss
        loss = model.score([batch_input_ids.to(device), batch_mask_ids.to(device), batch_input_type_ids.to(device), batch_label_ids.to(device), batch_rel_matrix.to(device)])
        
        idx += 1
        total_loss += loss.mean().item()
        
    loss = total_loss / idx
       
    print('\nValidation : Loss: {:.6f} Accuracy: {}/{} ({:.4f}%)\n'.format(loss, 0, 0, 0))

    return loss, 0

def draw_learning_curve(epoch, training_loss, valid_loss, 
			save_path="models/learning_curve.png", 
			title="Learning Curve"):
    plt.plot(epoch, training_loss, '-b')
    plt.plot(epoch, valid_loss, '--r')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.title(title)

    # save image
    plt.savefig(save_path)  # should before show method

def train(model, data_gen,  train_path, valid_path, start_epoch, num_epoches, save_path, device, 
	  batch_size=32, 
	  decay_rate=0.1, 
	  learning_rate=0.001,
      momentum=0.9,
	  update_lr_epoches=[], 
	  shuffle=True):
    
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=learning_rate, 
                                momentum=0.9)
    loss_history = [[], [], []]
    
    
    for epoch in range(start_epoch, num_epoches + start_epoch):
        step = 0
        train_gen = data_gen.get_generator(train_path, batch_size, is_shuffle=shuffle)
        if epoch in update_lr_epoches:
            learning_rate = learning_rate * decay_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            print('Updating the learning rate at epoch: ' + str(epoch) + ', value: ' + str(learning_rate))
            
        train_loss = []
        for batch_input_ids, batch_mask_ids, batch_input_type_ids, batch_label_types, batch_rel_matrix in train_gen:
            model.zero_grad()
            loss = model.score([batch_input_ids.to(device), batch_mask_ids.to(device), batch_input_type_ids.to(device), batch_label_types.to(device), batch_rel_matrix.to(device)])
            
            loss.backward()
            
            optimizer.step()    
            optimizer.zero_grad()
            loss = loss.data.cpu().tolist()
            train_loss.append(loss)
            print('Training: Epoch %d, step %5d / %d loss: %.3f'%(epoch + 1, step + 1, data_gen.num_samples/batch_size + 1, loss))
            step += 1
            
        valid_gen = data_gen.get_generator(valid_path, batch_size, is_shuffle=shuffle)
        valid_loss, valid_accuracy = evaluate(model, epoch, valid_gen, device )
        
        ### visualization the learning curve
        train_loss_value = np.mean(train_loss)
        valid_loss_value = np.mean(valid_loss)
        
        loss_history[0].append(epoch + 1)
        loss_history[1].append(train_loss_value)
        loss_history[2].append(valid_loss_value)
        
        draw_learning_curve(loss_history[0], loss_history[1], loss_history[2])
        
        model.train()
        
        save_data = {"model": model, "history": loss_history}
        with open(save_path + "bert_ner_epoches=" + str(epoch + 1) + "_valid_loss=" + str(valid_loss) +'.pickle', 'wb') as handle:
            pickle.dump(save_data, handle, protocol=2)
            
def load_pretrain_model(model, pretrained_path):
    model.load_state_dict(torch.load(pretrained_path))
    return model

if __name__ ==  "__main__":
    TRAIN_PATH = "data/train_analysis.txt"
    VALID_PATH = "data/test_analysis.txt"
    PRETRAINED_PATH = "./multi_cased_L-12_H-768_A-12/"
    SAVE_PATH = "models/rel_ner_v1_adam/"
    batch_size = 4
    shuffle = True 
    use_cuda = True
    use_extra = True
    freeze = False   ## !(fine tuning) or not
    start_epoch=0
    num_epoches = 50
    learning_rate = 0.001
    decay_rate = 0.1
    update_lr_epoches = [35,]
    momentum=0.9
    
    if use_cuda:
        device = "cuda"
    else:
        device = "cpu"
        
    data_gen = DataGenerator(model=BertModel, 
                             model_name=PRETRAINED_PATH, 
                             device=device)
    
    ner_size = len(data_gen.ner_list)
    rel_size = len(data_gen.rel_list)

    model = BertBiLSTMCRF.create(ner_size, 
                                 rel_size, 
                                 PRETRAINED_PATH, 
                                 freeze=freeze, 
                                 rnn_layers=2, 
                                 input_dropout=0.1, 
                                 use_cuda=use_cuda,
                                 use_extra=use_extra,
                                 hidden_size=64,
                                 label_embedding_size=32,
                                 enc_hidden_dim=64,
                                 activation="tanh")
    
    train(model=model, 
          data_gen=data_gen,
          train_path=TRAIN_PATH,
          valid_path=VALID_PATH,
          start_epoch=start_epoch,
          num_epoches=num_epoches,
          save_path=SAVE_PATH,
          device=device,
          batch_size=batch_size,
          decay_rate=decay_rate,
          learning_rate=learning_rate,
          momentum=momentum,
          update_lr_epoches=update_lr_epoches,
          shuffle=shuffle)
