import torch 
import json 
import networks 

import os 
from games import fen_to_7d
import numpy 
import random 
import time 
from torch.utils.data import DataLoader,Dataset
from model_utilities import save_model,load_model
from matplotlib import pyplot as plt 


DATASET_ROOT  	    = r"//FILESERVER/S Drive/Data/chessSL"
DEV                 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(dataroot):
    dataset     = []
    indexer     = {} 

    for file in os.listdir(dataroot):
        filename    = f"{dataroot}/{file}"

        with open(filename,"r") as file:
            data        = json.loads(file.read())

            for data_dict in data:
                move_vals   = numpy.zeros(shape=(1968))
                indices     = data_dict['ids']
                values      = data_dict['vals']

                numpy.put(move_vals,indices,values)
                repr        = fen_to_7d(data_dict['fen'],req_grad=True) 
                dataset.append((repr,move_vals))
                if data_dict['fen'] in indexer:
                    indexer[data_dict['fen']] += 1 
                else:
                    indexer[data_dict['fen']] = 1  
    return dataset,indexer 


def train_on_data(model:networks.FullNet,dataset:list,bs=512,epochs=10):
    split_point         = int(.1*len(dataset))
    test,train          = dataset[:split_point],dataset[split_point:]

    dataloader          = DataLoader(networks.ChessDataset(train),batch_size=bs,shuffle=True)


    #TRAIN 
    model.loss_fn       = torch.nn.CrossEntropyLoss()
    n_batches           = len(str(len(dataloader)))
    epochs_loss         = {"train":[],"valid":[]} 
    for epoch_i in range(epochs):
        epoch_train_loss    = [] 
        epoch_test_loss     = [] 
        print(f"\tEpoch {epoch_i}")
        for i,batch in enumerate(dataloader):
            
            #Clear Grad 
            for p in model.parameters():
                p.grad  = None 

            #Calc predictions 
            position_vector         = batch[0].to(DEV) 
            moves                   = batch[1].to(DEV) 
            predicted_moves         = model.forward(position_vector) 
            #Calc loss 
            batch_loss              = model.loss_fn(predicted_moves,moves)
            batch_loss.backward() 
            epoch_train_loss.append(float(batch_loss.cpu().detach().mean()))

            #Back prop 
            model.optimizer.step()
            batch_str       = str(i).rjust(n_batches)
            if i % 25 == 0:
                print(f"\t\tbatch {batch_str}/{len(dataloader)} loss = {epoch_train_loss[-1]:.4f}")
        
        print(f"\n\t\tepoch train loss={(sum(epoch_train_loss)/len(epoch_train_loss)):.4f}")
    

        with torch.no_grad():
            for i,batch in enumerate(DataLoader(test,batch_size=bs)):
                #Calc predictions 
                position_vector_v   = batch[0].to(DEV) 
                moves_v             = batch[1].to(DEV) 
                predicted_moves_v   = model.forward(position_vector_v) 
                #Calc loss 
                batch_loss          = model.loss_fn(predicted_moves_v,moves_v)
                epoch_test_loss.append(float(batch_loss.cpu().detach().mean()))
            print(f"\t\tepoch valid loss={(sum(epoch_test_loss)/len(epoch_test_loss)):.4f}\n\n")
        epochs_loss['train'].append(sum(epoch_train_loss)/len(epoch_train_loss))
        epochs_loss['valid'].append(sum(epoch_test_loss)/len(epoch_test_loss))
    save_model(model,root=DATASET_ROOT+"/model1")
    plt.plot(epochs_loss['train'],color='green',label='Train Loss')
    plt.plot(epochs_loss['valid'],color='dodgerblue',label='Valid. Loss')
    plt.legend()
    plt.show()
        


if __name__ == "__main__":
    ds,id       = load_data(DATASET_ROOT)
    model       = networks.PolicyNet(n_ch=7,optimizer_kwargs={"lr":.0001,"weight_decay":.00001})

    print(f"loaded {len(ds)} training examples")
    train_on_data(model,ds,epochs=10)
    