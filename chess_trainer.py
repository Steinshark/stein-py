import os 
import random 
from steinpy.ml import rl, networks 
import time 

import chess 
import torch 
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToPILImage


DEV     = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if "posix" in os.name:
    dataroot        = "/home/steinshark/data"
else:
    dataroot        = "C:/data"
do_print = print 
def print(instr):
    do_print("\t" + instr)

class ChessDataset(Dataset):

    def __init__(self,experience_set):
        self.data   = experience_set

    
    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)
    

def play_game(draw_thresh,search_iters,model=None,game_num=0):
    # model                       = model.half()
    # for layer in model.modules():
    #     if isinstance(layer,torch.nn.BatchNorm2d):
    #         layer.float()

    game_board              = chess.Board()
    MCtree                  = rl.Tree(game_board,model,draw_thresh=draw_thresh+200,device=DEV)
    state_repr              = [] 
    state_pi                = [] 
    state_outcome           = [] 
    draw_by_limit           = False 
    while not (game_board.is_checkmate() or game_board.is_stalemate() or game_board.is_seventyfive_moves() or game_board.is_fifty_moves()):
        
        #Build a local policy 
        #t0 =time.time()
        local_policy            = MCtree.get_policy(search_iters)
        #print(f"built {sum(list(local_policy.values()))} policy in {(time.time()-t0):.2f}s\n{local_policy}")
        
        #construct trainable policy 
        pi                                      = torch.zeros(1968)
        for move,prob in local_policy.items():
            pi[move]    = prob 

        #sample move from policy 
        next_move           = random.choices(list(rl.Chess.move_to_index.keys()),weights=pi,k=1)[0]

        #Add experiences to set 
        state_repr.append(MCtree.root.repr)
        state_pi.append(pi)
        game_board.push(next_move)

        #Check for draw by threshold 
        if len(state_repr) > draw_thresh:
                draw_by_limit      = True 
                break
        
        #Update tree root to node chosen 
        del MCtree.root.parent              #Save space 
        if game_board.ply() % 50 == 0:
            torch.cuda.empty_cache()
        MCtree              = rl.Tree(game_board,model,base_node=MCtree.root.children[rl.Chess.move_to_index[next_move]],draw_thresh=draw_thresh+200,lookups=MCtree.lookups,device=DEV)
    #Check game outcome 
    if not draw_by_limit and "1" in game_board.result():
        if game_board.result()[0] == "1":
            state_outcome = torch.ones(len(state_repr))
        else:
            state_outcome = torch.ones(len(state_repr)) * -1 
    else:
        state_outcome = torch.zeros(len(state_repr))

    torch.save(torch.stack(state_repr),f"{dataroot}/chess/experiences/gen1/game_{game_num}_states")
    torch.save(torch.stack(state_pi),f"{dataroot}/chess/experiences/gen1/game_{game_num}_localpi")
    torch.save(state_outcome,f"{dataroot}/chess/experiences/gen1/game_{game_num}_results")

    return game_board.ply()


def train(model:networks.FullNet,n_samples,gen,bs=8,epochs=3):
    #model = model.float()
    root                        = f"{dataroot}/chess/experiences/gen{gen}"
    experiences                 = []

    if not os.listdir(root):
        print(f"No data to train on")
        return
    max_i                       = max([int(f.split('_')[1]) for f in os.listdir(root)]) 

    for game_i in range(max_i+1):
        states                      = torch.load(f"{root}/game_{game_i}_states").float().to(DEV)
        pi                          = torch.load(f"{root}/game_{game_i}_localpi").float().to(DEV)
        results                     = torch.load(f"{root}/game_{game_i}_results").float().to(DEV)

        for i in range(len(states)):
            experiences.append((states[i],pi[i],results[i]))


    for epoch_i in range(epochs):
        train_set                   = random.sample(experiences,min(n_samples,len(experiences)))

        dataset                     = ChessDataset(train_set)
        dataloader                  = DataLoader(dataset,batch_size=bs,shuffle=True)
        
        total_loss                  = 0 
        
        for batch_i,batch in enumerate(dataloader):
            
            #Zero Grad 
            for p in model.parameters():
                p.grad                      = None

            #Get Data
            states                      = batch[0].to(torch.device('cuda'))
            pi                          = batch[1].to(torch.device('cuda'))
            outcome                     = batch[2].to(torch.device('cuda'))
            batch_len                   = len(states)

            #Get model predicitons
            pi_pred,v_pred              = model.forward(states)
            
            #Calc model loss 
            loss                        = torch.nn.functional.mse_loss(v_pred.view(-1),outcome) + torch.nn.functional.binary_cross_entropy(pi_pred,pi)
            total_loss                  += loss.mean().item()
            loss.backward() 

            #Backpropogate
            model.optimizer.step()
        
        print(f"\t\tEpoch {epoch_i} loss: {total_loss:.3f} with {len(train_set)}")


def save_model(model:networks.FullNet,gen=1):
    torch.save(model.state_dict(),f"{dataroot}/chess/models/gen{gen}")


def load_model(model:networks.FullNet,gen=1):
    try:
        model.load_state_dict(torch.load(f"{dataroot}/chess/models/gen{gen}"))
    except FileNotFoundError:
        print(f"Model gen{gen} not found, loading gen1")

def write_to_db(obj,path):

    #Ensure path exists 
    base_path   = "//FILESERVER/S Drive/"
    folders     = path.split("/")
    for folder in folders[:-1]:
        base_path = base_path + folder + "/"
    
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        
    base_path = base_path + folders[-1]

    torch.save(obj,base_path)

def read_from_db(obj,path):
    return torch.load(path)
    

        


    
#Optimizers 
#torch.backends.cudnn.benchmark = True

#DEFINE TRAINING PARAMETERS 
model                       = networks.ChessNet(optimizer_kwargs={"lr":1e-5,"weight_decay":1e-6},n_ch=19,device=DEV)
training_iters              = 5
games_per_iter              = 5
draw_thresh                 = 250
search_iters                = 150
load_model(model)
train(model,1024,1,bs=32,epochs=3)
for iter in range(training_iters):

    print(f"Training Iter {iter}")
    #Play out 'games_per_iter' games, then train on them  
    for game_num in range(games_per_iter):
        t0 = time.time()
        moves = play_game(draw_thresh,search_iters,model=model,game_num=game_num)
        print(f"\tfinished {moves}\tgame: {game_num}\ttook {(time.time()-t0):.2f}s")
    print(f"\tTraining")
    train(model,1024,1,bs=32)
    print(f"\n")
    rl.Chess.lookup = {}
    save_model(model)
print(f"Saved Model")
            


