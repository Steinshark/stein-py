import torch 
import networks 
import time 
import multiprocessing
import random 
import os 
import games 
import sys 
sys.path.append("C:/gitrepos/")
from steinpy.ml.rl_torch import Tree 
import numpy 
from sklearn.utils import extmath 
from torch.utils.data import DataLoader
import string 
import pickle 


def softmax(x):
		if len(x.shape) < 2:
			x = numpy.asarray([x],dtype=float)
		return extmath.softmax(x)[0]

class Color:
	HEADER = '\033[95m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	RED = '\033[91m'
	END = '\033[0m'
	TAN = '\033[93m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'



def save_model(model:networks.FullNet,gen=1,count=1,root=r"\\FILESERVER\S Drive\Data\chess"):
    torch.save(model.state_dict(),root)

    try:
        state_dict 	 = torch.load(root)
        model.load_state_dict(state_dict)
    except RuntimeError as re:
        if count > 5:
            print(f"{Color.RED}\tFailed to save model: {gen}{Color.END}")
        else:
            time.sleep(.1)
            save_model(model,gen,count+1)


def get_generations(DATASET_ROOT=r"\\FILESERVER\S Drive\Data\chess"):
    gens 	= [] 
    for model in os.listdir(DATASET_ROOT+f"\models\\"):
        gens.append(int(model.replace("gen","")))

    return gens 


def get_n_games(DATASET_ROOT=r"\\FILESERVER\S Drive\Data\chess",gen=0):
    files 	= os.listdir(DATASET_ROOT+f"/experiences/{gen}")


def load_model(model:networks.FullNet,gen=1,verbose=False,DATASET_ROOT=r"\\FILESERVER\S Drive\Data\chess"):
    while True:
        try:
            model.load_state_dict(torch.load(DATASET_ROOT+f"\models\gen{gen}"))
            if verbose:
                print(f"\tloaded model gen {gen}")
            return 
        
        except FileNotFoundError:
            gen -= 1 
            if gen <= 0:
                print(f"\tloaded stock model gen {gen}")
                return


def play_models(cur_model_net:torch.nn.Module,challenger_model_net:torch.nn.Module,search_depth,max_moves):

    t0 						= time.time() 
    
    game_board 				= games.Chess(max_moves=max_moves)
    
    move_indices            = list(range(game_board.move_space))
    state_repr              = [] 
    state_pi                = [] 
    state_outcome           = [] 
    n 						= 1

    while game_board.get_result() is None:

        #cur_model_net MOVE 
        #Build a local policy 
        if n == 1:
            model 	= cur_model_net
        else:
            model 	= challenger_model_net
        mcts_tree1 				= Tree(game_board,model)
        local_policy 			= mcts_tree1.update_tree(iters=search_depth)
        local_softmax 			= softmax(numpy.asarray(list(local_policy.values()),dtype=float))
        for key,prob in zip(local_policy.keys(),local_softmax):
            local_policy[key] 		= prob
        #construct trainable policy 
        pi              		= numpy.zeros(game_board.move_space)
        for move_i,prob in local_policy.items():
            pi[move_i]    			= prob 
        #sample move from policy 
        next_move             	= random.choices(move_indices,weights=pi,k=1)[0]

        #Add experiences to set 
        state_repr.append(game_board.get_repr())
        state_pi.append(pi)
        game_board.make_move(next_move)

        #Update MCTS tree 
        # child_node 				= mcts_tree.root.children[next_move_i]

        #Release references to other children
        #del mcts_tree.root.parent
        game_board.is_game_over()
    

    return game_board.get_result()
  

def duel(available_models,n_games,search_depth,cur_model=0,max_moves=120,DATASET_ROOT=r"\\FILESERVER\S Drive\Data\chess"):
    print(f"\t{Color.TAN}DUELING{Color.END}")
    best_model 				= cur_model
    challenger_model		= cur_model

    #Pick a random, past model
    while challenger_model == cur_model:
        challenger_model 		= random.choice(available_models)  	
    worst_model					= challenger_model

    print(F"\t{Color.TAN}Cur Best {cur_model} vs. Model {challenger_model}")

    #Load models 
    cur_model_net 			= networks.ChessSmall()
    challenger_model_net 	= networks.ChessSmall()
    load_model(cur_model_net,gen=cur_model,verbose=True,DATASET_ROOT=DATASET_ROOT)
    load_model(challenger_model_net,gen=challenger_model,verbose=True,DATASET_ROOT=DATASET_ROOT) 
    cur_model_net.eval()
    challenger_model_net.eval()

    #Keep track of how each model does
    current_best_games 		= 0
    challenger_games 		= 0 
    tiegames    			= 0

    #Play cur_model_net as X 
    for game_i in range(n_games):
        result 	= play_models(cur_model_net,challenger_model_net,search_depth=search_depth,max_moves=max_moves)

        if result == 1:
            current_best_games += 1 
        elif result == -1:
            challenger_games += 1
        elif result == 0:
            tiegames += 1

    #Play challenger_model_net as X 
    for game_i in range(n_games):
        result 	= play_models(challenger_model_net,cur_model_net,search_depth=search_depth,max_moves=max_moves)

        if result == 1:
            challenger_games += 1 
        elif result == -1:
            current_best_games += 1
        elif result == 0:
            tiegames += 1
    
    challenger_ratio 	= ((challenger_games) / (current_best_games+challenger_games+.01))

    if challenger_ratio >= .55:
        best_model 			= challenger_model
        worst_model			= cur_model

    


    print(f"\t{Color.GREEN}Cur model{cur_model}: {current_best_games}\tChallenger model{challenger_model}: {challenger_games}\ttie: {tiegames}\n")

    #Delete worst model 
    print(f"\t{Color.GREEN}best model is {best_model}{Color.END}")
    print(f"\t{Color.RED}removing {worst_model}{Color.END}")
    os.remove(DATASET_ROOT+f"\\models\\gen{worst_model}")
    cur_model 	= best_model

    return cur_model


def duel_muiltithread(available_models,n_games,max_moves,search_depth,cur_model=0,n_threads=4):
    print(f"\t{Color.TAN}DUELING{Color.END}")
    best_model 				= cur_model
    challenger_model		= cur_model

    #Pick a random, past model
    while challenger_model == cur_model:
        challenger_model 		= random.choice(available_models)  	
    worst_model					= challenger_model

    print(F"\t{Color.TAN}Cur Best {cur_model} vs. Model {challenger_model}")

    #Load models 
    cur_model_net 			= networks.ChessSmall()
    challenger_model_net 	= networks.ChessSmall()
    load_model(cur_model_net,gen=cur_model,verbose=True)
    load_model(challenger_model_net,gen=challenger_model,verbose=True) 
    cur_model_net.eval()
    challenger_model_net.eval()

    #Keep track of how each model does
    current_best_games 		= 0
    challenger_games 		= 0 
    tiegames    			= 0

    #Play cur_model_net as White
    args 		= [(cur_model_net,challenger_model_net,search_depth,max_moves) for _ in range(n_games)]
    with multiprocessing.Pool(n_threads) as pool:
        results 	= pool.starmap(self.play_models,args)

    for result in results:
        if result == 1:
            current_best_games += 1 
        elif result == -1:
            challenger_games += 1
        elif result == 0:
            tiegames += 1

    #Play challenger_model_net as White
    args 		= [(challenger_model_net,cur_model_net,search_depth,max_moves) for _ in range(n_games)]
    with multiprocessing.Pool(n_threads) as pool:
        results 	= pool.starmap(self.play_models,args)
    for result in results:
        if result == 1:
            challenger_games += 1 
        elif result == -1:
            current_best_games += 1
        elif result == 0:
            tiegames += 1
    
    challenger_ratio 	= ((challenger_games) / (current_best_games+challenger_games+.01))

    if challenger_ratio >= .55:
        best_model 			= challenger_model
        worst_model			= cur_model

    


    print(f"\t{Color.GREEN}Cur model{cur_model}: {current_best_games}\tChallenger model{challenger_model}: {challenger_games}\ttie: {tiegames}\n")

    #Delete worst model 
    print(f"\t{Color.GREEN}best model is {best_model}{Color.END}")
    print(f"\t{Color.RED}removing {worst_model}{Color.END}")
    os.remove(self.DATASET_ROOT+f"\\models\\gen{worst_model}")
    self.cur_model 	= best_model


def train(model,n_samples=2048,bs=32,epochs=2,DEV=torch.device('cuda' if torch.cuda.is_available else 'cpu'),DATASET_ROOT=r"\\FILESERVER\S Drive\Data\chess",gen=0):
    model 						= model.float()
    root                        = DATASET_ROOT+f"\experiences\gen{gen}"
    experiences                 = []
    model.train()


    if not os.listdir(root):
        print(f"No data to train on")
        return


    print(f"{Color.TAN}\t\tbegin Training:{Color.END}",end='')
    game_ids 	= set()
    for file in os.listdir(root):
        for letter in string.ascii_lowercase+string.ascii_uppercase+"_":
            file = file.replace(letter,"")
        
        game_ids.add(int(file))


    for game_i in game_ids:
        try:
            states                      = torch.load(f"{root}/game_{game_i}_states").float().to(DEV)
            pi                          = torch.load(f"{root}/game_{game_i}_localpi").float().to(DEV)
            results                     = torch.load(f"{root}/game_{game_i}_results").float().to(DEV)
            for i in range(len(states)):
                experiences.append((states[i],pi[i],results[i]))
        except FileNotFoundError:
            pass 
        except RuntimeError:
            pass
        except pickle.UnpicklingError:
            pass

    
    print(f"{Color.TAN}\tloaded {len(experiences)} datapoints in gen {gen}{Color.END}")

    if len(experiences) == 0:
        print(f"No data to train on")
        return
    

    for epoch_i in range(epochs):
        train_set                   = random.sample(experiences,min(n_samples,len(experiences)))

        dataset                     = networks.ChessDataset(train_set)
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
            loss                        = torch.nn.functional.mse_loss(v_pred.view(-1),outcome,reduction='mean') + torch.nn.functional.cross_entropy(pi_pred,pi,)
            total_loss                  += loss.mean().item()
            loss.backward() 

            #Backpropogate
            model.optimizer.step()
        
        print(f"\t\t{Color.BLUE}Epoch {epoch_i} loss: {total_loss/batch_i:.3f} with {len(train_set)}/{len(experiences)}{Color.END}")

    print(f"\n")
    return model 

