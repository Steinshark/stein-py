import os 
import random 
from steinpy.ml import rl, networks 
import time 

import chess 
import torch 
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToPILImage
import multiprocessing
from matplotlib import pyplot as plt 

draw_thresh         = 250
search_iters        = 750
DEV                 = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
softmax             = torch.nn.Softmax(dim=0)
global_game_count   = 0
torch.set_printoptions(threshold=10000)
class ChessDataset(Dataset):

    def __init__(self,experience_set):
        self.data   = experience_set

    
    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)
    

def play_game(draw_thresh=250,search_iters=400,model=None,game_num=0,half_precision=False,abbrev=False):
    if half_precision:
        model                       = model.half()
        for layer in model.modules():
            if isinstance(layer,torch.nn.BatchNorm2d):
                layer.float()

    total_time              = 0 
    game_board              = chess.Board()
    MCtree                  = rl.Tree(game_board,model,draw_thresh=draw_thresh,device=DEV)
    state_repr              = [] 
    state_pi                = [] 
    state_outcome           = [] 
    draw_by_limit           = False 
    move_indices            = list(range(1968))
    while not (game_board.is_checkmate() or game_board.is_stalemate() or game_board.is_seventyfive_moves() or game_board.is_fifty_moves()):
        #Build a local policy 
        t0 =time.time()
        local_policy            = MCtree.get_policy(search_iters,abbrev=abbrev)
        total_time += time.time()-t0
        
        #construct trainable policy 
        pi                                      = torch.ones(1968) * float("-inf")
        for move_i,prob in local_policy.items():
            pi[move_i]    = prob 

        pi                      = softmax(pi)
        #sample move from policy 
        next_move_i             = random.choices(move_indices,weights=pi,k=1)[0]
        next_move               = rl.Chess.index_to_move[next_move_i]

        #Add experiences to set 
        if MCtree.root.repr is None:
            state_repr.append(rl.Chess.fen_to_tensor(MCtree.root.board,DEV))
        else:
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
        MCtree              = rl.Tree(game_board,model,base_node=MCtree.root.children[next_move_i],draw_thresh=draw_thresh,device=DEV)
    #Check game outcome 
    if not draw_by_limit and "1" in game_board.result():
        if game_board.result()[0] == "1":
            state_outcome = torch.ones(len(state_repr))
        else:
            state_outcome = torch.ones(len(state_repr)) * -1 
    else:
        state_outcome = torch.zeros(len(state_repr))
    
    #Save tensors
    torch.save(torch.stack(state_repr).float(),f"C:/data/chess/experiences/gen1/game_{game_num}_states")
    torch.save(torch.stack(state_pi).float(),f"C:/data/chess/experiences/gen1/game_{game_num}_localpi")
    torch.save(state_outcome.float(),f"C:/data/chess/experiences/gen1/game_{game_num}_results")

    if half_precision:
        model               = model.float()
    return game_board.ply()


def play_game_thread(model):
    global global_game_count
    state_repr              = [] 
    state_pi                = [] 
    state_outcome           = [] 
    draw_by_limit           = False 
    move_indices            = list(range(1968))
      
    game_board              = chess.Board()
    MCtree                  = rl.Tree(game_board,model,draw_thresh=draw_thresh,device=DEV)
    
    t0 = time.time()
    while not (game_board.is_checkmate() or game_board.is_stalemate() or game_board.is_seventyfive_moves() or game_board.is_fifty_moves()):
        #Build a local policy 
        #t0 =time.time()
        local_policy            = MCtree.get_policy(search_iters)
        #print(f"built {sum(list(local_policy.values()))} policy in {(time.time()-t0):.2f}s\n{[(rl.Chess.chess_moves[k],v) for k,v in local_policy.items()]}")
        
        #construct trainable policy 
        pi                                      = torch.ones(1968) * float("-inf")
        for move_i,prob in local_policy.items():
            pi[move_i]    = prob 

        pi                      = softmax(pi)
        #sample move from policy 
        next_move_i             = random.choices(move_indices,weights=pi,k=1)[0]
        next_move               = rl.Chess.index_to_move[next_move_i]

        #Add experiences to set 
        if MCtree.root.repr is None:
            state_repr.append(rl.Chess.fen_to_tensor(MCtree.root.board,DEV))
        else:
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
        MCtree              = rl.Tree(game_board,model,base_node=MCtree.root.children[next_move_i],draw_thresh=draw_thresh,device=DEV)
    #Check game outcome 
    if not draw_by_limit and "1" in game_board.result() and not "/" in game_board.result():
        if game_board.result()[0] == "1":
            state_outcome = torch.ones(len(state_repr))
        else:
            state_outcome = torch.ones(len(state_repr)) * -1 
    else:
        state_outcome = torch.zeros(len(state_repr))
    
    #Save tensors
    # torch.save(torch.stack(state_repr).float(),f"C:/data/chess/experiences/gen1/game_{game_num}_states")
    # torch.save(torch.stack(state_pi).float(),f"C:/data/chess/experiences/gen1/game_{game_num}_localpi")
    # torch.save(state_outcome.float(),f"C:/data/chess/experiences/gen1/game_{game_num}_results")

    print(f"\t\t\tgame ran {game_board.ply()} moves in {(time.time()-t0):.2f}s\t- result [{game_board.result().replace('1/2','-')}]\t[{state_outcome[-1].item()}]")
    global_game_count += 1
    return (state_repr,state_pi,state_outcome)


def train(model:networks.FullNet,n_samples,gen,bs=8,epochs=5):
    model = model.float()
    root                        = f"C:/data/chess/experiences/gen{gen}"
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
            loss                        = (torch.nn.functional.mse_loss(v_pred.view(-1),outcome) + torch.nn.functional.cross_entropy(pi_pred,pi)).mean()
            total_loss                  += loss.mean().item()
            loss.backward() 

            #Backpropogate
            model.optimizer.step()
        
        print(f"\t\tEpoch {epoch_i} loss: {total_loss:.3f} with {len(train_set)}")


def save_model(model:networks.FullNet,gen=1):
    torch.save(model.state_dict(),f"C:/data/chess/models/gen{gen}")


def load_model(model:networks.FullNet,gen=1):
    model.load_state_dict(torch.load(f"C:/data/chess/models/gen{gen}"))
    print(f"\tloaded model")


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


def run_training(n_threads=6,training_iters=5,n_iters=10):
    #Optimizers 
    #torch.backends.cudnn.benchmark = True

    #DEFINE TRAINING PARAMETERS 
    model                       = networks.ChessNet(optimizer_kwargs={"lr":1e-5,"weight_decay":1e-6},n_ch=19,device=DEV)


    torch.backends.cudnn.benchmark = True
    #load_model(model)
    #train(model,1024,1,bs=32,epochs=3)


    ###############################################################################################
    #                                   MAIN TRAINING LOOP 
    ###############################################################################################
    for train_num in range(training_iters):
        print(f"\tBEGIN TRAIN ITER {train_num} /{training_iters}")

        #Run n_threads number of games x iter_num 
        t0 = time.time()
        load_model(model)
        print(f"\t\tStarting Set of {n_iters} Games")
        #Run all games simultaneously 
        with multiprocessing.Pool(processes=n_threads) as pool:
            experiences = pool.map(play_game_thread,(model for _ in range(n_iters)))
        
        #Save batch of games
        for game_num,tup in enumerate(experiences):
            state_repr,state_pi,state_outcome = tup
            state_outcome[0] = 0
            torch.save(torch.stack(state_repr).float(),f"C:/data/chess/experiences/gen1/game_{game_num}_states")
            torch.save(torch.stack(state_pi).float(),f"C:/data/chess/experiences/gen1/game_{game_num}_localpi")
            torch.save(state_outcome.float(),f"C:/data/chess/experiences/gen1/game_{game_num}_results")

    
        print(f"\t\t\tCompleted Games in {(time.time()-t0):.2f}s\n\n")
        train(model,1024,1,bs=64,epochs=5)
        save_model(model)
        print(f"\n")


def play_random(model,search_iters=750,draw_thresh=250):
    global global_game_count
    draw_by_limit           = False 
    move_indices            = list(range(1968))
    play_as                 = 1 
    game_board              = chess.Board()
    MCtree                  = rl.Tree(game_board,model,draw_thresh=draw_thresh,device=DEV)
    moves                   = "me"
    t0 = time.time()
    while not (game_board.is_checkmate() or game_board.is_stalemate() or game_board.is_seventyfive_moves() or game_board.is_fifty_moves()):

        if play_as == game_board.turn:
            #Build a local policy 
            local_policy            = MCtree.get_policy(search_iters)
            
            #construct trainable policy 
            pi                                      = torch.ones(1968) * float("-inf")
            for move_i,prob in local_policy.items():
                pi[move_i]    = prob 

            pi                      = softmax(pi)
            #sample move from policy 
            next_move_i             = random.choices(move_indices,weights=pi,k=1)[0]
            next_move               = rl.Chess.index_to_move[next_move_i]
            game_board.push(next_move)

        
            
            #Update tree root to node chosen 
            del MCtree.root.parent              #Save space 
            if game_board.ply() % 50 == 0:
                torch.cuda.empty_cache()

            MCtree              = rl.Tree(game_board,model,base_node=MCtree.root.children[next_move_i],draw_thresh=draw_thresh,device=DEV)
            moved               = "me"

        else:
            rand_move           = random.choice(list(game_board.generate_legal_moves()))
            rand_move_i         = rl.Chess.move_to_index[rand_move]

            game_board.push(rand_move)
            if rand_move_i in MCtree.root.children:
                MCtree              = rl.Tree(game_board,model,base_node=MCtree.root.children[rand_move_i])
            else:
                MCtree              = rl.Tree(game_board,model,base_node=None)
            moved               = "them"

        #Check for draw by threshold 
        if game_board.ply() > draw_thresh:
                break
    
    if not draw_by_limit and "1" in game_board.result() and not "/" in game_board.result():
        if moved == "me":
            res = 1
        else:
            res = -1
    else:
        res = 0

    return res


def play_model(model1,model2,search_iters,draw_thresh):
    global global_game_count
    draw_by_limit           = False 
    move_indices            = list(range(1968))
    play_as                 = 1 
    game_board              = chess.Board()
    moves                   = "me"
    t0 = time.time()
    MCtree1                 = rl.Tree(game_board,model1,base_node=None,draw_thresh=draw_thresh,device=DEV)
    MCtree2                 = rl.Tree(game_board,model1,base_node=None,draw_thresh=draw_thresh,device=DEV)
    
    while 1:

        #########################################################################################
        #                               MODEL  1
        if game_board.ply() > 1 and next_move_i in MCtree1.root.children:
            MCtree1                 = rl.Tree(game_board,model1,base_node=MCtree1.root.children[next_move_i],draw_thresh=draw_thresh,device=DEV)
        else:
            MCtree1                 = rl.Tree(game_board,model1,base_node=None,draw_thresh=draw_thresh,device=DEV)
        #Build a local policy 
        local_policy            = MCtree1.get_policy(search_iters)
        
        #construct trainable policy 
        pi                                      = torch.ones(1968) * float("-inf")
        for move_i,prob in local_policy.items():
            pi[move_i]    = prob 

        pi                      = softmax(pi)
        #sample move from policy 
        next_move_i             = random.choices(move_indices,weights=pi,k=1)[0]
        next_move               = rl.Chess.index_to_move[next_move_i]
        game_board.push(next_move)
        MCtree1                 = rl.Tree(game_board,model1,base_node=MCtree1.root.children[next_move_i],draw_thresh=draw_thresh,device=DEV)
        #Update tree root to node chosen 
        del MCtree1.root.parent              #Save space 
        moved               = "1"

        if game_board.is_checkmate() or game_board.is_stalemate() or game_board.is_seventyfive_moves() or game_board.is_fifty_moves():
            break


        #########################################################################################
        #                               MODEL  2
        #Build a local policy 
        if next_move_i in MCtree2.root.children:
            MCtree2                 = rl.Tree(game_board,model2,base_node=MCtree2.root.children[next_move_i],draw_thresh=draw_thresh,device=DEV)
        else:
            MCtree2                 = rl.Tree(game_board,model2,base_node=None,draw_thresh=draw_thresh,device=DEV)
        local_policy            = MCtree2.get_policy(search_iters)
        
        #construct trainable policy 
        pi                                      = torch.ones(1968) * float("-inf")
        for move_i,prob in local_policy.items():
            pi[move_i]    = prob 

        pi                      = softmax(pi)
        #sample move from policy 
        next_move_i             = random.choices(move_indices,weights=pi,k=1)[0]
        next_move               = rl.Chess.index_to_move[next_move_i]
        game_board.push(next_move)
        MCtree2                 = rl.Tree(game_board,model2,base_node=MCtree2.root.children[next_move_i],draw_thresh=draw_thresh,device=DEV)
        #Update tree root to node chosen 
        
        
        del MCtree2.root.parent              #Save space 
        if game_board.ply() % 50 == 0:
            torch.cuda.empty_cache()

        if game_board.is_checkmate() or game_board.is_stalemate() or game_board.is_seventyfive_moves() or game_board.is_fifty_moves():
            break
    
    if not draw_by_limit and "1" in game_board.result() and not "/" in game_board.result():
        if moved == "1":
            res = "model1"
        else:
            res = "model2"
    else:
        res = "draw"

    return res


def arena(model,n_games=5,search_iters=400,draw_thresh=250):

    won         = 0 
    lost        = 0 
    draw        = 0 

    with multiprocessing.Pool(6) as pool:
        results = pool.map(play_random,[model for _ in range(n_games)])
        
        for result in results:
            if result == 1:
                won += 1 
            elif result == -1:
                lost += 1
            else:
                draw += 1
        
    print(f"\t[won:{won},\tlost:{lost},\tdraw:{draw}]")


def model_arena(model1,model2,n_games):
    
    model1_w      = 0 
    model2_w      = 0 
    draw        = 0 

    with multiprocessing.Pool(6) as pool:
        results     = pool.starmap(play_model,[(model1,model2,100,5) for _ in range(n_games)])

        for result in results:
            if result == "model1":
                model1_w += 1 
            elif result == "model2":
                model2_w += 1
            else:
                draw += 1
    print(f"\t[model1:{model1_w},\tmodel2:{model2_w},\tdraw:{draw}]")


if __name__ == "__main__":
    if False:
        times = {"abbrev":[],"full":[]}
        #Optimizers 
        #torch.backends.cudnn.benchmark = True

        #DEFINE TRAINING PARAMETERS 
        model                       = networks.ChessNet(optimizer=torch.optim.SGD,optimizer_kwargs={"lr":1e-4,"weight_decay":1e-5,"momentum":.9},n_ch=19,device=DEV)
        training_iters              = 1
        games_per_iter              = 3
        draw_thresh                 = 50
        search_iters                = 20
        torch.backends.cudnn.benchmark = True
        #load_model(model)
        #train(model,1024,1,bs=32,epochs=3)

        for abbrev in [True,False]:
            for iter in range(training_iters):

                #print(f"Training Iter {iter}")
                #Play out 'games_per_iter' games, then train on them  
                for game_num in range(games_per_iter):
                    t0 = time.time()
                    moves = play_game(draw_thresh,search_iters,model=model,game_num=game_num)
                    if abbrev:
                        times["abbrev"].append((time.time()-t0)/moves)
                    else:
                        times["full"].append((time.time()-t0)/moves)
                #print(f"\tTraining")
                #train(model,1024,1,bs=32)
                #print(f"\n")
                
                #save_model(model)
            rl.Chess.lookup_table = dict()
        plt.plot(times["abbrev"],color="dodgerblue",label="Abbrev")
        plt.plot(times["full"],color="darkorange",label="Full")
        plt.show()
    if True:

        model                       = networks.ChessNet(optimizer=torch.optim.SGD,optimizer_kwargs={"lr":1e-4,"weight_decay":1e-5,"momentum":.9},n_ch=19,device=DEV)
        #load_model(model1)
        #model_arena(model1,model2,10)
        run_training(n_threads=6,training_iters=5,n_iters=20)
