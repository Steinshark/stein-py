import sys
sys.path.append("C:\steincode\steinpy\ml")
import networks 
from networks import ChessDataset,PolicyNetSm
import chess 
import numpy 
from games import fen_to_7d_parallel, Chess, fen_to_7d
import torch 
from torch.utils.data import DataLoader
import random 
import copy 
import multiprocessing
import time 

#TODO 
#   function for tensor representation 
#   feed-forward and choose a move 
#   store training data 

def normalize_weights(weights:numpy.array):

    #Correct all weights to be positive

    if min(weights) < 0:
        positive_shift  = abs(min(weights))
        weights         += positive_shift

    return numpy.array(weights)/max(weights)


def get_legal_move(board:chess.Board,position_eval:numpy.ndarray,ε=.05,τ=5):

    #Find legal indices  
    legal_move_indices      = [Chess.move_to_index[move] for move in board.generate_legal_moves()]

    if len(legal_move_indices) == 1:
        return Chess.index_to_move[legal_move_indices[0]],legal_move_indices
    
    #Epsilon-greedy
    if random.random() < ε:
        legal_move_values       = numpy.take(position_eval,legal_move_indices)
        return Chess.index_to_move[random.choices(legal_move_indices,normalize_weights(legal_move_values)**τ,k=1)[0]],legal_move_indices

    
    else:
        return Chess.index_to_move[max(legal_move_indices,key=lambda i:position_eval[i])],legal_move_indices 


def generate_training_games(model:networks.FullNet,n_games=16,max_ply=320):

    model.eval()
    chess_boards        = [chess.Board() for _ in range(n_games)]
    for i,board in enumerate(chess_boards):
        board.is_active     = True  
        board.game_id       = i 

    experiences         = []
    game_outcomes       = [0 for _ in chess_boards]
    glm                 = get_legal_move
    ft7d                = fen_to_7d_parallel

    while True in [board.is_active for board in chess_boards]:
        
        #Grab active games
        current_boards      = [board for board in chess_boards if board.is_active]

        #Get evals with network forward pass
        with torch.no_grad():
            position_reprs      = ft7d(set(board.fen() for board in current_boards),req_grad=True)
            position_evals      = model.forward(position_reprs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))).cpu().numpy()

        #Chose the top legal move 
        top_legal_moves     = [glm(board,eval) for board,eval in zip(current_boards,position_evals)]

        #Save the current repr and position eval
        experiences += list(zip(current_boards,[999 for _ in current_boards],[Chess.move_to_index[tlm[0]] for tlm in top_legal_moves],[tlm[1] for tlm in top_legal_moves]))

        #Make move 
        [board.push(move) for board,move in zip(current_boards,[tlm[0] for tlm in top_legal_moves])]

        #Remove game overs 
        for board in current_boards:
            if (board.is_game_over() or (board.ply() > max_ply)):
                board.is_active     = False 
                board.game_result   = 1 if board.result() == "1-0" else -1 if board.result() == "0-1" else 0

    #Fill in experiences 
    for i in range(len(experiences)):
        #                            cur board obj  -       outcome                  -  chosen_index   -     legal_moves
        board                   = experiences[i][0]
        experiences[i]          = (board.fen(),board.game_result,experiences[i][2],experiences[i][3])

    #print(f"played {n_games}")
    model.train()
    return experiences


def eval_loss(turn:str,predicted_moves:torch.Tensor,final_outcome,chosen_move_i,legal_moves_i,mode="reinforce"):
    turn            = [{"w":1,"b":-1}[t] for t in turn]

    actual_moves    = predicted_moves.detach().clone()

    if mode == "reinforce":
        for i,pkg  in enumerate(zip(final_outcome,chosen_move_i)):
            out,move_i = pkg 
            actual_moves[i,move_i]     = out
    
    elif mode == "random":
        for i,pkg in enumerate(zip(turn,final_outcome,chosen_move_i,legal_moves_i)):
            out,i,legals = pkg 

            if not turn == final_outcome:
                rands   = torch.randn(size=(1,len(legal_moves_i))).numpy()
                for rand,legal_i in zip(rands,legal_moves_i):
                    actual_moves[i,legal_i] = rand 
    else:
        raise NotImplementedError(f"Mode {mode} no implemented in eval_loss")
    
    return actual_moves.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                



           

        

    return 0


def collate_fn(data):
    #4 items 

    return [d[0] for d in data],[d[1] for d in data],[d[2] for d in data],[d[3] for d in data]


def train(model:networks.FullNet,dataset:ChessDataset,bs=32):
    model.train()
    #Create DataLoader 
    dataloader          = DataLoader(dataset,batch_size=bs,shuffle=True,collate_fn=collate_fn)
    loss_fn             = torch.nn.MSELoss()


    #Iterate over batches 
    #print(f"Training on {len(dataloader)} batches")
    losses      = [] 
    for batch_i,batch in enumerate(dataloader):
        #print(f"\tbatch {batch_i}/{len(dataloader)}")
        #CLEAR GRAD 
        for p in model.parameters():
            p.grad          = None 
        

        game_boards:str         = batch[0]
        final_outcomes:int      = batch[1]
        chosen_move_is:int      = batch[2]
        legal_indices:list      = batch[3]

        predicted_moves         = model.forward(fen_to_7d_parallel([fen for fen in game_boards]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))


        #Get model prediction   = 
        loss                    = loss_fn(predicted_moves,eval_loss([fen.split(" ")[1] for fen in game_boards],predicted_moves,final_outcomes,chosen_move_is,legal_indices,mode="reinforce"))
        loss.backward() 
        losses.append(loss.item())

        model.optimizer.step()
    print(f"\ttrain loss: {sum(losses)/len(losses)}")






    model.eval()
    return model 
        #Implement training alg 


def train_model(model,n_iters,n_games,max_ply):
    for _ in range(n_iters):
        training_data   = generate_training_games(model,n_games=n_games,max_ply=max_ply)
        train(model,training_data)
    return model 


def duel_models(model_w,model_b,max_ply=320):
    model_w.eval()
    model_b.eval()
    board           = chess.Board() 
    cur_model       = model_w

    while (not board.is_game_over()) and (board.ply() < max_ply):

        with torch.no_grad():
            next_move       = get_legal_move(board,cur_model.forward(fen_to_7d_parallel([board.fen()],req_grad=False).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))).cpu().numpy()[0],ε=.9,τ=1)[0]
        board.push(next_move)
        if cur_model == model_w:
            cur_model       = model_b
        else:
            cur_model       = model_w


    if board.result()     == "1-0":
        return "w"
    elif board.result()   == "0-1":
        return "b"
    else:
        return "draw"
    
    



if __name__ == "__main__":
    model       = PolicyNetSm(n_ch=11,optimizer_kwargs={"lr":.0001,"weight_decay":.00001})
    bad_model   = PolicyNetSm(n_ch=11)
    iter_games  = 128
    n_iters     = int(1024/iter_games)

    for _ in range(8):

        if _ % 4 == 0:
            print(f"run iter {_}")

        l_dataset   = [] 
        for j in range(4):
            t0 = time.time()
            l_dataset += generate_training_games(model,iter_games,320)
        model   = train(model,l_dataset,bs=512)

    good_wins   = 0
    bad_wins    = 0
    draws       = 0
    n_test_games    = 100
    for i in range(n_test_games):
        res     = duel_models(model,bad_model)
        if res  == "w":
            good_wins += 1 
        elif res == "b":
            bad_wins += 1 
        else:
            draws += 1  
    for i in range(n_test_games):
        res     = duel_models(bad_model,model)
        if res  == "b":
            good_wins += 1 
        elif res == "w":
            bad_wins += 1 
        else:
            draws += 1  
