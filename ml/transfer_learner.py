import chess 
import torch 
import networks 
import games 
import numpy 
import random
from torch.utils.data import DataLoader 
import time 
from model_utilities import save_model

DEV         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_legal_moves(model:networks.FullNet,bs=128,n_games=10_000,n_valid_games=20,n_iters=10,train_size=1024*32):

    for iter_i in range(n_iters):
        experiences     = [] 
        t0              = time.time()

        #Get train examples
        for _ in range(n_games):

            board   = chess.Board() 

            while not (board.is_game_over() or board.ply() > 240):
                #Get legal moves 
                legal_moves     = numpy.zeros(shape=(1968))
                legal_indices   = [games.Chess.move_to_index[m] for m in board.generate_legal_moves()]
                legal_moves[legal_indices] = 1

                experiences.append((board.fen(),legal_moves))
                board.push(games.Chess.index_to_move[random.choice(legal_indices)])
        print(f"Collected {len(experiences)} examples in {(time.time()-t0):.2f}s")


        #Train on examples
        dataset     = networks.ChessDataset(random.choices(experiences,k=train_size))
        dataloader  = DataLoader(dataset,shuffle=True,batch_size=bs)

        losses      = [] 
        loss_fn        = torch.nn.CrossEntropyLoss()
        for batch_i,batch in enumerate(dataloader):

            for p in model.parameters():
                p.grad          = None 

                input_vectors   = games.fen_to_7d_parallel(batch[0]).to(DEV)
                legal_moves     = batch[1].to(DEV)

                predicted_moves = model.forward(input_vectors)

                loss            = loss_fn(predicted_moves,legal_moves)
            loss.backward() 
            losses.append(float(loss.mean()))

            model.optimizer.step()
        
        print(f"\tEpoch Train Loss: {sum(losses)/len(losses):.6f}")

        #VALIDATION
        valid_experiences   = [] 
        for _ in range(n_valid_games):
            board   = chess.Board() 
            while not board.is_game_over():
                legal_moves     = numpy.zeros(shape=(1968))
                legal_indices   = [games.Chess.move_to_index[m] for m in board.generate_legal_moves()]
                legal_moves[legal_indices] = 1
                valid_experiences.append((board.fen(),legal_moves))
                board.push(games.Chess.index_to_move[random.choice(legal_indices)])
    
        with torch.no_grad():
            valid_losses  = []  
            vdataloader  = DataLoader(networks.ChessDataset(experiences),shuffle=True,batch_size=bs)
            for batch_i,batch in enumerate(vdataloader):
                input_vectors   = games.fen_to_7d_parallel(batch[0]).to(DEV)
                legal_moves     = batch[1].to(DEV)
                predicted_moves = model.forward(input_vectors)
                loss            = loss_fn(predicted_moves,legal_moves)
                valid_losses.append(float(loss.mean()))
        print(f"\tEpoch Valid Loss: {sum(valid_losses)/len(valid_losses):.6f}\n")
    
    save_model(model,0,1,root="C:/gitrepos/steinpy/ml/model_iter1")

if __name__ == "__main__":
    model   = networks.PolicyNetExp(n_ch=11,optimizer_kwargs={"lr":.002})
    train_legal_moves(model,bs=64,n_games=1024,n_valid_games=32,n_iters=5)