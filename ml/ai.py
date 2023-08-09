import chess
import chess.svg
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json 
import time 
from networks import ChessDataset
from torch.utils.data import DataLoader
class ChessNeuralNetwork(nn.Module):
    def __init__(self):
        super(ChessNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        
        # Move probabilities head
        self.move_probs = nn.Linear(512, 1968)  # Adjust the output dimension
        
        # Value head
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        x = F.relu(self.fc1(x))
        
        move_probs = self.move_probs(x)
        value = self.value(x)
        
        return F.softmax(move_probs, dim=1), torch.tanh(value)


class SelfTeachingChessAI:
    def __init__(self):
        lr=.0001

        self.board = chess.Board()
        self.neural_network = ChessNeuralNetwork().to(torch.device('cuda'))
        self.optimizer = optim.SGD(self.neural_network.parameters(), lr=lr,momentum=.9,weight_decay=lr/10)
        self.criterion = nn.CrossEntropyLoss()
        self.games_played = 0
        self.chess_moves 		= json.loads(open("C:/gitrepos/steinpy/ml/res/chessmoves.txt","r").read())
        self.trained_moves  = 0 
    def value_criterion(self,prediction, target):
        penalty_factor = 2.0 if target < 0 else 1.0
        huber_loss = F.smooth_l1_loss(prediction, target)
        return huber_loss * penalty_factor
    
    def encode_board(self):
        board_tensor = np.zeros(shape=(6, 8, 8),dtype=float)
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                piece_value = piece.piece_type
                if piece.color == chess.BLACK:
                    piece_value = -piece_value
                board_tensor[0, square // 8, square % 8] = piece_value
        
        # Encode castling rights and turn
        board_tensor[1, :, :] = 1 if self.board.has_kingside_castling_rights(chess.WHITE) else -1
        board_tensor[2, :, :] = 1 if self.board.has_queenside_castling_rights(chess.WHITE) else -1
        board_tensor[3, :, :] = 1 if self.board.has_kingside_castling_rights(chess.BLACK) else -1
        board_tensor[4, :, :] = 1 if self.board.has_queenside_castling_rights(chess.BLACK) else -1
        board_tensor[5, :, :] = 1 if self.board.turn == chess.WHITE else -1
    
        return torch.from_numpy(board_tensor).unsqueeze(0).to(torch.device('cuda')).type(torch.float)  # Add batch dimension

    def generate_training_data(self,total_games=50,max_moves=250):
        # Simulate games by playing against a random or other AI opponent
        # Record board states, chosen moves, and outcomes for training data
        training_data = []
        start_time = time.time()
        
        game_start_time = time.time()
        for game_num in range(total_games):
            game_data = []
            self.board.reset()
            
            while not self.board.is_game_over() and self.board.ply() < max_moves*2:
                encoded_board = self.encode_board()
                predicted_move_probs, _ = self.neural_network(encoded_board)
                
                # Get legal moves for the current state
                legal_moves = list(self.board.legal_moves)
                
                # Filter predicted move probabilities for legal moves only
                legal_move_indices = [self.chess_moves.index(move.uci()) for move in legal_moves]
                legal_probs = predicted_move_probs.cpu()[0, legal_move_indices]
                
                # Normalize probabilities and choose a move
                legal_probs /= legal_probs.sum()
                chosen_move_index = np.random.choice(len(legal_moves), p=legal_probs.detach().numpy())
                chosen_move = list(legal_moves)[chosen_move_index]
                
                # Record the board state, chosen move, and outcome
                game_data.append((encoded_board, chosen_move_index, self.board.result()))
                
                # Update the board with the chosen move
                self.board.push(chosen_move)
            
            training_data.extend(game_data)
            game_end_time = time.time()
            game_duration = round(game_end_time - game_start_time, 2)
            
            if game_num % 10 == 0:
                print(f"\tGame {game_num+1}/{total_games}\tcompleted in {game_duration} seconds.") 
                game_start_time = time.time()
        
        end_time = time.time()
        total_duration = round(end_time - start_time, 2)
        print(f"\tTraining data generation completed. Total time taken: {total_duration} seconds")
        
        return training_data


    def train(self, iterations=1000):
        for iteration in range(iterations):
            training_data = self.generate_training_data()
            move_losses = []
            value_losses = []
            
            print(f"\n\nStarting training")
            dataset             = ChessDataset(training_data)
            dataloader          = DataLoader(dataset,batch_size=32,shuffle=True)
            for batch_i,data in enumerate(dataloader):
                encoded_board,chosen_move_index,outcome     = data 
                encoded_board           = encoded_board.to(torch.device('cuda')).squeeze(1)
                chosen_move_index       = chosen_move_index.to(torch.device('cuda'))
                predicted_move_probs, predicted_value = self.neural_network(encoded_board)
                
                # Calculate move probability loss
                right_moves                     = torch.zeros(1968,device=torch.device('cuda'))
                right_moves[chosen_move_index]  = 1 
                move_loss = F.cross_entropy(predicted_move_probs[0], right_moves)
                move_losses.append(move_loss)
                
                # Calculate value loss based on the outcome
                target_value = 1.0 if outcome == "1-0" else -1.0 if outcome == "0-1" else 0.0
                value_loss = self.value_criterion(predicted_value[0], torch.tensor([target_value],device=torch.device('cuda')))
                value_losses.append(value_loss)
            
            total_move_loss = sum(move_losses)
            total_value_loss = sum(value_losses)
            total_loss = total_move_loss + total_value_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            torch.save(self.neural_network.state_dict(),"nn_1_dict")
            self.trained_moves += len(training_data)
            print(f"\tIteration: {iteration+1}, Moves trained on: {self.trained_moves}, Move Loss: {total_move_loss:.4f}, Value Loss: {total_value_loss:.4f}\n\n")

        

    def play(self, human_color='white'):
        self.board.reset()
        human_turn = True if human_color == 'white' else False
        
        while not self.board.is_game_over():
            if human_turn:
                human_move = input("Your move: ")
                self.board.push_san(human_move)
            else:
                if self.board.turn == chess.WHITE:
                    print("AI's turn (White):")
                else:
                    print("AI's turn (Black):")
                
                ai_move = self.get_best_move()
                self.board.push(ai_move)
                
            human_turn = not human_turn
            
        if self.board.is_checkmate():
            print("Checkmate! Game over.")
        elif self.board.is_stalemate():
            print("Stalemate! Game over.")
        elif self.board.is_insufficient_material():
            print("Insufficient material! Game over.")
        elif self.board.is_seventyfive_moves():
            print("75-move rule! Game over.")
        elif self.board.is_fivefold_repetition():
            print("Fivefold repetition! Game over.")
    
    # ... other methods ...

if __name__ == "__main__":
    ai = SelfTeachingChessAI()
    ai.train()
    ai.play()
