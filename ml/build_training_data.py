import chess 
import chess.engine 
import chess
import games 
import numpy 
from rl_notorch import softmax 
import random 

game_dummy_obj      = games.Chess(0) 
move_to_index       = game_dummy_obj.move_to_index
chess_moves         = game_dummy_obj.chess_moves
DATASET_ROOT  	    = r"//FILESERVER/S Drive/Data/chess"
engine              = chess.engine.SimpleEngine.popen_uci("C:/gitrepos/stockfish/sf16.exe")
moves_list          = [] 
n_saves             = 0

while True:

    game    =    chess.Board() 

    while not game.is_game_over():

        #get engine move 
        engine_res          = engine.analyse(game,limit=chess.engine.Limit(time=4),info=chess.engine.Info.ALL,multipv=10) 
        ids                 = [move_to_index[d['pv'][0]] for d in engine_res]
        vals                = softmax(numpy.asarray([d['score'].white().cp for d in engine_res]))
        move_repr           = numpy.zeros(1968)

        for id,score in zip(ids,vals):
            move_repr[id]       = score 
        
        moves_list.append(move_repr)

        #Make random move 
        game.push(random.choice(list(game.generate_legal_moves())))
    
    if len(moves_list) > 10000:
        numpy.save(f"{DATASET_ROOT}/exps{n_saves}",numpy.array(moves_list))
        moves_list  = [] 
        n_saves     += 1       




