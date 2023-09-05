import chess 
import chess.engine 
import chess
import games 
import numpy 
from rl_notorch import softmax 
import random 
import sys 
import pprint 
import json 

game_dummy_obj      = games.Chess(0) 
move_to_index       = game_dummy_obj.move_to_index
chess_moves         = game_dummy_obj.chess_moves
DATASET_ROOT  	    = r"//FILESERVER/S Drive/Data/chessSL"
engine              = chess.engine.SimpleEngine.popen_uci("C:/gitrepos/stockfish/sf16.exe")
data_list           = [] 
n_saves             = 0
import time 
t0                  = time.time()
offset              = int(sys.argv[1])
# game        = chess.Board(fen="1n1rkbnr/pb1ppp1p/1pp3p1/6Nq/2Q2PP1/PP6/1BPPP2P/3RKBNR b Kk - 7 19")
# engine_res          = engine.analyse(game,limit=chess.engine.Limit(depth=6),info=chess.engine.Info.ALL,multipv=100) 


# print([e['score'].white() for e in engine_res])

def process_score(move_evals:list,perspective:chess.WHITE,temp=5):
    #print(f"game turn is {perspective}")
    move_repr       = numpy.zeros(len(move_evals))
    #print(f"is_mate:{move_evals[0].white().is_mate()} perspectives {bool(move_evals[0].white().mate()+1)} == {perspective}--> {move_evals[0].white().is_mate() and (bool(move_evals[0].white().mate()) == perspective)}")
    mate_adjusted   = [2_000_000 if (e.white().is_mate() and (bool(e.white().mate()+1) == perspective)) else 0 if (e.white().is_mate()) else 1_000_000 for e in move_evals]

    
    if 2_000_000 in mate_adjusted or 0 in mate_adjusted:
        return softmax(numpy.array(mate_adjusted))
    else:
        maximum     = max([abs(e.white().score()) for e in move_evals])/temp
        try:
            return softmax(numpy.array([e.white().score()/maximum for e in move_evals]))
        except ZeroDivisionError:
            #print(f"Zero div: {[1/len(move_evals) for e in move_evals]}")
            return numpy.array([1/len(move_evals) for e in move_evals])






while True:

    game    =    chess.Board()

    while not game.is_game_over():

        #get engine move 
        engine_res          = engine.analyse(game,limit=chess.engine.Limit(depth=16),info=chess.engine.Info.ALL,multipv=5) 
        ids                 = [move_to_index[d['pv'][0]] for d in engine_res]
        #print(f"before_soft: {[numpy.array([e['score'].white() for e in engine_res])]}")
        vals                = process_score([d['score'] for d in engine_res],game.turn)

        #pprint.pp("outs:"+f"\n{vals}")
        #input()
        #continue 
        datapoint           = {"ids":[],"vals":[],"fen":None}

        datapoint["fen"]    = game.fen()
        for id,val in zip(ids,vals):
            datapoint["ids"].append(id)
            datapoint["vals"].append(val)

        data_list.append(datapoint)    

        #Make random move 
        game.push(random.choice(list(game.generate_legal_moves())))
    
        if len(data_list) > 20000:
            with open(f"{DATASET_ROOT}/exps{(offset*10000+n_saves)}","w") as file:
                file.write(json.dumps(data_list))
                data_list  = [] 
                n_saves     += 1       




