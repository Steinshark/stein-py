import random 
import time 
from math import sqrt
from numpy.random import default_rng
import os 
import socket 
import numpy 
import pickle 
import torch 
import multiprocessing 
from sklearn.utils import extmath 
from rlcopy import Node,Tree
import games

DATASET_ROOT  	=	 r"//FILESERVER/S Drive/Data/chess"

def softmax(x):
		if len(x.shape) < 2:
			x = numpy.asarray([x],dtype=float)
		return extmath.softmax(x)[0]

def run_game(args):
	game,model,move_limit,search_depth,game_id,gen = args
	
	t0 						= time.time() 
	game 					= game(max_moves=move_limit,gen=gen)
	mcts_tree 				= Tree(game,model,game_id=game_id)
	move_indices            = list(range(game.move_space))
	state_repr              = [] 
	state_pi                = [] 
	state_outcome           = [] 

	while game.get_result() is None:
		try:
			#Build a local policy 
			local_policy 		= mcts_tree.update_tree(iters=search_depth)
			local_softmax 		= softmax(numpy.asarray(list(local_policy.values()),dtype=float))

			for key,prob in zip(local_policy.keys(),local_softmax):
				local_policy[key] = prob

			#construct trainable policy 
			pi              	= numpy.zeros(game.move_space)
			for move_i,prob in local_policy.items():
				pi[move_i]    = prob 

			#sample move from policy 
			next_move             = random.choices(move_indices,weights=pi,k=1)[0]

			#Add experiences to set 
			state_repr.append(game.get_repr())
			state_pi.append(pi)
			game.make_move(next_move)

			#Update MCTS tree 
			del mcts_tree
			mcts_tree					= Tree(game,model,game_id=game_id)

			game.is_game_over()
		except RecursionError:
			pass

	del mcts_tree
	send_gameover("10.0.0.60",6969)
	#Check game outcome 
	if game.is_game_over() == 1:
		state_outcome = numpy.ones(len(state_repr),dtype=torch.int8)
	elif game.is_game_over() == -1:
		state_outcome = numpy.ones(len(state_repr),dtype=torch.int8) * -1 
	else:
		state_outcome = numpy.zeros(len(state_repr),dtype=torch.int8)
	
	print(f"\tgame no. {game_id}\t== {game.get_result()}\tafter\t{game.move} moves in {(time.time()-t0):.2f}s\t {(time.time()-t0)/game.move:.2f}s/move")
	#print(game_board)
	
	state_pi		= numpy.asarray(state_pi)
	#Save tensors
	if not os.path.isdir(DATASET_ROOT+f"/experiences/gen{gen}"):
		os.mkdir(DATASET_ROOT+f"/experiences/gen{gen}")

	#Save tensors
	torch.save(torch.stack(state_repr).float(),DATASET_ROOT+f"\experiences\gen{gen}\game_{game_id}_states")
	torch.save(torch.stack(state_pi).float(),DATASET_ROOT+f"\experiences\gen{gen}\game_{game_id}_localpi")
	torch.save(state_outcome.float(),DATASET_ROOT+f"\experiences\gen{gen}\game_{game_id}_results")

	return game_id,time.time()-t0

def send_gameover(ip,port):
	try:
		sock 			= socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		sock.sendto(pickle.dumps(("gameover",None,None)),(ip,port))
		sock.close()
	except:
		print(f"err in send")
		time.sleep(.1)
		send_gameover(ip,port)


if __name__ == "__main__":
	

	n_threads 			= 8
	n_games 			= 64 
	gen 				= 0 
	offset 				= 1 

	while True:

		print(f"\n\nTraining iter {gen}")
		time.sleep(.1)

		#play out games  
		with multiprocessing.Pool(n_threads,maxtasksperchild=None) as pool:
			pool.map(run_game,[(games.Chess,"NETWORK",5,10,i,gen) for i in range(n_games)])

			
