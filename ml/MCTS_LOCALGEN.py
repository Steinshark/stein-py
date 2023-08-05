import random 
import time 
import chess 
from math import sqrt
from numpy.random import default_rng
import json 
import os 
import socket 
import networks
from networks import ChessNet, ChessDataset
import numpy 
import pickle 
import torch 
import sys 
import multiprocessing 
from torch.utils.data import DataLoader
from sklearn.utils import extmath 
from rlcopy import Node,Tree
import games
socket.setdefaulttimeout(2)
DATASET_ROOT  	=	 r"//FILESERVER/S Drive/Data/chess3"

def softmax(x):
		if len(x.shape) < 2:
			x = numpy.asarray([x],dtype=float)
		return extmath.softmax(x)[0]


def run_game(game:games.TwoPEnv,model:networks.FullNet or str,search_depth,move_limit,game_id,gen=999):
	
	t0 						= time.time() 
	game 					= game(max_moves=move_limit)
	mcts_tree 				= Tree(game,model,game_id=game_id)
	move_indices            = list(range(game.move_space))
	state_repr              = [] 
	state_pi                = [] 
	state_outcome           = [] 

	while game.get_result() is None:

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
		# child_node 				= mcts_tree.root.children[next_move_i]
		del mcts_tree
		mcts_tree					= Tree(game,model,game_id=game_id)

		#Release references to other children
		#del mcts_tree.root.parent
		game.is_game_over()
	
	send_gameover("10.0.0.60",6969)
	#Check game outcome 
	if game.is_game_over() == 1:
		state_outcome = torch.ones(len(state_repr),dtype=torch.int8)
	elif game.is_game_over() == -1:
		state_outcome = torch.ones(len(state_repr),dtype=torch.int8) * -1 
	else:
		state_outcome = torch.zeros(len(state_repr),dtype=torch.int8)
	
	print(f"\tgame no. {game_id}\t== {game.get_result()}\tafter\t{game.move} moves in {(time.time()-t0):.2f}s\t {(time.time()-t0)/game.move:.2f}s/move")
	#print(game_board)
	
	state_pi		= [torch.tensor(pi,dtype=torch.float16) for pi in state_pi]
	#Save tensors
	if not os.path.isdir(DATASET_ROOT+f"/experiences/gen{gen}"):
		os.mkdir(DATASET_ROOT+f"/experiences/gen{gen}")
	torch.save(torch.stack(state_repr).float(),DATASET_ROOT+f"/experiences/gen{gen}/game_{game_id}_states")
	torch.save(torch.stack(state_pi).float(),DATASET_ROOT+f"/experiences/gen{gen}/game_{game_id}_localpi")
	torch.save(state_outcome.float(),DATASET_ROOT+f"/experiences/gen{gen}/game_{game_id}_results")

	return game_id,time.time()-t0

def send_gameover(ip,port):
	sock 			= socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
	sock.sendto(pickle.dumps(("gameover",None,None)),(ip,port))

def train(model:networks.FullNet,n_samples,gen,bs=8,epochs=5,DEV=torch.device('cuda' if torch.cuda.is_available else 'cpu')):
	model 						= model.float()
	root                        = DATASET_ROOT+f"\experiences\gen{gen}"
	experiences                 = []

	if not os.listdir(root):
		print(f"No data to train on")
		return


	for game_i in range(500):
		for local_iter in range(200):
			try:
				states                      = torch.load(f"{root}/game_{local_iter}-{game_i}_states").float().to(DEV)
				pi                          = torch.load(f"{root}/game_{local_iter}-{game_i}_localpi").float().to(DEV)
				results                     = torch.load(f"{root}/game_{local_iter}-{game_i}_results").float().to(DEV)
				for i in range(len(states)):
					experiences.append((states[i],pi[i],results[i]))
			except FileNotFoundError:
				pass 
			


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
			loss                        = torch.nn.functional.mse_loss(v_pred.view(-1),outcome,reduction='mean') + torch.nn.functional.cross_entropy(pi_pred,pi,)
			total_loss                  += loss.mean().item()
			loss.backward() 

			#Backpropogate
			model.optimizer.step()
		
		print(f"\t\tEpoch {epoch_i} loss: {total_loss/batch_i:.3f} with {len(train_set)}/{len(experiences)}")


if __name__ == "__main__":


	n_threads 			= 8
	n_games 			= 64 
	gen 				= 999 
	offset 				= 1 

	while True:
		train_ids 	= [0+offset,2+offset,4+offset]
		for train_id in train_ids:
			print(f"\n\nTraining iter {gen}")
			time.sleep(.1)

			#play out games  
			with multiprocessing.Pool(n_threads) as pool:
				results 	= pool.starmap(run_game,[(games.Chess,"NETWORK",100,10,i,gen) for i in range(n_games)])
			print(f"finished ")
			
