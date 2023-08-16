from rlcopy import Node,Tree
import games 
import time 
from sklearn.utils import extmath 
import numpy 
import random
import torch 
import itertools 
DATASET_ROOT  	=	 r"//FILESERVER/S Drive/Data/chess"
import os 

import networks
from torch.utils.data import DataLoader

class Color:
	HEADER = '\033[95m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	RED = '\033[91m'
	END = '\033[0m'
	TAN = '\033[93m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

def save_model(model:networks.FullNet,gen=1,verbose=True):
	torch.save(model.state_dict(),DATASET_ROOT+f"\models\gen{gen}")
	if verbose:
		print(f"\t\tsaved to ",DATASET_ROOT+f"\models\gen{gen}")


def load_model(model:networks.FullNet,gen=1,verbose=False,tablesize=0):
	while True:
		try:
			model.load_state_dict(torch.load(DATASET_ROOT+f"\models\gen{gen}"))
			if verbose:
				print(f"\t\t{Color.BLUE}loaded model gen {gen} - lookup table size {tablesize}{Color.END}")
			return 
		
		except FileNotFoundError:
			gen -= 1 
			if gen < 0:
				print(f"\t\tloaded stock model gen {gen}")
				return
	

def softmax(x):
		if len(x.shape) < 2:
			x = numpy.asarray([x],dtype=float)
		return extmath.softmax(x)[0]


def run_game(game:games.TwoPEnv,model,search_depth,move_limit,game_id,gen=999):
	
	t0 						= time.time() 
	game 					= game(max_moves=move_limit)
	mcts_tree 				= Tree(game,model)
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
		mcts_tree				= Tree(game,model)

		#Release references to other children
		#del mcts_tree.root.parent
		game.is_game_over()
		
	#Check game outcome 
	if game.is_game_over() == 1:
		state_outcome = torch.ones(len(state_repr),dtype=torch.int8)
	elif game.is_game_over() == -1:
		state_outcome = torch.ones(len(state_repr),dtype=torch.int8) * -1 
	else:
		state_outcome = torch.zeros(len(state_repr),dtype=torch.int8)
	
	#print(f"\tgame no. {game_id}\t== {game_board.get_result()}\tafter\t{game_board.move} moves in {(time.time()-t0):.2f}s\t {(time.time()-t0)/game_board.move:.2f}s/move")
	#print(game_board)
	
	state_pi		= [torch.tensor(pi,dtype=torch.float16) for pi in state_pi]
	#Save tensors
	if not os.path.isdir(DATASET_ROOT+f"/experiences/gen{gen}"):
		os.mkdir(DATASET_ROOT+f"/experiences/gen{gen}")
	torch.save(torch.stack(state_repr).float(),DATASET_ROOT+f"/experiences/gen{gen}/game_{game_id}_states")
	torch.save(torch.stack(state_pi).float(),DATASET_ROOT+f"/experiences/gen{gen}/game_{game_id}_localpi")
	torch.save(state_outcome.float(),DATASET_ROOT+f"/experiences/gen{gen}/game_{game_id}_results")

	return game_id,time.time()-t0


def train(model:networks.FullNet,n_samples,gen,bs=8,epochs=5,DEV=torch.device('cuda' if torch.cuda.is_available else 'cpu')):
	model 						= model.float()
	root                        = DATASET_ROOT+f"\experiences\gen{gen}"
	experiences                 = []
	model.train()
	if not os.listdir(root):
		print(f"No data to train on")
		return


	print(f"{Color.TAN}\n\t\tbegin Training:{Color.END}",end='')
	for game_i in range(500):
		try:
			states                      = torch.load(f"{root}/game_{game_i}_states").float().to(DEV)
			pi                          = torch.load(f"{root}/game_{game_i}_localpi").float().to(DEV)
			results                     = torch.load(f"{root}/game_{game_i}_results").float().to(DEV)
			for i in range(len(states)):
				experiences.append((states[i],pi[i],results[i]))
		except FileNotFoundError:
			pass 
	print(f"{Color.TAN}\tloaded {len(experiences)} datapoints{Color.END}")
	

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


def run_train_iteration(game:games.TwoPEnv,model:torch.nn.Module,search_depth,move_limit,train_round,n_iters,gen):
	print(f"\tSTART TRAINING ITER {train_round}")
	t0 				= time.time()
	model.eval()
	model 			= torch.jit.trace(model,[torch.randn((1,6,8,8)).to("cuda")])
	model 			= torch.jit.freeze(model)

	for id in range(n_iters):
		run_game(game,model,search_depth=search_depth,move_limit=move_limit,game_id=id,gen=gen)
	model.train()
	print(f"\t\tran {n_iters} games in {(time.time()-t0):.2f}s\t: {((time.time()-t0)/n_iters):.2f}s/game")


def play_models(cur_model_net,challenger_model_net,search_depth,max_moves):

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
		

#Duel a random 
def duel(available_models,n_games,search_depth,cur_model=0,max_moves=120):
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
	
	challenger_ratio 	= ((challenger_games) / (current_best_games+challenger_games))

	if challenger_ratio >= .55:
		best_model 			= challenger_model
		worst_model			= cur_model

	


	print(f"\t{Color.GREEN}Cur model{cur_model}: {current_best_games}\tChallenger model{challenger_model}: {challenger_games}\ttie: {tiegames}\n")

	#Delete worst model 
	print(f"\t{Color.GREEN}best model is {best_model}{Color.END}")
	print(f"\t{Color.RED}removing {worst_model}{Color.END}")
	os.remove(DATASET_ROOT+f"\\models\\gen{worst_model}")
	return best_model


def get_generations():
	
	gens 	= [] 
	for model in os.listdir(DATASET_ROOT+f"\models\\"):
		gens.append(int(model.replace("gen","")))

	return gens 





if __name__ == "__main__" and False:

	train_rotations 		 = 5
	search_depth 			 = 1000
	best_gen 				 = 0
	
	train_model 			= networks.ChessSmall(optimizer_kwargs={'lr':.0005,"weight_decay":.00005},n_ch=6)
	load_model(train_model,0)


	for generation in range(200):
		best_gen			= generation
		print(f"{Color.TAN}GENERATION {generation}{Color.END}")

		for round in range(train_rotations): 
			n_games 		= random.randint(4,16)
			run_train_iteration(games.Chess,train_model,300,100,round,n_games,generation)
			#load_model(base_model,gen=generation,verbose=False)
			train(train_model,n_samples=2048,gen=generation,bs=64,epochs=2)
			save_model(train_model,gen=generation,verbose=False)
		
		if generation >=	 3:
			best_gen 			= duel(get_generations(),50,search_depth=search_depth,cur_model=best_gen,max_moves=120)
			load_model(train_model,gen=best_gen)

if __name__ == "__main__" and False:
	model 	= networks.ChessSmall(device=torch.device('cuda'))
	model.eval()
	model 			= torch.jit.trace(model,[torch.randn((1,6,8,8)).to("cuda")])
	model 			= torch.jit.freeze(model)
	run_game(games.Chess,model,1000,5,-1,0)

if __name__ == "__main__" and True:

	from server_logistics import Server
	s 	= Server(10,300,300,start_gen=2)
	s.run_server(60)
