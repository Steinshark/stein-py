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
try:
	chess_moves 		= json.loads(open(os.path.join("C:/gitrepos/steinpy/ml/res/chessmoves.txt"),"r").read())
except FileNotFoundError:
	chess_moves 		= json.loads(open(os.path.join("/home/steinshark/code","steinpy","ml","res","chessmoves.txt"),"r").read())

move_to_index		= {chess.Move.from_uci(uci):i for i,uci in enumerate(chess_moves)}
index_to_move	    = {v : k  for k,v in move_to_index.items()}

def softmax(x):
    return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=0)

class Node:


	def __init__(self,board,p=.5,parent=None,c=.5):

		self.board 			= board 
		self.parent 		= parent 
		self.parents 		= [parent]
		self.children		= {}
		self.num_visited	= 0

		self.Q_val 			= 0 
		self.p				= p 
		self.c 				= c 

		self.score 			= 0

		self.pos_fen 		= self.board.fen().split(" ")[1]
	
	def get_score(self):
		return self.Q_val + ((self.c * self.p) * (sqrt(sum([m.num_visited for m in self.parent.children.values()])) / (1 + self.num_visited)))

class Tree:
	def __init__(self,board,base_node=None,draw_thresh=250,port=6969,hostname=socket.gethostname(),uid=0):
		self.board 			= board 
		self.draw_thresh	= draw_thresh

		self.chess_moves    = chess_moves 
		self.uid 			= uid
		self.sock 			= socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

		
		#self.sock.connect((hostname,port))
		if base_node: 
			self.root 			= base_node
			self.root.parent 	= None
		else:
			self.root 			= Node(board,0,None)
			self.root.parent	= None 
		
		self.nodes 				= {}
		
	#@memory_profiler.profile	
	def update_tree_nonrecursive_exp(self,x=.9,dirichlet_a=.3,rollout_p=.5,iters=300,abbrev=True,lookup_table=None,lock=None,game_id=None): 
		
		#DEFINE FUNCTIONS IN LOCAL SCOPE 
		noise_gen				= default_rng().dirichlet
		softmax_fn				= softmax


		t_test 					=  0

		self.root.parent		= None 
		flag					= False
		debugging 				= False 
		t_beg 					= 0 
		t_0						= time.time()
		for iter_i in range(iters):

			node = self.root
			score_mult = 1 if node.board.turn == chess.WHITE else -1

			#Find best leaf node 
			node, score_mult = self.get_best_node_max(node,score_mult)

			#Check if game over
			game_over 	= node.board.is_checkmate() or node.board.is_stalemate() or node.board.is_seventyfive_moves()
			if game_over:
				if "1" in node.board.result():
					if node.board.result()[0] == "1":
						v 	=   1 * score_mult
					else:
						v 	=  -1 * score_mult
				else:
					v 	=  0 
				if flag and debugging:
					print(f"result was {node.board.result()}")
					print(f"found result of {v} in position\n{node.board}\nafter {'white' if node.board.turn else 'black'} moved")
					continuing = input(f"policy is now/n{[ (chess_moves[k],v.get_score()) for k,v in self.root.children.items()]}")
					if continuing == "stop":
						flag = False
			
			#expand 
			else:

				#Do table lookup
				if abbrev:
					position_fen 			= node.board.fen().split(" ")[0]
				else:
					position_fen 			= node.board.fen()

				# if position_fen in self.lookup_table:
				# 	v, legal_moves,legal_probs = self.lookup_table[position_fen]
				# 	node.repr 				= None 

				# else:
				fen	= node.board.fen().split(" ")[0]
				t_1 = time.time()
				if not lookup_table is None and not lock is None:
					not_found_flag 	= True 
					with lock:
						if fen in lookup_table:
							prob,v = lookup_table[fen]
						else:
							not_found_flag = True 
					if not_found_flag:
						prob,v 				= self.SEND_EVAL_REQUEST(node.board.fen())
						with lock:
							lookup_table[fen]	= (prob,v)

				else:
					prob,v 				= self.SEND_EVAL_REQUEST(node.board.fen())
				t_beg += time.time()-t_1

				#PROB WILL BE A NUMPY ARRAY 

				legal_moves 				= [move_to_index[m] for m in node.board.legal_moves]	#Get move numbers
				legal_probs 				= [prob[i] for i in legal_moves]

				noise 						= noise_gen([dirichlet_a for _ in range(len(legal_probs))],1)
				#input(f"noise yields: {noise}")
				legal_probs					= softmax_fn([x*p for p in legal_probs] + (1-x)*noise)[0]
				#input(f"model yields probs: {legal_probs}")
				#self.lookup_table[position_fen]	=	 (v,legal_moves,legal_probs)
				print(f"position {node.board}\nwas eval as {v}")
				

				#Add all new nodes to children 
				node.children 		= {move_i : Node(node.board.copy(stack=False) ,p=p,parent=node) for p,move_i in zip(legal_probs,legal_moves)} 

				#update all new nodes 
				for move in node.children:
					node.children[move].board.push(index_to_move[move])
					

					if not node.pos_fen in self.nodes:
						self.nodes[node.pos_fen] = [node.children]
					else:
						self.nodes[node.pos_fen] += node.children
				
				#Remove rollout for now
				if False and random.random() < rollout_p:
					v = self.rollout_exp(random.choice(list(node.children.values())).board.copy()) * score_mult
							
			parents 	= [parent for parent in parents in self.nodes[node.pos_fen]]

			while parents:
				node 				= parents.pop()
				node.Q_val 			= (node.num_visited * node.Q_val + v) / (node.num_visited + 1) 
				node.num_visited 	+= 1 

				#Try updating score only on leaf nodes
				# node.score 			= node.get_score()
				# total_score_calls 	+= 1 

				v *= -1 
				parents 			+= [parent for parent in parents in self.nodes[node.pos_fen]]
		
		return {move:self.root.children[move].num_visited for move in self.root.children}

	def SEND_EVAL_REQUEST(self,fen='',port=6969,hostname=socket.gethostname()):
		
		self.sock.sendto(fen.encode(),(hostname,port))
		#print(f"\tuid {self.uid} sent req")

		prob,addr 			= self.sock.recvfrom(4096)
		#print(f"\tuid {self.uid} recieved back prob")
		v,addr 				= self.sock.recvfrom(512)
		#print(f"\tuid {self.uid} recieved back v")
		prob 				= pickle.loads(prob)
		v 					= pickle.loads(v)	
		return prob,v

	def rollout_exp(self,board:chess.Board):
		started 	= board.turn
		
		#If gameover return game result 
		while not (board.is_checkmate() or board.is_stalemate() or board.is_seventyfive_moves()):
			
			#If over draw thresh, return 0
			if board.ply() > self.draw_thresh:
				return 0
			
			board.push(random.choice(list(board.generate_legal_moves())))
		
		res = board.result()
		if res[0] == "1":
			v = 1 
		elif res[-1] == "1":
			v =  -1 
		else:
			v = 0

		return -v if started == board.turn else v	  

	def get_best_node_max(self,node:Node,score_mult):
		score_mult *= -1
		while node.children:
				#drive down to leaf
				best_node 			= max(list(node.children.values()),key = lambda x: x.get_score())
				node 				= best_node
				score_mult			*= -1

		return node, score_mult


def fen_to_tensor(fen):

	#Encoding will be an 8x8 x n tensor 
	#	7 for whilte, 7 for black 
	#	4 for castling 7+7+4 
	# 	1 for move 
	#t0 = time.time()
	#board_tensor 	= torch.zeros(size=(1,19,8,8),device=device,dtype=torch.float,requires_grad=False)
	board_tensor 	= numpy.zeros(shape=(19,8,8))
	piece_indx 	= {"R":0,"N":1,"B":2,"Q":3,"K":4,"P":5,"r":6,"n":7,"b":8,"q":9,"k":10,"p":11}
	
	#Go through FEN and fill pieces
	for i in range(1,9):
		fen 	= fen.replace(str(i),"e"*i)

	position	= fen.split(" ")[0].split("/")
	turn 		= fen.split(" ")[1]
	castling 	= fen.split(" ")[2]
	
	#Place pieces
	for rank_i,rank in enumerate(reversed(position)):
		for file_i,piece in enumerate(rank): 
			if not piece == "e":
				board_tensor[piece_indx[piece],rank_i,file_i]	= 1.  
	
	#print(f"init took {(time.time()-t0)}")
	#Place turn 
	slice 	= 12 
	#board_tensor[0,slice,:,:]	= torch.ones(size=(8,8)) * 1 if turn == "w" else -1
	board_tensor[slice,:,:]   = numpy.ones(shape=(8,8)) * 1 if turn == "w" else -1

	#Place all castling allows 
	for castle in ["K","Q","k","q"]:
		slice += 1
		#board_tensor[0,slice,:,:]	= torch.ones(size=(8,8)) * 1 if castle in castling else 0
		board_tensor[slice,:,:]	= numpy.ones(shape=(8,8)) * 1 if castle in castling else 0

	return torch.tensor(board_tensor,dtype=torch.int8,requires_grad=False)


def run_training(search_depth,move_limit,game_id,lookup_table=None,lock=None,gen=999):
	
	t0 						= time.time() 
	
	game_board 				= chess.Board() 
	mcts_tree 				= Tree(game_board,None,move_limit)
	move_indices            = list(range(1968))
	state_repr              = [] 
	state_pi                = [] 
	state_outcome           = [] 

	while not (game_board.is_checkmate() or game_board.is_fifty_moves() or game_board.is_stalemate() or game_board.ply() > move_limit):

		#Build a local policy 
		if not lookup_table is None and not lock is None:
			local_policy 		= mcts_tree.update_tree_nonrecursive_exp(iters=search_depth,lookup_table=lookup_table,lock=lock,game_id=game_id)
		else:
			local_policy 		= mcts_tree.update_tree_nonrecursive_exp(iters=search_depth,game_id=game_id)

		#construct trainable policy 
		pi                                      = numpy.ones(1968) * float("-inf")
		for move_i,prob in local_policy.items():
			pi[move_i]    = prob 
		pi                      = softmax(pi)

		#sample move from policy 
		next_move_i             = random.choices(move_indices,weights=pi,k=1)[0]
		next_move               = index_to_move[next_move_i]

		#Add experiences to set 
		state_repr.append(fen_to_tensor(game_board.fen()))
		state_pi.append(pi)
		game_board.push(next_move)

		#Update MCTS tree 
		# child_node 				= mcts_tree.root.children[next_move_i]
		del mcts_tree
		mcts_tree				= Tree(game_board,None,move_limit)

		#Release references to other children
		#del mcts_tree.root.parent
		
	#Check game outcome 
	if "1" in game_board.result() and not "1/2" in game_board.result():
		if game_board.result()[0] == "1":
			state_outcome = torch.ones(len(state_repr),dtype=torch.int8)
		else:
			state_outcome = torch.ones(len(state_repr),dtype=torch.int8) * -1 
	else:
		state_outcome = torch.zeros(len(state_repr),dtype=torch.int8)
	
	print(f"\tgame {game_id}\t== in {game_board.result() if not '1/2' in game_board.result() else '---'} after {game_board.ply()} moves in {(time.time()-t0):.2f}s\t {(time.time()-t0)/game_board.ply():.2f}s/move")
	state_pi		= [torch.tensor(pi,dtype=torch.float16) for pi in state_pi]
	#Save tensors
	torch.save(torch.stack(state_repr).float(),f"C:/data/chess/experiences/gen{gen}/game_{game_id}_states")
	torch.save(torch.stack(state_pi).float(),f"C:/data/chess/experiences/gen{gen}/game_{game_id}_localpi")
	torch.save(state_outcome.float(),f"C:/data/chess/experiences/gen{gen}/game_{game_id}_results")

	return game_id,time.time()-t0


def save_model(model:networks.FullNet,gen=1):
    torch.save(model.state_dict(),f"C:/data/chess/models/gen{gen}")


def load_model(model:networks.FullNet,gen=1,verbose=False):
	while True:
		try:
			model.load_state_dict(torch.load(f"C:/data/chess/models/gen{gen}"))
			if verbose:
				print(f"\tloaded model gen {gen}")
			return 
		
		except FileNotFoundError:
			gen -= 1 
			if gen <= 0:
				print(f"\tloaded stock model gen {gen}")
				return


def train(model:networks.FullNet,n_samples=8192,gen=999,bs=8,epochs=5,DEV=torch.device('cuda' if torch.cuda.is_available else 'cpu')):
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
            loss                        = torch.nn.functional.mse_loss(v_pred.view(-1),outcome,reduction='mean') + torch.nn.functional.cross_entropy(pi_pred,pi,)
            total_loss                  += loss.mean().item()
            loss.backward() 

            #Backpropogate
            model.optimizer.step()
        
        print(f"\t\tEpoch {epoch_i} loss: {total_loss:.3f} with {len(train_set)}/{len(experiences)}")


if __name__ == "__main__":
	#game_id		= int(sys.argv[1])
	#game_id,play_time 	= run_training(500,250,game_id=game_id)
	#print(f"\t{game_id} finished in {play_time:.2f}")

	model 			= ChessNet(optimizer=torch.optim.SGD,optimizer_kwargs={"lr":.0001,"weight_decay":.00001,'momentum':.9})
	manager 		= multiprocessing.Manager()
	lookup_table 	= manager.dict()
	lock 			= manager.Lock() 

	lookup_table 	= None
	lock 			= None

	if sys.argv[1] == "train":
		load_model(model,gen=999,verbose=True)
		train(model,n_samples=8096*8,gen=999,bs=32,epochs=5)
		save_model(model,999)
	
	elif sys.argv[1] == "sim":
		if len(sys.argv) >= 3:
				n_threads 	= int(sys.argv[2])

		if len(sys.argv) >= 4:
			n_games 	= int(sys.argv[3])
		
		if len(sys.argv) >= 5:
			model_n 	= int(sys.argv[4])
		else:
			model_n 	= 0
		train_iters 	= 20 
		load_model(model,gen=model_n,verbose=True)
		for gen in range(train_iters):
			print(f"\n\nTraining iter {gen}")
			time.sleep(3)

			#play out games  
			with multiprocessing.Pool(n_threads) as pool:
				results 	= pool.starmap(run_training,[(600,250,i) for i in range(n_games)])
			
			print(f"\n\tTraining:")
			train(model,n_samples=8096*2,gen=model_n,bs=32,epochs=5)
			save_model(model,model_n)
	elif sys.argv[1] == "test":
		load_model(model,1)
		run_training(500,250,999)
