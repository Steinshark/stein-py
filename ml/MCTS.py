import random 
import time 
import chess 
from math import sqrt
from numpy.random import default_rng
from scipy.special import softmax
import json 
import os 

try:
	chess_moves 		= json.loads(open(os.path.join("steinpy","ml","res","chessmoves.txt"),"r").read())
except FileNotFoundError:
	chess_moves 		= json.loads(open(os.path.join("/home/steinshark/code","steinpy","ml","res","chessmoves.txt"),"r").read())


class Node:


	def __init__(self,board,p=.5,parent=None,c=10):

		self.board 			= board 
		self.parent 		= parent 
		self.parents 		= [parent]
		self.children		= {}
		self.num_visited	= 0

		self.Q_val 			= 0 
		self.p				= p 
		self.c 				= c 

		self.score 			= 0

	
	def get_score(self):
		return self.Q_val + ((self.c * self.p) * (sqrt(sum([m.num_visited for m in self.parent.children.values()])) / (1 + self.num_visited)))

class Tree:
		
	def __init__(self,board,model,base_node=None,draw_thresh=250,chess_moves={},index_to_move={},move_to_index={}):
		self.board 			= board 
		self.model 			= model 
		self.draw_thresh	= draw_thresh

		self.chess_moves    = chess_moves 
		self.move_to_index  = move_to_index 
		self.index_to_move  = index_to_move 
		
		if base_node: 
			self.root 			= base_node
			self.root.parent 	= None
		else:
			self.root 			= Node(board,0,None)
			self.root.parent	= None 
		
		
	def update_tree_nonrecursive_exp(self,x=.8,dirichlet_a=.3,rollout_p=.25,iters=300,abbrev=True): 
		
		#DEFINE FUNCTIONS IN LOCAL SCOPE 
		infer 					= self.model.forward
		noise_gen				= default_rng().dirichlet
		softmax_fn				= softmax


		t_test 					=  0

		self.root.parent		= None 
		flag					= True
		debugging 				= False 

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
					continuing = input(f"policy is now/n{[ (Chess.chess_moves[k],v.get_score()) for k,v in self.root.children.items()]}")
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
				node.repr 				= self.SEND_EVAL_REQUEST()
				
				#	
				with torch.no_grad():
					model_in 					= node.repr.unsqueeze(0)
					prob,v 						= infer(model_in)
				prob_cpu					= prob[0].to(torch.device('cpu'),non_blocking=True).numpy()
				legal_moves 				= [move_to_index[m] for m in node.board.legal_moves]	#Get move numbers
				legal_probs 				= [prob_cpu[i] for i in legal_moves]

				noise 						= noise_gen([dirichlet_a for _ in range(len(legal_probs))],1)
				#input(f"noise yields: {noise}")
				legal_probs					= softmax_fn([x*p for p in legal_probs] + (1-x)*noise)[0]
				#input(f"model yields probs: {legal_probs}")
				#self.lookup_table[position_fen]	=	 (v,legal_moves,legal_probs)

				if debugging:
					input(f"\n{node.board}\nposition evaluates to {v}")
				node.children 		= {move_i : Node(node.board.copy(stack=False) ,p=p,parent=node) for p,move_i in zip(legal_probs,legal_moves)} 


				for move in node.children:
					node.children[move].board.push(index_to_move[move])
				
				#Remove rollout for now
				if False and random.random() < rollout_p:
					v = self.rollout_exp(random.choice(list(node.children.values())).board.copy()) * score_mult
							
			while node.parent:
				if isinstance(v,torch.Tensor):
					v = v.item()
				node.Q_val 			= (node.num_visited * node.Q_val + v) / (node.num_visited + 1) 
				node.num_visited 	+= 1 

				#Try updating score only on leaf nodes
				# node.score 			= node.get_score()
				# total_score_calls 	+= 1 

				v *= -1 
				node = node.parent
		
		return {move:self.root.children[move].num_visited for move in self.root.children}

	def SEND_EVAL_REQUEST(self):
		return None,None 