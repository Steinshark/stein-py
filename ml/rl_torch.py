import torch 
import time 
import tkinter as tk
from math import sqrt 
import numpy 
import sys 
import numpy 
from sklearn.utils import extmath 
import games 
import pickle

NETWORK_BUFFER_SIZE 			= 1024*16
def softmax(x):
	if len(x.shape) < 2:
		x = numpy.asarray([x],dtype=float)
	return extmath.softmax(x)[0]

sys.path.append("C:/gitrepos/steinpy/ml")

class Node:


	def __init__(self,game_obj:games.TwoPEnv,p=.5,parent=None,c=20,uuid=""):

		self.game_obj 		= game_obj 
		self.parent 		= parent 
		self.parents 		= [parent]
		self.children		= {}
		self.num_visited	= 0

		self.Q_val 			= 0 
		self.p				= p 
		self.c 				= c 

		self.score 			= 0

		self.uuid			= uuid

		self.fen 			= game_obj.board.fen().split("-")[0]
	
	def get_score(self):
		return self.Q_val + ((self.c * self.p) * (sqrt(sum([m.num_visited for m in self.parent.children.values()])) / (1 + self.num_visited)))
	
	def bubble_up(self,v):

		#Update this node
		self.Q_val 			= (self.num_visited * self.Q_val + v) / (self.num_visited + 1) 
		self.num_visited 	+= 1

		# for parent in self.parents:	
		# 	#Recursively update all parents 
		if not self.parent is None:
			self.parent.bubble_up(-1*v)
		
	def cleanup(self):
		del self.game_obj
		del self.parent 
		del self.parents 
		del self.children 
		del self.num_visited
		del self.move 
		del self.Q_val
		del self.p 
		del self.c 
		del self.score 
		del self.uuid 
		del self.fen 

def get_best_node_max(node:Node):
	while node.children:
			
			best_node 			= max(list(node.children.values()),key = lambda x: x.get_score())
			node 				= best_node

	return node

class Tree:

	
	def __init__(self,game_obj:games.TwoPEnv,model:torch.nn.Module,base_node=None,local_cache={}):
		
		self.game_obj 		= game_obj 
		self.model 			= model 
		self.model.eval()

		self.root 			= base_node if not base_node is None else Node(game_obj,0,None)
		self.root.parent 	= None
		self.local_cache 	= {} 


	def update_tree(self,x=.95,dirichlet_a=.3,iters=200): 
		
		#DEFINE FUNCTIONS IN LOCAL SCOPE 
		noise_gen				= numpy.random.default_rng().dirichlet
		softmax_fn				= softmax
		get_best_fn				= get_best_node_max



		self.root.parent		= None 
		self.nodes 				= {}

		for _ in range(iters):

			#Define starting point
			node 					= self.root
			starting_move 			= 1 if self.root.game_obj.board.turn else -1

			#Find best leaf node 
			node 					= get_best_fn(node)

			#Add node to global list
			if not node.fen in self.nodes:
				self.nodes[node.fen]= [node]
			else:
				self.nodes[node.fen].append(node)

			#Check if game over
			result:float or bool 	= node.game_obj.is_game_over()
			
			if not result is None:
				if result == 0:
					v = 0 
				elif result == starting_move:
					v = 1 
				else:
					v = -1 
					
			
			#expand 
			else:
				if node.fen in self.local_cache:
					legal_probs,legal_moves,v 					= self.local_cache[node.fen]
				else:
					with torch.no_grad():
						#receive local evaluation
						prob,v 						= self.model.forward(node.game_obj.get_repr().unsqueeze_(0))
						prob_cpu					= prob[0].to(torch.device('cpu'),non_blocking=True).numpy()
						legal_moves 				= node.game_obj.get_legal_moves()
						legal_probs 				= numpy.array([prob_cpu[i] for i in legal_moves])
						self.local_cache[node.fen]	= (legal_probs,legal_moves,v)


				
				noise 						= noise_gen([dirichlet_a for _ in range(len(legal_probs))],1)*(1-x)
				legal_probs					= softmax_fn(legal_probs*x + noise)

				node.children 		= {move_i : Node(node.game_obj.copy() ,p=p,parent=node) for p,move_i in zip(legal_probs,legal_moves)} 
				[node.children[move].game_obj.make_move(move) for move in node.children]

			#Ensure v is in a builtin type 
			v = float(v)

			for identical_node in self.nodes[node.fen]:
				identical_node.bubble_up(v)				

		#Cleanup and return policy
		del self.nodes
		return {move:self.root.children[move].num_visited for move in self.root.children}, self.local_cache


	#217 is downstairs
	#60  is room 
	def SEND_EVAL_REQUEST(self,port=6969,hostname="10.0.0.217",sleep_time=.2):
		
		try:
			self.sock.sendto(pickle.dumps(self.game_obj.build_as_network()),(hostname,port))
			server_response,addr 			= self.sock.recvfrom(NETWORK_BUFFER_SIZE)
			prob,v 							= pickle.loads(server_response)
			return prob,v
		except TimeoutError:
			time.sleep(sleep_time)
			return self.SEND_EVAL_REQUEST(port=port,hostname=hostname,sleep_time=sleep_time*2)
		except OSError as ose:
			print(f"\tos err\n\t{ose}")
			time.sleep(2)
			return self.SEND_EVAL_REQUEST(port=port,hostname=hostname)
		

	def get_policy(self,search_iters,abbrev=True):
		return self.update_tree_nonrecursive_exp(iters=search_iters,abbrev=abbrev)


if __name__ == "__main__":
	pass
