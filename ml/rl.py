import torch 
import os 
import time 
import random 
import tkinter as tk
import chess
import chess.svg
import json 
from math import sqrt 
import numpy 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys 
import numpy 
from sklearn.utils import extmath 
import games 
import pickle
import copy
import socket
import numba 

def softmax(x):
	if len(x.shape) < 2:
		x = numpy.asarray([x],dtype=float)
	return extmath.softmax(x)[0]

sys.path.append("C:/gitrepos/steinpy/ml")

class Node:


	def __init__(self,game_obj:games.TwoPEnv,p=.5,parent=None,c=20,move=None,uuid=""):

		self.game_obj 		= game_obj 
		self.parent 		= parent 
		self.parents 		= [parent]
		self.children		= {}
		self.num_visited	= 0
		self.move 			= None 

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


class Tree:

	
	def __init__(self,game_obj:games.TwoPEnv,base_node=None,game_id=0,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),server_addr:str="10.0.0.217",x=.9,dirichlet_a=.3,search_depth=200):
		
		self.game_obj 		= game_obj 
		self.device	 		= device
		self.uid 			= game_id
		self.depth			= 0 
		self.search_complete= False
		self.search_depth 	= search_depth 

		if base_node: 
			self.root 			= base_node
			self.root.parent 	= None
		else:
			self.root 			= Node(game_obj,0,None)
			self.root.parent	= None 

		self.noise_gen	 		= numpy.random.default_rng().dirichlet 
		self.softmax_fn 		= softmax 

		self.x 					= x 
		self.dirichlet_a 		= dirichlet_a

		self.nodes 				= {}


		

	def pre_network_call(self):
	
		self.network_call_ready = False 
		node 					= self.root 
		starting_move 			= 1 if self.root.game_obj.board.turn == chess.WHITE else -1

		#Find best leaf node 
		node 					= self.get_best_node_max(node)

		#Add node to global list
		if not node.fen in self.nodes:
			self.nodes[node.fen]	= [node]
		else:
			self.nodes[node.fen].append(node)

		#Check if game over
		result:float or bool 	= node.game_obj.is_game_over()

		self.depth 	+= 1 
		if self.depth > self.search_depth:
			self.search_complete = True 
			if not hasattr(self,"current_node"):
				self.current_node 		= node 
			return 
		
		#If endgame, recursively call pre_network_call until not an endgame node 
		if not result is None:
			if result == 0:
				v = 0 
			elif result == starting_move:
				v = 1 
			else:
				v = -1 
			
			v = float(v)

			for identical_node in self.nodes[node.fen]:
				identical_node.bubble_up(v)
			
			self.pre_network_call()
		
		else:
			self.current_node 		= node 
			self.network_call_ready = False 
			

	def post_network_call(self,prob,v):
		if self.search_complete:
			return
		#Process Node 
		node 						= self.current_node
		prob_cpu:torch.Tensor		= prob.to(torch.device('cpu'),non_blocking=True).numpy()
		legal_moves 				= node.game_obj.get_legal_moves()
		legal_probs 				= numpy.array([prob_cpu[i] for i in legal_moves])
		noise 						= self.noise_gen([self.dirichlet_a for _ in range(len(legal_probs))],1)*(1-self.x)
		legal_probs					= self.softmax_fn(legal_probs*self.x + noise)

		node.children 				= {move_i : Node(node.game_obj.copy() ,p=p,parent=node) for p,move_i in zip(legal_probs,legal_moves)} 

		for move in node.children:
			node.children[move].game_obj.make_move(move)
			node.children[move].move 	= move	
		
		v = float(v)

		for identical_node in self.nodes[node.fen]:
			identical_node.bubble_up(v)		


	def get_best_node_max(self,node:Node):
		while node.children:
				
				best_node 			= max(list(node.children.values()),key = lambda x: x.get_score())
				node 				= best_node

		return node
	

	def get_policy(self):
		return {move:self.root.children[move].num_visited for move in self.root.children}

def get_best_node_max(node:Node):
	while node.children:
			
			best_node 			= max(list(node.children.values()),key = lambda x: x.get_score())
			node 				= best_node

	return node
	

def pre_network_call(root_node,nodes_table):
	node 					= root_node
	starting_move 			= 1 if root_node.game_obj.board.turn == chess.WHITE else -1

	#Find best leaf node 
	node 					= get_best_node_max(node)

	#Add node to global list
	if not node.fen in nodes_table:
		nodes_table[node.fen]= [node]
	else:
		nodes_table[node.fen].append(node)

	#Check if game over
	result:float or bool 	= node.game_obj.is_game_over()
	
	#If endgame, recursively call pre_network_call until not an endgame node 
	if not result is None:
		if result == 0:
			v = 0 
		elif result == starting_move:
			v = 1 
		else:
			v = -1 
		
		v = float(v)

		for identical_node in nodes_table[node.fen]:
			identical_node.bubble_up(v)
		
		nodes_table	= pre_network_call(node,nodes_table)
	
	else:
		return (node,nodes_table) 


def post_network_call(current_node,prob,v,nodes_table,dirichlet_a=.3,x=.95,noise_gen=numpy.random.default_rng().dirichlet):

	#Process Node 
	node 						= current_node
	prob_cpu:torch.Tensor		= prob.to(torch.device('cpu'),non_blocking=True).numpy()
	legal_moves 				= node.game_obj.get_legal_moves()
	legal_probs 				= numpy.array([prob_cpu[i] for i in legal_moves])
	noise 						= noise_gen([dirichlet_a for _ in range(len(legal_probs))],1)*(1-x)
	legal_probs					= softmax(legal_probs*x + noise)

	node.children 				= {move_i : Node(node.game_obj.copy() ,p=p,parent=node) for p,move_i in zip(legal_probs,legal_moves)} 

	for move in node.children:
		node.children[move].game_obj.make_move(move)
		node.children[move].move 	= move	
	
	v = float(v)

	for identical_node in nodes_table[node.fen]:
		identical_node.bubble_up(v)		


if __name__ == "__main__":
	pass
