import pickle 
import time 
import socket 
import torch 
import numpy 
import networks
import games 
from rlcopy import Tree
from sklearn.utils import extmath 
import random 
import os 


def softmax(x):
		if len(x.shape) < 2:
			x = numpy.asarray([x],dtype=float)
		return extmath.softmax(x)[0]


class Color:
	HEADER = '\033[95m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	RED = '\033[91m'
	END = '\033[0m'
	TAN = '\033[93m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'


class Server:

	def __init__(self,queue_cap=16,max_moves=120,search_depth=800,socket_timeout=.0002):
		self.queue          	= {} 
		socket.setdefaulttimeout(socket_timeout)
		self.socket    			= socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		self.socket.bind((socket.gethostname(),6969))

		self.model 				= networks.ChessSmall()

		#GAME OPTIONS 
		self.max_moves 			= max_moves
		self.search_depth 		= search_depth

		#QUEUE OPTIONS 
		self.queue_cap			= queue_cap
		self.checked_updates	= 0
		self.timeout			= .000002
		self.original_queue_cap = queue_cap

		#METRICS 
		self.sessions 			= [] 
		self.queue_maxs			= [] 
		self.games_finished 	= [0]
		self.n_games_finished 	= 0 
		self.n_moves 			= 0 

		#TIME METRICS 
		self.serve_times		= 0
		self.compute_iter		= 0
		self.tensor_times		= 0
		self.compute_times		= 0
		self.pickle_times		= 0

		self.server_start 		= None
		self.started 			= False 

		self.DATASET_ROOT  	=	 r"\\FILESERVER\S Drive\Data\chess3"


	def run_server(self,update_every=10):
		self.started 			= True 
		self.server_start_time	= time.time()
		self.next_update_time 	= update_every
		self.update_freq		= update_every
		self.chunk_fills 		= [] 
		self.chunk_maxs			= [] 


		self.games_start 	= time.time()
		while True:
			
			self.fill_start 	= time.time()
			self.fill_queue()
			self.process_start 	= time.time()
			self.process_queue()
			self.update_start	= time.time()
			self.update()
			self.display_upate()


	def fill_queue(self):
		
		self.queue           	= {}
		start_listen_t  		= time.time()
		iters 					= 0

		while len(self.queue) < self.queue_cap and ((time.time()-start_listen_t) < self.timeout):
			iters += 1 

			#Listen for a connection
			try:
				repr,addr            	= self.socket.recvfrom(1024)
				repr,game_id,gen        = pickle.loads(repr) 

				#Check for gameover notification 
				if isinstance(repr,str) and repr 	== "gameover":
					self.n_games_finished += 1
					continue 

				self.queue[addr]      	= repr 
				iters 					+= 1
				self.n_moves			+= 1 
				
			#Idle 
			except TimeoutError:
				t1 						= time.time()

		self.chunk_fills.append(len(self.queue))
		self.chunk_maxs.append(self.queue_cap)

		self.sessions.append(len(self.queue))
		self.queue_maxs.append(self.queue_cap)


	def process_queue(self):

		if not self.queue:
			return

		#Send boards through model 
		returnables     = [] 
		t_compute 	 	= time.time()
		encodings   	= torch.from_numpy(numpy.stack(list(self.queue.values()))).float().to(torch.device('cuda'))
		self.tensor_times	+= time.time()-t_compute
		t_compute		= time.time()

		with torch.no_grad():
			probs,v     	= self.model.forward(encodings)
			probs 			= probs.type(torch.float16).cpu().numpy()
			v				= v.cpu().numpy()
		self.compute_times 	+= time.time()-t_compute

		#Pickle obkects
		t_pickle 		= time.time()
		for prob,score in zip(probs,v):
			pickled_prob    = pickle.dumps(prob)
			pickled_v       = pickle.dumps(score)
			returnables.append((pickled_prob,pickled_v))
		self.pickle_times 	+= time.time()-t_pickle

		#Return all computations
		t_send 			= time.time()
		for addr,returnable in zip(self.queue,returnables):
			prob,v     = returnable
			sent 		= False 
			while not sent:
				try:
					self.socket.sendto(prob,addr)
					self.socket.sendto(v,addr)
					sent 	= True
				except TimeoutError:
					pass

		self.serve_times += time.time()-t_send
		
		self.compute_iter += 1
			
	
	#Update length if theres:
		#-over 1000 entries 
		#-average past 1000 > 1 
	def update(self):
		
		if len(self.sessions[-1000:]) == 1000 and sum(self.sessions[-1000:])/len(self.sessions[-1000:]) > 1:
			self.queue_cap	= max(self.sessions[-1000:])
		
		#add every 20 
		if self.checked_updates % 20 == 0 and self.queue_cap < self.original_queue_cap:
			self.queue_cap 			+= 1
		
		#Check for train 
		if self.n_games_finished % 100 == 0 and not self.n_games_finished in self.games_finished:
			print(f"\t{Color.RED}ive finished {self.n_games_finished} games {Color.END}")
			pass

	

	def display_upate(self,update_every=10):

		if not self.started:
			return 
		
		cur_time    			= round(time.time()-self.server_start_time,2)
		
		#Update every n seconds 
		if cur_time > self.next_update_time:

			#Get numbers over last chunk 
			percent_served 			= round(100*(sum(self.chunk_fills) / sum(self.chunk_maxs)))

			if percent_served < 50:
				color 					= Color.RED 
			elif percent_served < 75:
				color 					= Color.TAN 
			else:
				color 					= Color.GREEN 

			telemetry_out 			= ""

			#Add timeup 
			telemetry_out += f"\t{Color.BLUE}Uptime:{Color.TAN}{cur_time}"
			#Add served stats
			telemetry_out += f"\t{Color.BLUE}Cap:{color} {percent_served}%{Color.TAN}\tMax:{self.queue_cap}"
			#Add process time
			telemetry_out += f"\t{Color.BLUE}Net:{Color.GREEN}{(self.process_start-self.fill_start):.4f}s\t{Color.BLUE}Comp:{Color.GREEN}{(self.update_start-self.process_start):.4f}s\tGames:{self.n_games_finished}{Color.END}"
			
			print(telemetry_out)

			self.chunk_fills		= [] 
			self.chunk_maxs			= [] 

			self.next_update_time	+= self.update_freq


	def play_models(self,cur_model_net,challenger_model_net,search_depth,max_moves):

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
		

	def duel(self,available_models,n_games,search_depth,cur_model=0,max_moves=120):
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


	def get_generations(self):
		
		gens 	= [] 
		for model in os.listdir(self.DATASET_ROOT+f"\models\\"):
			gens.append(int(model.replace("gen","")))

		return gens 


	def save_model(self,model:networks.FullNet,gen=1):
		torch.save(model.state_dict(),self.DATASET_ROOT+f"\models\gen{gen}")


	def load_model(self,model:networks.FullNet,gen=1,verbose=False):
		while True:
			try:
				model.load_state_dict(torch.load(self.DATASET_ROOT+f"\models\gen{gen}"))
				if verbose:
					print(f"\tloaded model gen {gen}")
				return 
			
			except FileNotFoundError:
				gen -= 1 
				if gen <= 0:
					print(f"\tloaded stock model gen {gen}")
					return


	def get_n_games(self):
		files 	= os.listdir(self.DATASET_ROOT+f"/experiences/{self.generation}")