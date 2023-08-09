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
from torch.utils.data import DataLoader
from networks import ChessDataset

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

	def __init__(self,queue_cap=16,max_moves=200,search_depth=800,socket_timeout=.00004):
		self.queue          	= {} 
		socket.setdefaulttimeout(socket_timeout)
		self.socket    			= socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		self.socket.bind((socket.gethostname(),6969))

		self.model 				= networks.ChessSmall()
		self.cur_model 			= 0 
		self.gen 				= 0 
		self.train_tresh		= 16
		self.new_gen_thresh		= 64

		#GAME OPTIONS 
		self.max_moves 			= max_moves
		self.search_depth 		= search_depth

		#QUEUE OPTIONS 
		self.queue_cap			= queue_cap
		self.checked_updates	= 0
		self.timeout			= .002
		self.original_queue_cap = queue_cap

		#METRICS 
		self.sessions 			= [] 
		self.queue_maxs			= [] 
		self.games_finished 	= [0]
		self.generations 		= [] 
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

		self.DATASET_ROOT  	=	 r"\\FILESERVER\S Drive\Data\chess"


	def run_server(self,update_every=10):
		self.started 			= True 
		self.server_start_time	= time.time()
		self.next_update_time 	= update_every
		self.update_freq		= update_every
		self.chunk_fills 		= [] 
		self.chunk_maxs			= [] 

		self.load_model(self.model,5)

		self.games_start 	= time.time()
		while True:
			try:
				self.fill_start 	= time.time()
				self.fill_queue()
				self.process_start 	= time.time()
				self.process_queue()
				self.update_start	= time.time()
				self.update()
				self.display_upate()
			except ConnectionResetError:
				print(f"\t{Color.RED}Connection Reset - Idling 10s{Color.END}")
				time.sleep(10)
				self.socket    			= socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
				self.socket.bind((socket.gethostname(),6969))
				print(f"\t{Color.GREEN}Socket Reset{Color.END}\n\n")


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
			
				if not gen in self.generations:
					self.generations.append(gen)

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
			
	
	def update(self):
		
		if len(self.sessions[-1000:]) == 1000 and sum(self.sessions[-1000:])/len(self.sessions[-1000:]) > 1:
			self.queue_cap	= max(self.sessions[-1000:])
		
		#add every 20 
		if self.checked_updates % 20 == 0 and self.queue_cap < self.original_queue_cap:
			self.queue_cap 			+= 1
		
		#Check for train 
		if self.n_games_finished % self.train_tresh == 0 and not self.n_games_finished in self.games_finished:
			print(f"\n\t{Color.TAN}Finished {self.n_games_finished} games in {(time.time()-self.games_start):.2f}s{Color.END}")
			self.train(epochs=2,n_samples=2048,bs=32)
			self.games_start = time.time()
			self.games_finished.append(self.n_games_finished)
			self.save_model(self.model,gen=self.gen)
			
			#Update gen
			if self.n_games_finished % self.new_gen_thresh == 0:
				self.gen	+= 1
				print(f"\n\n\t{Color.GREEN}UPDATED MODEL GENERATION -> {self.gen}\n\n")

				#Duel models
				if len(self.get_generations()) > 3:
					self.duel(self.get_generations(),25,325,self.cur_model,300)

			
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
			percent_served	= str(percent_served).ljust(8)
			telemetry_out += f"\t{Color.BLUE}Cap:{color} {percent_served}%{Color.TAN}\tMax:{self.queue_cap}"
			#Add process time
			telemetry_out += f"\t{Color.BLUE}Net:{Color.GREEN}{(self.process_start-self.fill_start):.4f}s\t{Color.BLUE}Comp:{Color.GREEN}{(self.update_start-self.process_start):.4f}s\tGames:{self.n_games_finished}{Color.END}"
			
			print(telemetry_out)

			self.chunk_fills		= [] 
			self.chunk_maxs			= [] 

			self.next_update_time	+= self.update_freq


	def play_models(self,cur_model_net:torch.nn.Module,challenger_model_net:torch.nn.Module,search_depth,max_moves):

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
		self.load_model(cur_model_net,gen=cur_model,verbose=True)
		self.load_model(challenger_model_net,gen=challenger_model,verbose=True) 
		cur_model_net.eval()
		challenger_model_net.eval()

		#Keep track of how each model does
		current_best_games 		= 0
		challenger_games 		= 0 
		tiegames    			= 0

		#Play cur_model_net as X 
		for game_i in range(n_games):
			result 	= self.play_models(cur_model_net,challenger_model_net,search_depth=search_depth,max_moves=max_moves)

			if result == 1:
				current_best_games += 1 
			elif result == -1:
				challenger_games += 1
			elif result == 0:
				tiegames += 1

		#Play challenger_model_net as X 
		for game_i in range(n_games):
			result 	= self.play_models(challenger_model_net,cur_model_net,search_depth=search_depth,max_moves=max_moves)

			if result == 1:
				challenger_games += 1 
			elif result == -1:
				current_best_games += 1
			elif result == 0:
				tiegames += 1
		
		challenger_ratio 	= ((challenger_games) / (current_best_games+challenger_games+.01))

		if challenger_ratio >= .55:
			best_model 			= challenger_model
			worst_model			= cur_model

		


		print(f"\t{Color.GREEN}Cur model{cur_model}: {current_best_games}\tChallenger model{challenger_model}: {challenger_games}\ttie: {tiegames}\n")

		#Delete worst model 
		print(f"\t{Color.GREEN}best model is {best_model}{Color.END}")
		print(f"\t{Color.RED}removing {worst_model}{Color.END}")
		os.remove(self.DATASET_ROOT+f"\\models\\gen{worst_model}")
		self.cur_model 	= best_model


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
	

	def train(self,n_samples=2048,bs=32,epochs=2,DEV=torch.device('cuda' if torch.cuda.is_available else 'cpu')):
		gen 						= max(self.generations)
		model 						= self.model.float()
		root                        = self.DATASET_ROOT+f"\experiences\gen{gen}"
		experiences                 = []
		model.train()
		if not os.listdir(root):
			print(f"No data to train on")
			return


		print(f"{Color.TAN}\t\tbegin Training:{Color.END}",end='')
		for game_i in range(500):
			try:
				states                      = torch.load(f"{root}/game_{game_i}_states").float().to(DEV)
				pi                          = torch.load(f"{root}/game_{game_i}_localpi").float().to(DEV)
				results                     = torch.load(f"{root}/game_{game_i}_results").float().to(DEV)
				for i in range(len(states)):
					experiences.append((states[i],pi[i],results[i]))
			except FileNotFoundError:
				pass 
			except RuntimeError:
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

