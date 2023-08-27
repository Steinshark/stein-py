import pickle 
import time 
import socket 
import torch 
import numpy 
import networks
import games 
from rl_notorch import Tree
from sklearn.utils import extmath 
import random 
import os 
from torch.utils.data import DataLoader
import multiprocessing
import string 
import sys 
NETWORK_BUFFER_SIZE 			= 1024*16

def softmax(x):
		if len(x.shape) < 2:
			x = numpy.asarray([x],dtype=float)
		return extmath.softmax(x)[0]

def exchange_ints(fen):
	for i in range(1,9):
		fen 	= fen.replace(str(i),"e"*i) 
	return fen

def fen_to_tensor(fen_list):

	batch_size 		= len(fen_list)

	board_tensors 	= numpy.zeros(shape=(batch_size,6,8,8),dtype=float)

	piece_indx 		= {"R":4,"N":2,"B":3,"Q":5,"K":6,"P":1,"r":-4,"n":-2,"b":-3,"q":-5,"k":-6,"p":-1}
	
	#Go through FEN and fill pieces
	replaced_fens 	= map(exchange_ints,fen_list)

	for i,fen in enumerate(replaced_fens):
		try:
			position	= fen.split(" ")[0].split("/")
			turn 		= fen.split(" ")[1]
			castling 	= fen.split(" ")[2]
		except IndexError:
			print(f"weird fen: {fen}")
		
		#Place pieces
		for rank_i,rank in enumerate(reversed(position)):
			for file_i,piece in enumerate(rank): 
				if not piece == "e":
					board_tensors[i,0,rank_i,file_i]	= piece_indx[piece]  
		
		#Place turn 
		slice 		= 1 
		board_tensors[i,slice,:,:]   = numpy.ones(shape=(1,8,8),dtype=float) * 1 if turn == "w" else -1

		#Place all castling allows 
		for castle in ["K","Q","k","q"]:
			slice += 1
			board_tensors[i,slice,:,:]	= numpy.ones(shape=(1,8,8),dtype=float) * 1 if castle in castling else 0

	return board_tensors


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
	
	def __init__(self,queue_cap=16,max_moves=200,search_depth=800,socket_timeout=.0004,start_gen=0,timeout=.01,server_ip="10.0.0.60"):
		self.queue          	= {} 
		self.socket    			= socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		self.socket.bind((server_ip,6969))
		self.server_ip 			= server_ip 
		self.socket_timeout 	= socket_timeout
		self.model 				= networks.ChessSmall()
		self.model.eval()
		self.frozen_model 		= torch.jit.freeze(torch.jit.trace(self.model,[torch.randn((queue_cap,6,8,8)).to("cuda")]))

		self.cur_model 			= 0 
		self.gen 				= 0 
		self.train_tresh		= 32
		self.new_gen_thresh		= 128
		self.lookup_table 		= {} 

		#GAME OPTIONS 
		self.max_moves 			= max_moves
		self.search_depth 		= search_depth

		#QUEUE OPTIONS 
		self.queue_cap			= queue_cap
		self.checked_updates	= 0
		self.timeout			= timeout
		self.original_queue_cap = queue_cap

		#METRICS 
		self.sessions 			= [] 
		self.queue_maxs			= [] 
		self.games_finished 	= [0]
		self.generations 		= [] 
		self.n_games_finished 	= 0 
		self.n_moves 			= 0 
		self.lookups 			= 0 

		#TIME METRICS 
		self.serve_times		= 0
		self.compute_iter		= 0
		self.tensor_times		= 0
		self.compute_times		= 0
		self.pickle_times		= 0

		self.server_start 		= None
		self.started 			= False 

		self.DATASET_ROOT  		=	 r"\\FILESERVER\S Drive\Data\chess"
		self.load_model(self.model,start_gen)

		#TEST MODEL 
		test_input 				= torch.randn(size=(1,6,8,8),device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),dtype=torch.float)
		self.model.forward(test_input)


	def run_server(self,update_every=10):
		self.started 			= True 
		self.server_start_time	= time.time()
		self.next_update_time 	= update_every
		self.update_freq		= update_every
		self.chunk_fills 		= [] 
		self.chunk_maxs			= [] 


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
			except ConnectionResetError as cre:
				print(f"\t{Color.RED}Connection Reset - Idling 2s{Color.END}")
				print(f"{Color.RED}{cre}{Color.END}\n")
				time.sleep(2)
				self.socket.close()
				self.socket    			= socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
				self.socket.bind((socket.gethostname(),6969))
				print(f"\t{Color.GREEN}Socket Reset{Color.END}")


	def fill_queue(self):
		
		self.queue.clear()
		start_listen_t  		= time.time()
		iters 					= 0
		found_something 		= False 
		#self.socket    			= socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		#self.socket.bind((self.server_ip,6969))
		#self.socket.settimeout(self.socket_timeout)
		while len(self.queue) < self.queue_cap and ((time.time()-start_listen_t) < self.timeout):
			iters += 1 

			#Listen for a connection
			try:
				data,addr            	= self.socket.recvfrom(NETWORK_BUFFER_SIZE)
				fen,game_id,gen        = pickle.loads(data) 

				#Check for gameover notification 
				if fen 	== "gameover":
					self.n_games_finished += 1
				
				else:
					if not gen in self.generations:
						self.generations.append(gen)

					self.queue[addr]      	= fen 
					iters 					+= 1
					self.n_moves			+= 1 
				found_something 				= True 
					
			#Idle 
			except TimeoutError:
				t1 						= time.time()

		self.chunk_fills.append(len(self.queue))
		self.chunk_maxs.append(self.queue_cap)
		#print(f"had {len(self.queue)} / {self.queue_cap}")
		self.sessions.append(len(self.queue))
		self.queue_maxs.append(self.queue_cap)

		self.sessions				= self.sessions[-5000:]
		self.queue_maxs				= self.queue_maxs[-5000:]

		self.chunk_fills			= self.chunk_fills[-5000:]
		self.chunk_maxs				= self.chunk_maxs[-5000:]


	def process_queue(self):

		if not self.queue:
			return

		#Send boards through model 
		returnables     		= [] 
		t_compute 	 			= time.time()
		encodings   			= torch.from_numpy(fen_to_tensor(self.queue.values())).float().to(torch.device('cuda'))
		self.tensor_times		+= time.time()-t_compute
		t_compute				= time.time()

		with torch.no_grad():
			probs,v     		= self.frozen_model.forward(encodings)
			probs 				= probs.type(torch.float16).cpu().numpy()
			v					= v.cpu().numpy()
		self.compute_times 		+= time.time()-t_compute

		#Pickle objects
		t_pickle 				= time.time()
		for prob,score,addr in zip(probs,v,self.queue.keys()):
			returnables.append(pickle.dumps((prob,score)))

		self.pickle_times 		+= time.time()-t_pickle

		#Return all computations
		t_send 					= time.time()
		for addr,returnable in zip(self.queue,returnables):
			self.socket.sendto(returnable,addr)

		self.serve_times 		+= time.time()-t_send
		
		self.compute_iter 		+= 1
			
	
	def update(self):

		
		if (self.checked_updates % 200 == 0) and sum(self.sessions[-1000:])/len(self.sessions[-1000:]) > 0:
			self.queue_cap	= max(self.sessions[-1000:])

		#Reset on low 
		no_reset_flag 	= False 
		if self.process_start-self.fill_start > .5:
			self.queue_cap 	= self.sessions[-1]
			no_reset_flag	= True 
		
		#add every 20 
		if self.checked_updates % 400 == 0 and self.queue_cap < self.original_queue_cap and not no_reset_flag:
			self.queue_cap 			+= 1
		
		#Check for train 
		if self.n_games_finished % self.train_tresh == 0 and not self.n_games_finished in self.games_finished:
			print(f"\n\t{Color.TAN}Finished {self.n_games_finished} games in {(time.time()-self.games_start):.2f}s{Color.END}")
			self.train(epochs=4,n_samples=4096,bs=32)
			self.games_start = time.time()
			self.games_finished.append(self.n_games_finished)
			self.save_model(self.model,gen=self.gen)
			
			#Update gen
			if self.n_games_finished % self.new_gen_thresh == 0:
				self.gen	+= 1
				print(f"\n\n\t{Color.GREEN}UPDATED MODEL GENERATION -> {self.gen}\n\n")

				#Duel models
				if len(self.get_generations()) > 3:
					self.duel_muiltithread(self.get_generations(),25,self.max_moves,self.search_depth,self.cur_model,4)
					self.load_model(self.model,self.cur_model)

		self.checked_updates += 1 


	def display_upate(self,update_every=10):

		if not self.started:
			return 
		
		cur_time    			= round(time.time()-self.server_start_time,2)
		cycle_time 				= f"{(self.update_start - self.fill_start):.4f}"
		#Update every n seconds 
		if cur_time > self.next_update_time:

			#Get numbers over last chunk 
			percent_served 			= 100*(sum(self.chunk_fills) / (sum(self.chunk_maxs)+.001))

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
			percent_served	= f"{percent_served:.2f}".ljust(6)
			telemetry_out += f"\t{Color.BLUE}Cap:{color} {percent_served}%{Color.TAN}\t{Color.BLUE}Max:{Color.GREEN}{self.queue_cap}"
			#Add process time
			telemetry_out += f"\t{Color.BLUE}Net:{Color.GREEN}{(self.process_start-self.fill_start):.4f}s\t{Color.BLUE}Comp:{Color.GREEN}{(self.update_start-self.process_start):.4f}s\t{Color.BLUE}Iter:{Color.GREEN}{cycle_time}s\t{Color.BLUE}Games:{Color.GREEN}{self.n_games_finished}{Color.END}"
			
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


	def duel_muiltithread(self,available_models,n_games,max_moves,search_depth,cur_model=0,n_threads=5):
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

		#Play cur_model_net as White
		args 		= [(cur_model_net,challenger_model_net,search_depth,max_moves) for _ in range(n_games)]
		with multiprocessing.Pool(n_threads) as pool:
			results 	= pool.starmap(self.play_models,args)

		for result in results:
			if result == 1:
				current_best_games += 1 
			elif result == -1:
				challenger_games += 1
			elif result == 0:
				tiegames += 1

		#Play challenger_model_net as White
		args 		= [(challenger_model_net,cur_model_net,search_depth,max_moves) for _ in range(n_games)]
		with multiprocessing.Pool(n_threads) as pool:
			results 	= pool.starmap(self.play_models,args)
		for result in results:
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


	def save_model(self,model:networks.FullNet,gen=1,count=1):
		torch.save(model.state_dict(),self.DATASET_ROOT+f"\models\gen{gen}")

		try:
			state_dict 	 = torch.load(self.DATASET_ROOT+f"\models\gen{gen}")
			model.load_state_dict(state_dict)
		except RuntimeError as re:
			if count > 5:
				print(f"{Color.RED}\tFailed to save model: {gen}{Color.END}")
			else:
				time.sleep(.1)
				self.save_model(model,gen,count+1)
	

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
		model 						= self.model.float().train()
		root                        = self.DATASET_ROOT+f"\experiences\gen{gen}"
		experiences                 = []
		model.train()
		if not os.listdir(root):
			print(f"No data to train on")
			return


		print(f"{Color.TAN}\t\tbegin Training:{Color.END}",end='')
		game_ids 	= set()
		for file in os.listdir(root):
			for letter in string.ascii_lowercase+string.ascii_uppercase+"_.":
				file = file.replace(letter,"")
			
			game_ids.add(int(file))

		for game_i in game_ids:
			try:
				states                      = torch.from_numpy(numpy.load(f"{root}/game_{game_i}_states.npy")).to(DEV).type(torch.float)
				pi                          = torch.from_numpy(numpy.load(f"{root}/game_{game_i}_localpi.npy")).to(DEV).type(torch.float)
				results                     = torch.from_numpy(numpy.load(f"{root}/game_{game_i}_results.npy")).to(DEV).type(torch.float)
				for i in range(len(states)):
					experiences.append((states[i],pi[i],results[i]))
			except FileNotFoundError:
				pass 
			except RuntimeError:
				pass
			except pickle.UnpicklingError:
				pass

		
		print(f"{Color.TAN}\tloaded {len(experiences)} datapoints in gen {gen}{Color.END}")

		if len(experiences) == 0:
			print(f"No data to train on")
			return
		

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
		self.model.eval()
		self.frozen_model 		= torch.jit.freeze(torch.jit.trace(self.model,[torch.randn((queue_cap,6,8,8)).to("cuda")]))
		print(f"\n")


if __name__ == "__main__":

	queue_cap 			= 10 
	max_moves 			= 200 
	search_depth 		= 225
	
	for arg in sys.argv:
		if "queue_cap=" in arg:
			queue_cap=int(arg.replace("queue_cap=",""))
		elif "iter_depth=" in arg:
			iter_depth=int(arg.replace("iter_depth=",""))
		elif "max_moves=" in arg:
			max_moves=int(arg.replace("max_moves=",""))
	chess_server 	= Server(queue_cap=queue_cap,max_moves=max_moves,search_depth=search_depth,server_ip=socket.gethostbyname(socket.gethostname()))
	chess_server.run_server(20)