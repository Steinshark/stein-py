import socket 
import random 
import time 
import numpy 
import torch
import networks 
from networks import ChessNet, FullNet, ChessNetCompat, ChessDataset
import pickle 
import sys 
import warnings
import os 
from torch.utils.data import DataLoader

warnings.simplefilter('ignore')

socket.setdefaulttimeout(.00004)
DATASET_ROOT  	=	 r"\\FILESERVER\S Drive\Data\chess"

class Color:
	HEADER = '\033[95m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	RED = '\033[91m'
	END = '\033[0m'
	TAN = '\033[93m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

#check for connection
if os.listdir(DATASET_ROOT):
	print(f"{Color.GREEN}SERVER CHECK - GOOD {Color.END}") 

def fen_to_tensor(fen,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

	#Encoding will be an 8x8 x n tensor 
	#	7 for whilte, 7 for black 
	#	4 for castling 7+7+4 
	# 	1 for move 
	#t0 = time.time()
	#board_tensor 	= torch.zeros(size=(1,19,8,8),device=device,dtype=torch.float,requires_grad=False)
	board_tensor 	= numpy.zeros(shape=(17,8,8))
	piece_indx 	= {"R":0,"N":1,"B":2,"Q":3,"K":4,"P":5,"r":6,"n":7,"b":8,"q":9,"k":10,"p":11}
	#Go through FEN and fill pieces
	for i in range(1,9):
		fen 	= fen.replace(str(i),"e"*i)

	splitted    = fen.split(" ")
	position	= splitted[0].split("/")
	turn 		= splitted[1]
	castling 	= splitted[2]
	
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

	return board_tensor
	#return torch.tensor(board_tensor,dtype=torch.float,device=device,requires_grad=False)


def fen_to_tensor_no_castle(fen,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

	#Encoding will be an 8x8 x n tensor 
	#	7 for whilte, 7 for black 
	#	4 for castling 7+7+4 
	# 	1 for move 
	#t0 = time.time()
	#board_tensor 	= torch.zeros(size=(1,19,8,8),device=device,dtype=torch.float,requires_grad=False)
	board_tensor 	= numpy.zeros(shape=(13,8,8))
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

	#return torch.tensor(board_tensor,device=device,dtype=torch.float,requires_grad=False)
	return board_tensor


def save_model(model:FullNet,gen=1):
	torch.save(model.state_dict(),DATASET_ROOT+f"\models\gen{gen}")


def load_model(model:FullNet,gen=1,verbose=False,tablesize=0):
	while True:
		try:
			model.load_state_dict(torch.load(DATASET_ROOT+f"\models\gen{gen}"))
			if verbose:
				print(f"\t\t{Color.BLUE}loaded model gen {gen} - lookup table size {tablesize}{Color.END}")
			return 
		
		except FileNotFoundError:
			gen -= 1 
			if gen < 0:
				print(f"\tloaded stock model gen {gen}")
				return
	

def train(model:networks.FullNet,n_samples,gen,bs=8,epochs=5,DEV=torch.device('cuda' if torch.cuda.is_available else 'cpu')):
	model = model.float()
	root                        = DATASET_ROOT+f"\experiences\gen{gen}"
	experiences                 = []

	if not os.listdir(root):
		print(f"No data to train on")
		return


	print(f"{Color.TAN}\n\n\tbegin Training:{Color.END}")
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
		
		print(f"\t\t{Color.BLUE}Epoch {epoch_i} loss: {total_loss/batch_i:.3f} with {len(train_set)}/{len(experiences)}{Color.END}")



def check_train(n=128):
	
	#Check for n experiences 
	exp_list	= os.listdir(DATASET_ROOT+"\experiences\gen1")
	if int(len(exp_list)/3) > 128:
		train()

#Server Code 

#TODO:
#	Incorporate a dynamic lookup table to reduce 
# 	inference forward passes.   -challenge will be to drop indices on forw pass and add them back in properly 
if __name__ == "__main__":
	trailer             			= 0
	trailer2            			= 0 
	og_lim              			= 10 
	queue_fill_cap               	= 10 
	timeout_thresh      			= .006
	serve_avg 						= 0 
	serve_times 					= 0 
	compute_times 					= 0 
	tensor_times					= 0 
	pickle_times 					= 0 
	serve_iter 						= 0 
	compute_iter					= 1
	model_gen 						= 1 
	i 								= 1
	fills 							= [] 
	queue   = {}
	lookup_table 					= {}
	use_lookups 					= True
	lookup_ply_len					= 25
	lookups 						= 0
	total_lookups	 				= 0 
	exp_trt 						= True 
	train_thresh					= 16
	play_table 						= {}
	trained_on						= [0,16]

	if len(sys.argv) >= 2:
		queue_fill_cap 					= int(sys.argv[1])
		old_cap							= queue_fill_cap
	if len(sys.argv) >= 3:
		model_gen 						= int(sys.argv[2])

	
	sock    						= socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
	sock.bind((socket.gethostname(),6969))
	print(f"{Color.GREEN}Server Online{Color.END}",flush =True)
	#torch.backends.cudnn.benchmark = True

	model 	= ChessNet(optimizer=torch.optim.SGD,optimizer_kwargs={"lr":2e-5,"weight_decay":2.5e-6,"momentum":.75,'nesterov':True},n_ch=17,device=torch.device('cuda'),n_layers=20)
	server_start        = time.time()



	while i:
		listen_start        = time.time()



		#Train model every 100 games 
		if (len(play_table) % train_thresh == 0) and not len(play_table) in trained_on:

			#Reset the model 
			del model 
			model 				= ChessNet(optimizer=torch.optim.SGD,optimizer_kwargs={"lr":2e-5,"weight_decay":2.5e-6,"momentum":.75,'nesterov':True},n_ch=17,device=torch.device('cuda'),n_layers=20)
			
			#Train model on random sample of games 
			load_model(model,gen=model_gen,verbose=True)
			train(model,8192,gen=model_gen,bs=8,epochs=1)
			save_model(model,gen=model_gen)
			trained_on.append(len(play_table))

			#Reset the lookup table 
			del lookup_table 
			lookup_table 		= {}

			#Set mode for training
			model.eval()
			if exp_trt:
				model 			= torch.jit.trace(model,[torch.randn((queue_fill_cap,17,8,8)).to("cuda")])
				model 			= torch.jit.freeze(model)
				model.loaded 	= True 
				print(f"\t\tConverted to JIT")
			if len(sys.argv) >= 2:
				queue_fill_cap 					= int(sys.argv[1])
			
			


		#If same num clients for last 1000, reduct to that num clients 
		if len(fills) > 1000  and fills[-1] > 0 and sum(fills[-1000:])/1000 > 1:
			queue_fill_cap = max(fills[-1000:])
		
		#Every 20000 add 1 
		if ((len(fills) % 5000) == 0) and queue_fill_cap < old_cap :
			queue_fill_cap += 1

		#Fill up queue or timeout after timeout_thresh seconds
		while len(queue) < queue_fill_cap and time.time()-listen_start < timeout_thresh:
			
			#Listen for a connection
			try:
				fen,addr            = sock.recvfrom(1024)
				fen                 = fen.decode() 
				game_id 			= fen.split('$')[0]
				if game_id in play_table:
					play_table[game_id]	+= 1 
				else:
					play_table[game_id] = 1 

				fen					= fen.split('$')[1]
				fen_stripped 		= " ".join(fen.split(' ')[:2])

				#if fen is in lookup, immediate send it back out 
				if use_lookups and fen_stripped in lookup_table:
					prob,v     	= lookup_table[fen_stripped]
					sent 		= False 
					while not sent:
						try:
							sock.sendto(prob,addr)
							sock.sendto(v,addr)
							sent 	= True
						except TimeoutError:
							pass
					lookups += 1
					continue
				else:
					queue[addr]         = fen 

			#Idle 
			except TimeoutError:
				cur_time    = int(time.time()-listen_start) % 10 == 0

				if cur_time == trailer and len(queue) == 0:
					print(f"idling")
					trailer += 1

		if queue:

			#Send boards through model 
			returnables     = [] 
			t_compute 	 	= time.time()
			#encodings   	= torch.stack([fen_to_tensor(fen,torch.device("cuda" if torch.cuda.is_available() else "cpu")) for fen in queue.values()])
			#encodings   	= torch.stack(list(map(fen_to_tensor_no_castle, list(queue.values()))))
			encodings   	= torch.from_numpy(numpy.asarray(list(map(fen_to_tensor, list(queue.values()))))).float().to(torch.device('cuda'))
			tensor_times	+= time.time()-t_compute
			t_compute		= time.time()

			with torch.no_grad():
				probs,v     	= model.forward(encodings)
				probs 			= probs.cpu().numpy()
				v				= v.cpu().numpy()
			compute_times 	+= time.time()-t_compute

			#Pickle obkects
			t_pickle 		= time.time()
			for prob,score in zip(probs,v):
				pickled_prob    = pickle.dumps(prob)
				pickled_v       = pickle.dumps(score)
				returnables.append((pickled_prob,pickled_v))
			pickle_times 	+= time.time()-t_pickle

			#Return all computations
			t_send 			= time.time()
			for addr,returnable in zip(queue,returnables):
				prob,v     = returnable
				sent 		= False 
				while not sent:
					try:
						sock.sendto(prob,addr)
						sock.sendto(v,addr)
						sent 	= True
					except TimeoutError:
						pass

				#Add to lookup table 
				if use_lookups:
					# unpacked_prob,unpacked_v = returnable 
					# unpacked_prob 	= unpacked_prob.float16()
					# v 				= v.float() 
					lookup_table[" ".join(queue[addr].split(' ')[:2])]	= returnable
			serve_times += time.time()-t_send
			
			compute_iter += 1
			i += 1
		fills.append(len(queue))
			
		#Get serve stats 
		cur_time    = int(time.time()-server_start)
		serve_avg   += len(queue)
		serve_iter	+= 1 
		if cur_time >= trailer2:
			if len(queue)/queue_fill_cap < .5:
				color = Color.RED 
			elif len(queue)/queue_fill_cap < .75:
				color = Color.TAN 
			else:
				color = Color.GREEN

			print(f"\t{Color.TAN}served {color}{len(queue)}/{queue_fill_cap}{Color.TAN}\tavg {color}{100*(serve_avg/serve_iter)/queue_fill_cap:.1f}{Color.TAN}%\tcalc_t {Color.BLUE}{1000*tensor_times/compute_iter:.2f}{Color.TAN}/{Color.BLUE}{1000*compute_times/compute_iter:.2f}{Color.TAN}ms\tpick_t {Color.BLUE}{1000*pickle_times/compute_iter:.2f}{Color.TAN}ms\ttraf_t {Color.BLUE}{1000*serve_times/compute_iter:.2f}{Color.TAN}ms\tuptime {Color.BLUE}{(time.time()-server_start):.2f}{Color.TAN}s lookups {Color.BLUE}{lookups}{Color.END}")
			total_lookups += lookups
			print(f"\t\t{Color.BLUE}table size:\t{len(lookup_table)}")
			print(f"\t\t{Color.BLUE}table lookups:\t{total_lookups}")
			lookups = 0 
			trailer2+= 10

		queue = {}