import socket 
import random 
import time 
import numpy 
import torch
from networks import ChessNet, FullNet
import pickle 
import sys 

socket.setdefaulttimeout(.00001)
#						 .002	


def fen_to_tensor(fen,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

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

	return torch.tensor(board_tensor,dtype=torch.float,device=device,requires_grad=False)

def save_model(model:FullNet,gen=1):
	torch.save(model.state_dict(),f"C:/data/chess/models/gen{gen}")


def load_model(model:FullNet,gen=1,verbose=False):
	while True:
		try:
			model.load_state_dict(torch.load(f"C:/data/chess/models/gen{gen}"))
			if verbose:
				print(f"\tloaded model gen {gen}")
			return 
		
		except FileNotFoundError:
			gen -= 1 
			if gen < 0:
				print(f"\tloaded stock model gen {gen}")
				return
	
class Color:
	HEADER = '\033[95m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	RED = '\033[91m'
	END = '\033[0m'
	TAN = '\033[93m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	

#Server Code 

#TODO:
#	Incorporate a dynamic lookup table to reduce 
# 	inference forward passes.   -challenge will be to drop indices on forw pass and add them back in properly 
if __name__ == "__main__":

	trailer             			= 0
	trailer2            			= 0 
	og_lim              			= 10 
	queue_fill_cap               	= 10 
	timeout_thresh      			= .005
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
	if len(sys.argv) >= 2:
		queue_fill_cap 					= int(sys.argv[1])
	if len(sys.argv) >= 3:
		model_gen 						= int(sys.argv[2])

	
	sock    						= socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
	sock.bind((socket.gethostname(),6969))
	#sock.listen()  
	print(f"{Color.TAN}Server Online",flush =True)


	model   = ChessNet()
	queue   = {}
	server_start        = time.time()

	while i:
		listen_start        = time.time()

		#If no activity for 1 second, reload model, reset limit 
		if ((len(fills)+1) % int(1/timeout_thresh) == 0) and sum(fills[-1000:]) == 0:
			load_model(model,gen=model_gen,verbose=True)
			if len(sys.argv) >= 2:
				queue_fill_cap 					= int(sys.argv[1])
			
		#If same num clients for last 100, reduct to that num clients 
		if len(fills) > 1000 and abs(sum(fills[-1000:])/1000 - fills[-1]) < .0001 and fills[-1] > 0:
			queue_fill_cap = fills[-1]
		
		#Every ((len(fills)+1) % int(1/timeout_thresh) == 0)
		#Fill up queue or timeout after timeout_thresh seconds
		while len(queue) < queue_fill_cap and time.time()-listen_start < timeout_thresh:
			
			#Listen for a connection
			try:
				fen,addr            = sock.recvfrom(1024)
				fen                 = fen.decode() 
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
			encodings   	= torch.stack(list(map(fen_to_tensor, list(queue.values()))))
			tensor_times	+= time.time()-t_compute
			t_compute		= time.time()
			with torch.no_grad():
				probs,v     	= model.forward(encodings)
				probs 			= probs.type(torch.float16).cpu().numpy()
				v				= v.cpu().numpy()
			compute_times 	+= time.time()-t_compute

			#Pickle obkects
			t_pickle 		= time.time()
			for prob,score in zip(probs,v):
				pickled_prob    = pickle.dumps(prob)  #Convert probs to f16 to save bandwidth
				pickled_v       = pickle.dumps(score)
				returnables.append((pickled_prob,pickled_v))
			pickle_times 	+= time.time()-t_pickle

			#Return all computations
			t_send 			= time.time()
			for addr,returnable in zip(queue,returnables):
				prob,v     = returnable
				sock.sendto(prob,addr)
				sock.sendto(v,addr)
			serve_times += time.time()-t_send
			
			compute_iter += 1
			i += 1
		fills.append(len(queue))
			
		#Get serve stats 
		cur_time    = int(time.time()-server_start)
		serve_avg   += len(queue)
		serve_iter	+= 1 
		if cur_time >= trailer2:
			print(f"\tserved {len(queue)}/{queue_fill_cap}\t- avg: {100*(serve_avg/serve_iter)/queue_fill_cap:.2f}%\t- calc_t {1000*tensor_times/compute_iter:.2f}/{1000*compute_times/compute_iter:.2f}ms\t- pick_t: {1000*pickle_times/compute_iter:.2f}ms\t- traf_t: {1000*serve_times/compute_iter:.2f}ms\t- uptime: {(time.time()-server_start):.2f}s")
			trailer2+= 10

		queue = {}