import random 
import time 
import os 
import socket 
import numpy 
import pickle 
import multiprocessing 
from rl_notorch import Node,Tree, softmax
import games
import sys 
DATASET_ROOT  	= r"//FILESERVER/S Drive/Data/chess"
KNOWN_HOSTS		= {"10.0.0.60":{"n_threads":8,"offset":2},"10.0.0.217":{"n_threads":16,"offset":1}}

def run_game(args):
	game_fn,move_limit,search_depth,game_id,gen,server_addr = args
	
	t0 						= time.time() 
	game:games.TwoPEnv 		= game_fn(max_moves=move_limit,gen=gen)
	mcts_tree 				= Tree(game,server_addr=server_addr)
	move_indices            = list(range(game.move_space))
	state_repr              = [] 
	state_pi                = [] 
	state_outcome           = [] 


	while game.get_result() is None:
		try:
			#Build a local policy 
			t1 = time.time()
			local_policy 		= mcts_tree.update_tree(iters=search_depth)
			local_softmax 		= softmax(numpy.asarray(list(local_policy.values()),dtype=float))

			#print(f"policy of {local_policy} in {(time.time()-t1):.2f}s\nexplored:{sum(list(local_policy.values()))}")
			for key,prob in zip(local_policy.keys(),local_softmax):
				local_policy[key] = prob

			#construct trainable policy 
			pi              	= numpy.zeros(game.move_space)
			for move_i,prob in local_policy.items():
				pi[move_i]    		= prob 

			#sample move from policy 
			next_move           = random.choices(move_indices,weights=pi,k=1)[0]

			#Add experiences to set 
			state_repr.append(game.get_repr(numpy=True))
			state_pi.append(pi)
			game.make_move(next_move)

			mcts_tree 				= Tree(game,server_addr=server_addr)
			game.is_game_over()
		except RecursionError:
			pass

	del mcts_tree
	
	if isinstance(model,str):
		send_gameover(server_addr,6969)

	#Check game outcome 
	if game.is_game_over() == 1:
		state_outcome = numpy.ones(len(state_repr),dtype=numpy.int8)
	elif game.is_game_over() == -1:
		state_outcome = numpy.ones(len(state_repr),dtype=numpy.int8) * -1 
	else:
		state_outcome = numpy.zeros(len(state_repr),dtype=numpy.int8)
	
	print(f"\tgame no. {game_id}\t== {game.get_result()}\tafter\t{game.move} moves in {(time.time()-t0):.2f}s\t {(time.time()-t0)/game.move:.2f}s/move")
	#print(game_board)
	
	#Save tensors
	if not os.path.isdir(DATASET_ROOT+f"/experiences/gen{gen}"):
		os.mkdir(DATASET_ROOT+f"/experiences/gen{gen}")

	#Save tensors
	
	state_repr 				= numpy.stack(state_repr)
	state_pi 				= numpy.stack(state_pi) 
	numpy.save(DATASET_ROOT+f"\experiences\gen{gen}\game_{game_id}_localpi",state_pi.astype(numpy.float16))
	numpy.save(DATASET_ROOT+f"\experiences\gen{gen}\game_{game_id}_states",state_repr.astype(numpy.float16))
	numpy.save(DATASET_ROOT+f"\experiences\gen{gen}\game_{game_id}_results",state_outcome.astype(numpy.float16))


	return game_id,time.time()-t0, game.move

def send_gameover(ip,port):
	try:
		sock 			= socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
		sock.sendto(pickle.dumps(("gameover",None,None)),(ip,port))
		sock.close()
	except:
		print(f"err in send")
		time.sleep(.1)
		send_gameover(ip,port)


if __name__ == "__main__" and True:
	

	n_threads 			= KNOWN_HOSTS[socket.gethostbyname(socket.gethostname())]['n_threads']
	n_games 			= 128 
	gen 				= 0 
	offset 				= KNOWN_HOSTS[socket.gethostbyname(socket.gethostname())]['offset']

	if not len(sys.argv) > 1:
		print(f"specify server IP")
		exit() 

	server_addr 		= sys.argv[1]

	#while True:
	while True:
		print(f"\n\n\tTraining iter {gen}\n\tn_threads:{n_threads}\n\toffset:{offset}\n\tgen:{gen}\n\n")
		time.sleep(.1)
		t0 = time.time()
		#play out games  
		with multiprocessing.Pool(n_threads,maxtasksperchild=None) as pool:
			pool.map(run_game,[(games.Chess,200,225,i+(10000*offset),gen,server_addr) for i in range(n_games)])
		
		print(f"ran {n_games} in {(time.time()-t0):.2f}s")
		#run_game((games.Chess,"NETWORK",10,225,10000,0,server_addr))
		#run_game((games.Chess,networks.ChessSmall(),10,225,10000,0,server_addr))

if __name__ == "__main__" and False:


	server_addr 		= sys.argv[1]
	import networks
	model 				= networks.ChessSmall()
	model.eval()
	iter 				= 0  
	#play out games  
	id,t,moves 	= run_game((games.Chess,20,225,0,0,server_addr))
	print(f"finished game in {t:.2f}s\t{t/moves:.3f}s/move")


