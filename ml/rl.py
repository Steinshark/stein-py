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
import pycuda.driver as cuda_driver
import numpy 
from sklearn.utils import extmath 

def softmax(x):
	if len(x.shape) < 2:
		x = numpy.asarray([x],dtype=float)
	return extmath.softmax(x)[0]

sys.path.append("C:/gitrepos/steinpy/ml")
class QLearner:

	def __init__(	self,
		  		environment,
				model_fn,
				model_kwargs,
				loss_fn=torch.nn.MSELoss,
				optimizer_fn=torch.optim.Adam,
				optimizer_kwargs={"lr":1e-5},
				device=torch.device('cpu'),
				verbose=True,
				path="models"):
		

		#Set model vars  
		self.device 	        				= device

		#Set runtime vars 
		self.verbose 							= verbose
		self.path 								= path 

		#Set telemetry vars 
		self.best_score							= 0

		#Training Vars
		self.environment						= environment
		self.model_fn         					= model_fn 
		self.model_kwargs 						= model_kwargs

		self.model_kwargs['loss_fn']			= loss_fn
		self.model_kwargs['optimizer']			= optimizer_fn
		self.model_kwargs['optimizer_kwargs']	= optimizer_kwargs

	def init_training(self,
		   iters=10000,
		   train_freq=10,
		   update_freq=10,
		   sample_size=512,
		   batch_size=16,
		   pool_size=10000,
		   epsilon=.25,
		   gamma=.9,
		   env_kwargs={},
		   run_kwargs={}
		   ):
		
		self.iters				= iters 
		self.train_freq			= train_freq
		self.update_freq		= update_freq
		self.sample_size 		= sample_size
		self.batch_size 		= batch_size
		self.pool_size		= pool_size
		self.epsilon			= epsilon
		self.gamma 				= gamma 

		self.env_kwargs 		= env_kwargs
		self.run_kwargs 		= run_kwargs
		self.target_model		= self.model_fn(**self.model_kwargs)
		self.learning_model		= self.model_fn(**self.model_kwargs)

		self.experience_pool					= [0 for _ in range(self.pool_size)]

	def run_training(self):
	
		#	Sliding window memory update 
		#	Instead of copying a new memory_pool list 
		#	upon overflow, simply replace the next window each time 
		self.tstart 			= time.time()
		self.exp_i 				= 0

		#	Train 
		for i in range(self.iters):

			iter_t0					= time.time()
					
			#	UPDATE EPSILON
			self.update_epsilon(i/self.iters)	
		
			#	RUN ENVIRONMENT
			self.environment.init_environment(self.target_model,**self.env_kwargs)
			experiences,metrics  	= self.environment.run_environment(**self.run_kwargs)

			#	UPDATE MEMORY POOL 
			for exp in experiences:
				self.experience_pool[self.exp_i%self.pool_size] = exp 
				self.exp_i += 1		

			#	UPDATE VERBOSE 
			if self.verbose:
				print(f"[Episode {str(i).rjust(15)}/{int(self.iters)} -  {(100*i/self.iters):.2f}% complete  \tavg game len: {(sum(metrics['g_lens'])/len(metrics['g_lens'])):.2f}\t{(time.time()-self.tstart):.2f}s")
						
			# 	GET TRAINING SAMPLES
			#	AND TRAIN MODEL 
			if self.exp_i > self.sample_size:
				training_set 			= self.get_samples()
				self.train_on_experiences(training_set)
						
			#	UPDATE MODELS 
			if i % self.update_freq == 0 and i > 0:
				self.transfer_models()
			
			i += self.train_freq

		return self.cleanup()

	# WEIGHTED IS NOT IMPLEMENTED
	def get_samples(self,mode="random"):

		if mode == "random":
			return random.sample(self.experience_pool[:self.exp_i],self.sample_size)
		
		if mode == "weighted":
			return random.sample(self.experience_pool[:self.exp_i],self.sample_size)

	def cleanup(self):
		blocked_scores		= reduce_arr(self.all_scores,self.x_scale)
		blocked_lived 		= reduce_arr(self.all_lived,self.x_scale)
		graph_name = f"{self.name}_[{str(self.loss_fn).split('.')[-1][:-2]},{str(self.optimizer_fn).split('.')[-1][:-2]}@{self.kwargs['lr']}]]]"

		if self.save_fig:
			plot_game(blocked_scores,blocked_lived,graph_name)

		if self.gui:
			self.output.insert(tk.END,f"Completed Training\n\tHighScore:{self.best_score}\n\tSteps:{sum(self.all_lived[-1000:])/1000}")
		return blocked_scores,blocked_lived,self.best_score,graph_name

	def train_on_experiences(self,sample_set,epochs=1):
		
		#Telemetry 
		if self.verbose:
			print(f"TRAINING:")
			print(f"\tDataset:\n\t\t{'size'.ljust(12)}: {len(sample_set)}\n\t\t{'batch_size'.ljust(12)}: {self.batch_size}\n\t\t{'epochs'.ljust(12)}: {epochs}\n\t\t{'lr'.ljust(12)}: {self.learning_model.optimizer.param_groups[0]['lr']:.6f}\n")

		#	Telemetry Vars 
		t0 									= time.time()
		t_gpu 								= 0
		num_equals 							= 40 
		printed 							= 0
		total_loss							= 0

		#	Telemetry
		if self.verbose:
			print(f"\tPROGRESS- [",end='')

		#	Do one calc for all runs 
		num_batches = int(len(sample_set) / self.batch_size)

		# 	Iterate through batches
		for batch_i in range(num_batches):

			i_start 							= batch_i*self.batch_size
			i_end   							= i_start + self.batch_size
			
			#	Telemetry
			percent 							= batch_i / num_batches
			if self.verbose:
				while (printed / num_equals) < percent:
					print("=",end='',flush=True)
					printed+=1
			
			#BELLMAN UPDATE 
			self.learning_model.optimizer.zero_grad()

			#Gather batch experiences
			batch_set 							= sample_set[i_start:i_end]

			init_states 						= torch.stack([exp['s']  for exp in batch_set]).type(torch.float)
			action 								= [exp['a'] for exp in batch_set]
			next_states							= torch.stack([exp['s`'] for exp in batch_set]).type(torch.float)
			rewards 							= [exp['r']  for exp in batch_set]
			done								= [exp['done'] for exp in batch_set]
			
			#Calc final targets 
			initial_target_predictions 			= self.learning_model.forward(init_states.to(self.device))
			final_target_values 				= initial_target_predictions.clone().detach()
			
			#print(f"\naction taken was {action[0]}")
			#print(f"init value was {initial_target_predictions[0][action[0]]}")
			#Get max from s`
			with torch.no_grad():
				stepped_target_predictions 			= self.target_model.forward(next_states.to(self.device))
				best_predictions 					= torch.max(stepped_target_predictions,dim=1)[0]

			#print(f"update value was {best_predictions[0]}")
			#Update init values 
			for i,val in enumerate(best_predictions):
				chosen_action						= action[i]
				final_target_values[i,chosen_action]= rewards[i] - (done[i] * self.gamma * val)
			
			#input(f"final value was {final_target_values[0][action[0]]}")
			#	Calculate Loss
			t1 									= time.time()
			batch_loss 							= self.learning_model.loss(initial_target_predictions,final_target_values)
			total_loss 							+= batch_loss.item()

			#Back Propogate
			batch_loss.backward()
			self.learning_model.optimizer.step()
			t_gpu 								+= time.time() - t1
		
		#	Telemetry
		if self.verbose:
			print(f"]\ttime: {(time.time()-t0):.2f}s\tt_gpu:{(t_gpu):.2f}\tloss: {(total_loss/num_batches):.6f}")
			print("\n\n")

	def transfer_models(self):
		if self.verbose:
			print("\ntransferring models\n\n")

		#Check for dir 
		if not os.path.isdir(self.path):
			os.mkdir(self.path)

		torch.save(self.learning_model.state_dict(),os.path.join(self.path,f"{self.environment.name}_lm_state_dict"))

		#Load the learning model as the target model
		self.target_model = self.model_fn(**self.model_kwargs)

		self.target_model.load_state_dict(torch.load(os.path.join(self.path,f"{self.environment.name}_lm_state_dict")))
		self.target_model = self.target_model.to(self.device)

	@staticmethod
	def update_epsilon(percent):
		radical = -.4299573*100*percent -1.2116290 
		if percent > .50:
			return 0
		else:
			return pow(2.7182,radical)


class DoubleQLearner:

	def __init__(	self,
		  		environment,
				model_fn,
				model_kwargs,
				loss_fn=torch.nn.MSELoss,
				optimizer_fn=torch.optim.Adam,
				optimizer_kwargs={"lr":1e-5},
				device=torch.device('cpu'),
				verbose=True,
				path="models",
				loading=False):
		
		#Set model vars  
		self.device 	        				= device

		#Set runtime vars 
		self.verbose 							= verbose
		self.path 								= path 

		#Set telemetry vars 
		self.best_score							= 0

		#Training Vars
		self.environment						= environment
		self.model_fn         					= model_fn 
		self.model_kwargs 						= model_kwargs

		self.model_kwargs['loss_fn']			= loss_fn
		self.model_kwargs['optimizer']			= optimizer_fn
		self.model_kwargs['optimizer_kwargs']	= optimizer_kwargs

		self.loading							= loading


	def init_training(self,
		   iters=10000,
		   train_freq=10,
		   update_freq=10,
		   sample_size=512,
		   batch_size=16,
		   exp_pool_size=10000,
		   epsilon=.25,
		   gamma=.9,
		   env_kwargs={},
		   run_kwargs={}
		   ):
		
		self.iters				= iters 
		self.train_freq			= train_freq
		self.update_freq		= update_freq
		self.sample_size 		= sample_size
		self.batch_size 		= batch_size
		self.exp_pool_size		= exp_pool_size
		self.epsilon			= epsilon
		self.gamma 				= gamma 

		self.env_kwargs 		= env_kwargs
		self.run_kwargs 		= run_kwargs
		self.learning_modelA	= self.model_fn(**self.model_kwargs).to(self.device)
		self.target_modelA		= self.model_fn(**self.model_kwargs).to(self.device)
		
		self.learning_modelB	= self.model_fn(**self.model_kwargs).to(self.device)
		self.target_modelB		= self.model_fn(**self.model_kwargs).to(self.device)

		self.experience_poolA	= [0] * self.exp_pool_size
		self.experience_poolB	= [0] * self.exp_pool_size

		if self.loading:
			self.load_models()
	
	
	def run_training(	self,
						verbose=False):
	
		#	Sliding window memory update 
		#	Instead of copying a new memory_pool list 
		#	upon overflow, simply replace the next window each time 
		self.tstart 			= time.time()
		self.exp_i_A 					= 0
		self.exp_i_B 					= 0
		self.episode_num				= 0 

		#	Train 
		i = 0
		while i < self.iters:

			iter_t0					= time.time()
					
			#	UPDATE EPSILON
			self.update_epsilon(i/self.iters)	
		
			#	RUN ENVIRONMENT
			self.environment.init_environment(self.target_modelA,self.target_modelB,**self.env_kwargs)
			expA, expB, metrics = self.environment.run_environment(**self.run_kwargs)

			#	UPDATE MEMORY POOL A
			for exp in expA:
				self.experience_poolA[self.exp_i_A%self.exp_pool_size] = exp 
				self.exp_i_A += 1	

			#	UPDATE MEMORY POOL B
			for exp in expB:
				self.experience_poolB[self.exp_i_B%self.exp_pool_size] = exp 
				self.exp_i_B += 1		

			#	UPDATE VERBOSE 
			if verbose:
				print(f"[Episode {str(i).rjust(15)}/{int(self.iters)} -  {(100*i/self.iters):.2f}% complete  \tavg game len: {(sum(metrics['g_lens'])/len(metrics['g_lens'])):.2f}\t{(time.time()-self.tstart):.2f}s")
						
			# 	GET TRAINING SAMPLES
			#	AND TRAIN MODEL 
			if self.exp_i_A > self.sample_size:
				training_setA 			= self.get_samples("A")
				self.train_on_experiencesA(training_setA,verbose=self.verbose)
			if self.exp_i_B > self.sample_size:
				training_setB 			= self.get_samples("B")
				self.train_on_experiencesB(training_setB,verbose=self.verbose)
			print(f"\n\n")
			if self.exp_i_A < self.sample_size and self.exp_i_B < self.sample_size:
				i = 0
			else:
				self.episode_num += self.train_freq

						
			#	UPDATE MODELS 
			if i % self.update_freq == 0 and i > 0:
				self.transfer_models(verbose=verbose)
			
			i += 1
		return


	def get_samples(self,model,mode="random"):

		if mode == "random":
			if model == "A":
				return random.sample(self.experience_poolA[:self.exp_i_A],self.sample_size)
			if model == "B":
				return random.sample(self.experience_poolB[:self.exp_i_B],self.sample_size)
		
		if mode == "weighted":
			if model == "A":
				return random.sample(self.experience_poolA[:self.exp_i_A],self.sample_size)
			if model == "B":
				return random.sample(self.experience_poolB[:self.exp_i_B],self.sample_size)


	def cleanup(self):
		blocked_scores		= reduce_arr(self.all_scores,self.x_scale)
		blocked_lived 		= reduce_arr(self.all_lived,self.x_scale)
		graph_name = f"{self.name}_[{str(self.loss_fn).split('.')[-1][:-2]},{str(self.optimizer_fn).split('.')[-1][:-2]}@{self.kwargs['lr']}]]]"

		if self.save_fig:
			plot_game(blocked_scores,blocked_lived,graph_name)

		if self.gui:
			self.output.insert(tk.END,f"Completed Training\n\tHighScore:{self.best_score}\n\tSteps:{sum(self.all_lived[-1000:])/1000}")
		return blocked_scores,blocked_lived,self.best_score,graph_name


	def train_on_experiences(self,sample_setA,sample_setB,epochs=1,verbose=False):
		
		#Telemetry 
		if verbose:
			print(f"TRAINING A:")
			print(f"\tDataset:\n\t\t{'size'.ljust(12)}: {len(sample_setA)}\n\t\t{'batch_size'.ljust(12)}: {self.batch_size}\n\t\t{'epochs'.ljust(12)}: {epochs}\n\t\t{'lr'.ljust(12)}: {self.learning_modelA.optimizer.param_groups[0]['lr']:.6f}\n")

		#	Telemetry Vars 
		t0 									= time.time()
		t_gpu 								= 0
		num_equals 							= 40 
		printed 							= 0
		total_lossA							= 0
		total_lossB							= 0 

		#	Telemetry
		if verbose:
			print(f"\tPROGRESS- [",end='')

		#	Do one calc for all runs 
		num_batches = int(len(sample_setA) / self.batch_size)

		# 	Iterate through batches
		for batch_i in range(num_batches):

			i_start 							= batch_i*self.batch_size
			i_end   							= i_start + self.batch_size
			
			#	Telemetry
			percent 							= batch_i / num_batches
			if verbose:
				while (printed / num_equals) < percent:
					print("=",end='',flush=True)
					printed+=1
			
			#BELLMAN UPDATE 
			self.learning_modelA.optimizer.zero_grad()


			#Gather batch experiences
			batch_set 							= sample_setA[i_start:i_end]

			init_states 						= torch.stack([exp['s']  for exp in batch_set]).type(torch.float).to(self.device)
			action 								= [exp['a'] for exp in batch_set]
			next_states							= torch.stack([exp['s`'] for exp in batch_set]).type(torch.float).to(self.device)
			rewards 							= [exp['r']  for exp in batch_set]
			done								= [exp['done'] for exp in batch_set]

			#Calc final targets 
			initial_target_predictions 			= self.learning_modelA.forward(init_states)
			final_target_values 				= initial_target_predictions.clone().detach()
			
			#Get max from s`
			with torch.no_grad():
				stepped_target_predictions 			= self.target_modelA.forward(next_states)
				best_predictions 					= torch.max(stepped_target_predictions,dim=1)[0]

			#Update init values 
			for i,val in enumerate(best_predictions):
				chosen_action						= action[i]
				final_target_values[i,chosen_action]= rewards[i] + (done[i] * self.gamma * val)

			#	Calculate Loss
			t1 									= time.time()
			batch_lossA 						= self.learning_modelA.loss(initial_target_predictions,final_target_values)
			total_lossA 						+= batch_lossA.item()

			#Back Propogate
			batch_lossA.backward()
			self.learning_modelA.optimizer.step()
			t_gpu 								+= time.time() - t1
		# 	Iterate through batches
		#Telemetry 
		if verbose:
			print(f"]\ttime: {(time.time()-t0):.2f}s\tt_gpu:{(t_gpu):.2f}\tloss: {(total_lossA/num_batches):.6f}")

		#	Telemetry Vars 
		t0 									= time.time()
		t_gpu 								= 0
		num_equals 							= 40 
		printed 							= 0

		#	Telemetry
		if verbose:
			print(f"\tPROGRESS- [",end='')

		#	Do one calc for all runs 
		num_batches = int(len(sample_setB) / self.batch_size)
		for batch_i in range(num_batches):

			i_start 							= batch_i*self.batch_size
			i_end   							= i_start + self.batch_size
			
			#	Telemetry
			percent 							= batch_i / num_batches
			if verbose:
				while (printed / num_equals) < percent:
					print("=",end='',flush=True)
					printed+=1
			
			#BELLMAN UPDATE 
			self.learning_modelB.optimizer.zero_grad()

			#Gather batch experiences
			batch_set 							= sample_setB[i_start:i_end]

			init_states 						= torch.stack([exp['s']  for exp in batch_set]).type(torch.float).to(self.device)
			action 								= [exp['a'] for exp in batch_set]
			next_states							= torch.stack([exp['s`'] for exp in batch_set]).type(torch.float).to(self.device)
			rewards 							= [exp['r']  for exp in batch_set]
			done								= [exp['done'] for exp in batch_set]
			
			#Calc final targets 
			initial_target_predictions 			= self.learning_modelB.forward(init_states)
			final_target_values 				= initial_target_predictions.clone().detach()
			
			#Get max from s`
			with torch.no_grad():
				stepped_target_predictions 			= self.target_modelB.forward(next_states)
				best_predictions 					= torch.max(stepped_target_predictions,dim=1)[0]

			#Update init values 
			for i,val in enumerate(best_predictions):
				chosen_action						= action[i]
				final_target_values[i,chosen_action]= rewards[i] + (done[i] * self.gamma * val)

			#	Calculate Loss
			t1 									= time.time()
			batch_lossB 						= self.learning_modelB.loss(initial_target_predictions,final_target_values)
			total_lossB 						+= batch_lossB.item()

			#Back Propogate
			batch_lossB.backward()
			self.learning_modelB.optimizer.step()
			t_gpu 								+= time.time() - t1
		
		#	Telemetry
		if verbose:
			print(f"]\ttime: {(time.time()-t0):.2f}s\tt_gpu:{(t_gpu):.2f}\tloss: {(total_lossB/num_batches):.6f}")
			print("\n\n")

	def train_on_experiencesA(self,sample_setA,epochs=1,verbose=False):
		
		#Telemetry 
		if verbose:
			print(f"TRAINING A:")
			print(f"\tDataset:\n\t\t{'size'.ljust(12)}: {len(sample_setA)}\n\t\t{'batch_size'.ljust(12)}: {self.batch_size}\n\t\t{'epochs'.ljust(12)}: {epochs}\n\t\t{'lr'.ljust(12)}: {self.learning_modelA.optimizer.param_groups[0]['lr']:.6f}\n")

		#	Telemetry Vars 
		t0 									= time.time()
		t_gpu 								= 0
		num_equals 							= 40 
		printed 							= 0
		total_lossA							= 0
		total_lossB							= 0 

		#	Telemetry
		if verbose:
			print(f"\tPROGRESS- [",end='')

		#	Do one calc for all runs 
		num_batches = int(len(sample_setA) / self.batch_size)

		# 	Iterate through batches
		for batch_i in range(num_batches):

			i_start 							= batch_i*self.batch_size
			i_end   							= i_start + self.batch_size
			
			#	Telemetry
			percent 							= batch_i / num_batches
			if verbose:
				while (printed / num_equals) < percent:
					print("=",end='',flush=True)
					printed+=1
			
			#BELLMAN UPDATE 
			self.learning_modelA.optimizer.zero_grad()


			#Gather batch experiences
			batch_set 							= sample_setA[i_start:i_end]

			init_states 						= torch.stack([exp['s']  for exp in batch_set]).type(torch.float).to(self.device)
			action 								= [exp['a'] for exp in batch_set]
			next_states							= torch.stack([exp['s`'] for exp in batch_set]).type(torch.float).to(self.device)
			rewards 							= [exp['r']  for exp in batch_set]
			done								= [exp['done'] for exp in batch_set]
			#print(f"action taken was {action[-1]}")
			
			#Calc final targets 
			initial_target_predictions 			= self.learning_modelA.forward(init_states)
			final_target_values 				= initial_target_predictions.clone().detach()
			#print(f"init value was {final_target_values[-1][action[-1]]}")
			
			#Get max from s`
			with torch.no_grad():
				stepped_target_predictions 			= self.target_modelA.forward(next_states)
				best_predictions 					= torch.max(stepped_target_predictions,dim=1)[0]

			#Update init values 
			for i,val in enumerate(best_predictions):
				chosen_action						= action[i]
				final_target_values[i,chosen_action]= rewards[i] + (done[i] * self.gamma * val)
			
			#input(f"updated value was {final_target_values[-1][action[-1]]}")
			#	Calculate Loss
			t1 									= time.time()
			batch_lossA 						= self.learning_modelA.loss(initial_target_predictions,final_target_values)
			total_lossA 						+= batch_lossA.item()

			#Back Propogate
			batch_lossA.backward()
			self.learning_modelA.optimizer.step()
			t_gpu 								+= time.time() - t1
		# 	Iterate through batches
		#Telemetry 
		if verbose:
			print(f"]\ttime: {(time.time()-t0):.2f}s\tt_gpu:{(t_gpu):.2f}\tloss: {(total_lossA/num_batches):.6f}")

	def train_on_experiencesB(self,sample_setB,epochs=1,verbose=False):
		
		#	Telemetry Vars 
		t0 									= time.time()
		t_gpu 								= 0
		num_equals 							= 40 
		printed 							= 0
		total_lossB							= 0
		#	Telemetry
		if verbose:
			print(f"\tPROGRESS- [",end='')

		#	Do one calc for all runs 
		num_batches = int(len(sample_setB) / self.batch_size)
		for batch_i in range(num_batches):

			i_start 							= batch_i*self.batch_size
			i_end   							= i_start + self.batch_size
			
			#	Telemetry
			percent 							= batch_i / num_batches
			if verbose:
				while (printed / num_equals) < percent:
					print("=",end='',flush=True)
					printed+=1
			
			#BELLMAN UPDATE 
			self.learning_modelB.optimizer.zero_grad()

			#Gather batch experiences
			batch_set 							= sample_setB[i_start:i_end]

			init_states 						= torch.stack([exp['s']  for exp in batch_set]).type(torch.float).to(self.device)
			action 								= [exp['a'] for exp in batch_set]
			next_states							= torch.stack([exp['s`'] for exp in batch_set]).type(torch.float).to(self.device)
			rewards 							= [exp['r']  for exp in batch_set]
			done								= [exp['done'] for exp in batch_set]
			
			#Calc final targets 
			initial_target_predictions 			= self.learning_modelB.forward(init_states)
			final_target_values 				= initial_target_predictions.clone().detach()
			
			#Get max from s`
			with torch.no_grad():
				stepped_target_predictions 			= self.target_modelB.forward(next_states)
				best_predictions 					= torch.max(stepped_target_predictions,dim=1)[0]

			#Update init values 
			for i,val in enumerate(best_predictions):
				chosen_action						= action[i]
				final_target_values[i,chosen_action]= rewards[i] + (done[i] * self.gamma * val)

			#	Calculate Loss
			t1 									= time.time()
			batch_lossB 						= self.learning_modelB.loss(initial_target_predictions,final_target_values)
			total_lossB 						+= batch_lossB.item()

			#Back Propogate
			batch_lossB.backward()
			self.learning_modelB.optimizer.step()
			t_gpu 								+= time.time() - t1
		
		#	Telemetry
		if verbose:
			print(f"]\ttime: {(time.time()-t0):.2f}s\tt_gpu:{(t_gpu):.2f}\tloss: {(total_lossB/num_batches):.6f}")
			print("\n\n")


	def transfer_models(self,verbose=False):
		if verbose:
			print("\ntransferring models\n\n")

		#Check for dir 
		if not os.path.isdir(self.path):
			os.mkdir(self.path)

		torch.save(self.learning_modelA.state_dict(),os.path.join(self.path,f"A_lm_state_dict"))
		torch.save(self.learning_modelB.state_dict(),os.path.join(self.path,f"B_lm_state_dict"))

		#Load the learning model as the target model
		self.target_modelA = self.model_fn(**self.model_kwargs)
		self.target_modelB = self.model_fn(**self.model_kwargs)

		self.target_modelA.load_state_dict(torch.load(os.path.join(self.path,f"A_lm_state_dict")))

		self.target_modelB.load_state_dict(torch.load(os.path.join(self.path,f"B_lm_state_dict")))


	@staticmethod
	def update_epsilon(percent):
		radical = -.4299573*100*percent -1.2116290 
		if percent > .50:
			return 0
		else:
			return pow(2.7182,radical)


	def load_models(self,path="models"):
		#Load the learning model as the target model
		self.target_modelA = self.model_fn(**self.model_kwargs)
		self.target_modelB = self.model_fn(**self.model_kwargs)

		self.target_modelA.load_state_dict(torch.load(os.path.join(path,f"A_lm_state_dict")))

		self.target_modelB.load_state_dict(torch.load(os.path.join(path,f"B_lm_state_dict")))


class Environment:

	def __init__(self,name,device=torch.device('cpu')):

		self.name               = name
		self.running            = True 
		self.experiences        = [] 
		self.device 			= device

	
	#   Will setup the environment for prep to train 
	#   May place starting pieces, init movements, etc... 
	def init_environment(self):
		raise NotImplementedError(f"'init_environment' method not implemented")

	#   Stepping environment will cause one discrete action in the environment 
	#   And automatically add experiences to the pool 
	def run_environment(self):
		raise NotImplementedError(f"'run_environment' method not implemented")
	

class Chess(Environment):
	try:
		chess_moves 		= json.loads(open(os.path.join("steinpy","ml","res","chessmoves.txt"),"r").read())
	except FileNotFoundError:
		chess_moves 		= json.loads(open(os.path.join("/home/steinshark/code","steinpy","ml","res","chessmoves.txt"),"r").read())
	
	move_to_index		= {chess.Move.from_uci(uci):i for i,uci in enumerate(chess_moves)}
	index_to_move	    = {v : k  for k,v in move_to_index.items()}
	base_tensor 		= torch.zeros(size=(2,130,130),device=torch.device('cpu'),dtype=torch.half,requires_grad=False)
	created_base 		= False
	piece_tensors 		= {} 

	lookup_table 		= dict()
	prob 				= None
	#Build 200 random noise vectors on the GPU
	noise				= numpy.random.default_rng()

	def __init__(self,name,device=torch.device('cpu')):
		super(Chess,self).__init__(name,device=device)

		self.times = {"exploit_stack":0,
					  "exploit_model":0,
					  "exploit_picks":0,
					  "run":0}


	def init_environment(	self,
			  				self_model,
							adversary_model,
							simul_games=10,
							rewards={"capture":.1,"captured":-.1,"win":1,"lose":-1,"draw":0},
							img_dims=(480,270),
							epsilon=.1,
							img_size=130):
		

		self.experiences		= []
		self.rewards 			= rewards

		self.board				= chess.Board()
		self.img_repr			= None

		self.img_w 				= img_dims[0]
		self.img_h 				= img_dims[1]

		self.epsilon			= epsilon


		self.model 				= model.to(self.device)

		self.chess_moves 		= json.loads(open(os.path.join("C:/","gitrepos","steinpy","ml","res","chessmoves.txt"),"r").read())
		self.next_move			= None 

		self.piece_tensors 		= {}
		self.img_size			= img_size

		self.move_num		 	= 0 
		self.capture_multiplier = 14
		self.dummy_tensor		= torch.empty(size=(2,130,130))

		self.self_model 		= self_model
		self.adversary_model	= adversary_model




	#PROPER run_environment 
	def run_environment_OLD(self,kwargs=None):
		t0 = time.time()
		while len(self.active_boards) > 0:
			
			
			if self.move_num % 10 == 0 and False:
				print(f"playing move {self.move_num}")
			if self.move_num > 100 and False:
				break
			
			# input(f"starting move: {self.move_num}\nplaying: {len(self.active_boards)}\nlen boards: {len(self.boards)}\nlen moves: {len(self.next_moves)}")
			markdel 				= [] 
			self.next_exp_set 		= [{"s":self.dummy_tensor,"a":None,"r":0,"s`":None,"done":1} for _ in range(self.simul_games)] 
			
			#Start experiences 
			for board_i in self.active_boards:
				#Prep experience
				to_play 										= "w" if self.boards[board_i].turn == chess.WHITE else "b"
				played 											= {"w":"b","b":"w"}[to_play]

				if len(self.experiences[played][board_i]) > 1:
					self.next_exp_set[board_i]['s']			= self.experiences[played][board_i][-1]['s`'].clone()
				else:
					self.next_exp_set[board_i]['s']			= self.create_board_img(board_i)
			
			#Decide Moves in batch
			if random.random() < self.epsilon:
				self.explore()
			else:
				self.exploit() 
			
			for board_i in self.active_boards:
				# print(self.boards[board_i])
				# print(f"playing move {self.next_moves[board_i].uci()}")
				if self.boards[board_i].is_capture(self.next_moves[board_i]):
					captured_piece 					= self.boards[board_i].piece_at(self.next_moves[board_i].to_square)
					val 							= 0 
					if captured_piece == chess.QUEEN:
						val = 9 
					if captured_piece == chess.ROOK:
						val = 5 
					if captured_piece == chess.BISHOP:
						val = 3.5 
					if captured_piece == chess.KNIGHT:
						val = 3 
					if captured_piece == chess.PAWN:
						val = 1 
					
					self.next_exp_set[board_i]['r']				= val / self.capture_multiplier 
					self.experiences[played][board_i][-1]['r'] += val / self.capture_multiplier 

				#Make move 
				self.boards[board_i].push_san(self.next_moves[board_i].uci())

				#Check outcomes
				if self.boards[board_i].is_checkmate():
					self.next_exp_set[board_i]['r']					= self.rewards['win']
					self.experiences[played][board_i][-1]['r']		= self.rewards['lose']
					self.next_exp_set[board_i]['done']				= 0
					self.experiences[played][board_i][-1]['done']	= 0 
					markdel.append(board_i)

				elif self.boards[board_i].is_fifty_moves():
					self.next_exp_set[board_i]['r']					= self.rewards['draw']
					self.experiences[played][board_i][-1]['r']		= self.rewards['draw']
					self.next_exp_set[board_i]['done']				= 0
					self.experiences[played][board_i][-1]['done']	= 0 
					markdel.append(board_i)

				elif self.boards[board_i].is_stalemate():
					self.next_exp_set[board_i]['r']					= self.rewards['draw']
					self.experiences[played][board_i][-1]['r']		= self.rewards['draw']
					self.next_exp_set[board_i]['done']				= 0
					self.experiences[played][board_i][-1]['done']	= 0 
					markdel.append(board_i)

				elif self.boards[board_i].is_insufficient_material():
					self.next_exp_set[board_i]['r']					= self.rewards['draw']
					self.experiences[played][board_i][-1]['r']		= self.rewards['draw']
					self.next_exp_set[board_i]['done']				= 0
					self.experiences[played][board_i][-1]['done']	= 0 
					markdel.append(board_i)
				
				elif self.boards[board_i].is_repetition():
					self.next_exp_set[board_i]['r']					= self.rewards['draw']
					self.experiences[played][board_i][-1]['r']		= self.rewards['draw']
					self.next_exp_set[board_i]['done']				= 0
					self.experiences[played][board_i][-1]['done']	= 0 
					markdel.append(board_i)

				#Get next 
				self.next_exp_set[board_i]['s`']			= self.create_board_img(board_i)

				#Add experience 
				self.experiences[to_play][board_i].append(self.next_exp_set[board_i])		

				# print(f"played success\n\n")
			for board_j in markdel:
				# print(f"Game {board_j} ended with W: {self.experiences['w'][board_j][-1]['r']} vs. B: {self.experiences['b'][board_j][-1]['r']}\t- {self.move_num} ply: {(time.time()-t0)/self.move_num:.3f}s")
				self.active_boards.remove(board_j)

			self.move_num += 1
		
		w_exps				= [] 
		for explist in self.experiences['w']:
			w_exps += explist
		b_exps				= [] 
		for explist in self.experiences['b']:
			b_exps += explist

		self.times['run'] += time.time() - t0

		self.times['exploit_stack'] = 5*self.times['exploit_stack'] / self.move_num
		self.times['exploit_model'] = 5*self.times['exploit_model'] / self.move_num
		self.times['exploit_picks'] = 5*self.times['exploit_picks'] / self.move_num

		return w_exps,b_exps


	#SWITCH WHEN DONE 
	def run_environment(self,kwargs=None,unimodel=True):
		t0 = time.time()
		while len(self.active_boards) > 0:
			
			
			if self.move_num % 10 == 0 and False:
				print(f"playing move {self.move_num}")
			if self.move_num > 100 and False:
				break
			
			# input(f"starting move: {self.move_num}\nplaying: {len(self.active_boards)}\nlen boards: {len(self.boards)}\nlen moves: {len(self.next_moves)}")
			markdel 				= [] 
			self.next_exp_set 		= [{"s":self.dummy_tensor,"a":None,"r":0,"s`":self.dummy_tensor,"done":1} for _ in range(self.simul_games)] 
			
			#Start experiences 
			for board_i in self.active_boards:
				#Prep experience
				cur_player 								= 1 if self.boards[board_i].turn == chess.WHITE else -1
				next_player 							= cur_player * -1

				if len(self.experiences[next_player][board_i]) > 1:
					self.next_exp_set[board_i]['s']			= self.experiences[next_player][board_i][-1]['s`'].clone()
					self.next_exp_set[board_i]['s'][1] 		*= -1														#Change player
					
				else:
					self.next_exp_set[board_i]['s'][0]		= self.create_board_img(board_i)
					self.next_exp_set[board_i]['s'][1]		= torch.ones(size=(130,130)) * cur_player
			
			#input(f"ch2 of exp is {self.next_exp_set[board_i]['s'][1][0][0]}")
			

			#Decide Moves
			if random.random() < self.epsilon:
				self.explore_san()
			else:
				self.exploit_legal() 
			

			#Play with move for each board 
			for board_i in self.active_boards:
				move 												= self.next_moves[board_i]
				self.next_exp_set[board_i]['a']						= self.chess_moves.index(move)

				#Check illegal move
				if not move in [m.uci() for m in self.boards[board_i].legal_moves]:
					self.next_exp_set[board_i]['r']					= self.rewards['lose']
					self.next_exp_set[board_i]['done']				= 0
					markdel.append(board_i)
					self.next_exp_set[board_i]['s`'][0]				= self.create_board_img(board_i)
					self.next_exp_set[board_i]['s`'][1]				= torch.ones(size=(130,130)) * next_player
					self.experiences[cur_player][board_i].append(self.next_exp_set[board_i])	
					continue
				#Else Play legal move 
				else:
					self.next_moves[board_i] = self.boards[board_i].parse_san(move)

				#Check for move rewards
				if self.boards[board_i].is_capture(self.next_moves[board_i]):
					captured_piece 					= self.boards[board_i].piece_at(self.next_moves[board_i].to_square)
					val 							= 0 
					if captured_piece == chess.QUEEN:
						val = 9 
					if captured_piece == chess.ROOK:
						val = 5 
					if captured_piece == chess.BISHOP:
						val = 3.5 
					if captured_piece == chess.KNIGHT:
						val = 3 
					if captured_piece == chess.PAWN:
						val = 1 
					
					self.next_exp_set[board_i]['r']				= val / self.capture_multiplier 
					self.experiences[next_player][board_i][-1]['r'] += val / self.capture_multiplier 

				#Make move 
				self.boards[board_i].push_san(self.next_moves[board_i].uci())

				#Check outcomes
				if self.boards[board_i].is_checkmate():
					self.next_exp_set[board_i]['r']						= self.rewards['win']
					self.experiences[next_player][board_i][-1]['r']		= self.rewards['lose']
					self.next_exp_set[board_i]['done']					= 0
					self.experiences[next_player][board_i][-1]['done']	= 0 
					markdel.append(board_i)

				elif self.boards[board_i].is_fifty_moves():
					self.next_exp_set[board_i]['r']						= self.rewards['draw']
					self.experiences[next_player][board_i][-1]['r']		= self.rewards['draw']
					self.next_exp_set[board_i]['done']					= 0
					self.experiences[next_player][board_i][-1]['done']	= 0 
					markdel.append(board_i)

				elif self.boards[board_i].is_stalemate():
					self.next_exp_set[board_i]['r']						= self.rewards['draw']
					self.experiences[next_player][board_i][-1]['r']		= self.rewards['draw']
					self.next_exp_set[board_i]['done']					= 0
					self.experiences[next_player][board_i][-1]['done']	= 0 
					markdel.append(board_i)

				elif self.boards[board_i].is_insufficient_material():
					self.next_exp_set[board_i]['r']						= self.rewards['draw']
					self.experiences[next_player][board_i][-1]['r']		= self.rewards['draw']
					self.next_exp_set[board_i]['done']					= 0
					self.experiences[next_player][board_i][-1]['done']	= 0 
					markdel.append(board_i)
				
				elif self.boards[board_i].is_repetition():
					self.next_exp_set[board_i]['r']						= self.rewards['draw']
					self.experiences[next_player][board_i][-1]['r']		= self.rewards['draw']
					self.next_exp_set[board_i]['done']					= 0
					self.experiences[next_player][board_i][-1]['done']	= 0 
					markdel.append(board_i)

				#Get next 
				self.next_exp_set[board_i]['s`'][0]			= self.create_board_img(board_i)
				self.next_exp_set[board_i]['s`'][1]			= torch.ones(size=(self.img_size,self.img_size)) * next_player

				#Add experience 
				self.experiences[cur_player][board_i].append(self.next_exp_set[board_i])		

				# print(f"played success\n\n")
			for board_j in markdel:
				# print(f"Game {board_j} ended with W: {self.experiences['w'][board_j][-1]['r']} vs. B: {self.experiences['b'][board_j][-1]['r']}\t- {self.move_num} ply: {(time.time()-t0)/self.move_num:.3f}s")
				self.active_boards.remove(board_j)

			self.move_num += 1
		
		exps				= [] 

		g_lens = [0 for _ in range(self.simul_games)]

		for board_i,explist in enumerate(self.experiences[1]):
			g_lens[board_i] += len(explist)
			exps += explist

		for board_i,explist in enumerate(self.experiences[-1]):
			g_lens[board_i] += len(explist)
			exps += explist

		self.times['run'] += time.time() - t0

		self.times['exploit_stack'] = 5*self.times['exploit_stack'] / self.move_num
		self.times['exploit_model'] = 5*self.times['exploit_model'] / self.move_num
		self.times['exploit_picks'] = 5*self.times['exploit_picks'] / self.move_num

		return exps, {"g_lens":g_lens}



	def explore(self):
		#Find which moves were legal
		for i, board_i in enumerate(self.active_boards):
			self.next_moves[board_i]	= random.sample(list(self.boards[board_i].legal_moves),1)[0]




	def exploit(self):
		t0 = time.time()


		t1 = time.time()
		with torch.no_grad():
			next_moves 				= self.model.forward(this_batch)
			indices			= torch.sort(next_moves,descending=True,dim=1)[1].cpu().numpy()
		
		t2 = time.time()

		#Find which moves were legal
		for i, board_i in enumerate(self.active_boards):
			
			current_board			= self.boards[board_i]
			found_move 				= False 
			
			legal_moves 	= set(m.uci() for m in list(current_board.legal_moves))

			for move_i in indices[i]:
				potential_move 	= self.chess_moves[move_i]

				if potential_move in legal_moves:
					self.next_moves[board_i] = current_board.parse_san(potential_move)
					found_move = True 
					break

			if not found_move:
				input("never found a move ") 
			

		self.times['exploit_stack'] += t1 - t0
		self.times['exploit_model'] += t2 - t1
		self.times['exploit_picks'] += time.time() - t2
		return 


	@staticmethod
	def fen_to_tensor(board,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),is_fen=False,no_castle=False):

		#Encoding will be an 8x8 x n tensor 
		#	6 for white, 6 for black 
		#	OPTIONALLY 4 for castling 7+7+4 
		# 	1 for move 
		board_tensor 	= numpy.zeros(shape=(13 if no_castle else 17,8,8))
		piece_indx 		= {"R":0,"N":1,"B":2,"Q":3,"K":4,"P":5,"r":6,"n":7,"b":8,"q":9,"k":10,"p":11}
		#Go through FEN and fill pieces

		fen 	= board.fen() if not is_fen else board
		
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
		
		#Place turn 
		slice 	= 12 
		board_tensor[slice,:,:]   = numpy.ones(shape=(8,8)) * 1 if turn == "w" else -1

		#Place all castling allows if not no_castle
		for castle in [] if no_castle else ["K","Q","k","q"] :
			slice += 1
			board_tensor[slice,:,:]	= numpy.ones(shape=(8,8)) * 1 if castle in castling else 0

		return torch.tensor(board_tensor,dtype=torch.float,device=device,requires_grad=False)

	@staticmethod
	def fen_to_numpy_np(board):

		#Encoding will be a 19x8x8  tensor 
		#	7 for whilte, 7 for black 
		#	4 for castling 7+7+4 
		# 	1 for move 
		#t0 = time.time()
		#board_tensor 	= torch.zeros(size=(1,19,8,8),device=device,dtype=torch.float,requires_grad=False)
		board_tensor 	= numpy.zeros(shape=(17,8,8),dtype=numpy.float32)
		piece_indx 	= {"R":0,"N":1,"B":2,"Q":3,"K":4,"P":5,"r":6,"n":7,"b":8,"q":9,"k":10,"p":11}
		#Go through FEN and fill pieces
		fen 	= board.fen()
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

		return board_tensor



	@staticmethod
	def coord_to_xy(rank,file):
		
		y_start 	= 5 + (8 - rank)*15
		x_start 	= 5 + (file-1)*15

		return y_start,x_start







#MCTS Pseudo code 

# 	start with root node 
#	if children:
#		recurse down following max(move_score)
#	
# 	once: end_pos or unexplored:
#		end_pos: update curnode with v,n and backup with -1 * v each node  
#		unexplored: expand node's legal moves into nodes IAW P and return value v from network 
class Node:


	def __init__(self,board,p=.5,parent=None,c=1,move=None):

		self.board 			= board 
		self.parent 		= parent 
		self.parents 		= [parent]
		self.children		= {}
		self.num_visited	= 0
		self.move 			= None 

		self.Q_val 			= 0 
		self.p				= p 
		self.c 				= c 

		self.score 			= 0

		self.fen 			= "".join(board.fen().split(" ")[:2])
	
	def get_score(self):
		return self.Q_val + ((self.c * self.p) * (sqrt(sum([m.num_visited for m in self.parent.children.values()])) / (1 + self.num_visited)))
	
	def bubble_up(self,v):

		#Update this node
		self.Q_val 			= (self.num_visited * self.Q_val + v) / (self.num_visited + 1) 
		self.num_visited 	+= 1

		for parent in self.parents:	
			#Recursively update all parents 
			if not parent is None:
				parent.bubble_up(-1*v)


class Tree:

	
	def __init__(self,board,model,base_node=None,draw_thresh=250,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
		self.board 			= board 
		self.model 			= model 
		self.draw_thresh	= draw_thresh
		self.device	 		= device
		if base_node: 
			self.root 			= base_node
			self.root.parent 	= None
		else:
			self.root 			= Node(board,0,None)
			self.root.parent	= None 

		self.parents 		= {self.root:{None}}

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
	
	def update_tree_nonrecursive_exp(self,x=.9,dirichlet_a=.3,rollout_p=.25,iters=300,abbrev=True): 
		
		#DEFINE FUNCTIONS IN LOCAL SCOPE 
		create_repr				= Chess.fen_to_tensor
		noise_gen				= numpy.random.default_rng().dirichlet
		softmax_fn				= softmax
		move_to_index 			= Chess.move_to_index
		index_to_move 			= Chess.index_to_move


		t_test 					=  0
		try:
			chess_moves 			= json.loads(open(os.path.join("steinpy","ml","res","chessmoves.txt"),"r").read())
		except FileNotFoundError:
			chess_moves 			= json.loads(open(os.path.join("/home/steinshark/code","steinpy","ml","res","chessmoves.txt"),"r").read())
		
		self.root.parent		= None 
		flag					= False
		debugging 				= False
		overflag				= False 
		for iter_i in range(iters):
			overflag = False
			node = self.root
			score_mult = 1 if node.board.turn == chess.WHITE else -1
			#print(f"turn was {node.board.turn} -> mult was {score_mult}")

			#Find best leaf node 
			node, score_mult = self.get_best_node_max(node,score_mult)
			#print(f"mult now {score_mult}")

			#Check if game over

			game_over 	= node.board.is_checkmate() or node.board.is_stalemate() or node.board.is_seventyfive_moves()
			if game_over:
				overflag 	= True 
				res 		= node.board.result()
				if "1" in res and not "1/2" in res:
					if res[0] == "1":
						v 	=   1 * score_mult
					else:
						v 	=  -1 * score_mult
				else:
					v 	=  0 
				if flag:
					print(f"result was {node.board.result()}")
					print(f"found result of {v} in position\n{node.board}\nafter {'white' if not node.board.turn else 'black'} moved")
					
			
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
				node.repr 				= create_repr(node.board,self.device,no_castle=False)
				
				#	
				with torch.no_grad():
					model_in 					= torch.stack([node.repr])
					prob,v 						= self.model(model_in)
				prob_cpu					= prob[0].to(torch.device('cpu'),non_blocking=True).numpy()
				legal_moves 				= [move_to_index[m] for m in node.board.legal_moves]	#Get move numbers
				legal_probs 				= [prob_cpu[i] for i in legal_moves]

				noise 						= noise_gen([dirichlet_a for _ in range(len(legal_probs))],1)
				#input(f"noise yields: {noise}")
				legal_probs					= softmax_fn([x*p for p in legal_probs] + (1-x)*noise)
				#input(f"model yields probs: {legal_probs}")
				#self.lookup_table[position_fen]	=	 (v,legal_moves,legal_probs)

				if debugging and abs(v) > .5:
					input(f"\n{node.board}\nposition evaluates to {v}")
				node.children 		= {move_i : Node(node.board.copy(stack=False) ,p=p,parent=node) for p,move_i in zip(legal_probs,legal_moves)} 


				for move in node.children:
					node.children[move].board.push(index_to_move[move])
					node.children[move].move 	= index_to_move[move]

					#Add to parents dict 
					if not node.children[move].fen in self.parents:
						self.parents[node.children[move].fen] = {node}
					else:
						node.children[move].parents += list(self.parents[node.children[move].fen])
						self.parents[node.children[move].fen].add(node)

						

				
				#Remove rollout for now
				if False and random.random() < rollout_p:
					v = self.rollout_exp(random.choice(list(node.children.values())).board.copy()) * score_mult
			
			if overflag and False:
				print(f"updating {node.move} with val {v} was eq {node.get_score():.3f}",end='')

			if isinstance(v,torch.Tensor):
				v = v.item()

			node.bubble_up(v)				
			# while node.parent:
			# 	if isinstance(v,torch.Tensor):
			# 		v = v.item()
			
			if overflag and False:
				print(f" now is {node.get_score():.3f} - Q = {node.Q_val:.3f} U = {node.get_score()-node.Q_val:.3f}")

		#input(f"\nnew policy:\n{[ (Chess.chess_moves[k],(int(1000*v.Q_val)/1000),int(100*(v.get_score()-v.Q_val))/100) for k,v in self.root.children.items()]}")
				
		
		return {move:self.root.children[move].num_visited for move in self.root.children}

	def get_best_node_max(self,node:Node,score_mult):
		score_mult *= -1
		while node.children:
				#drive down to leaf
				best_node 			= max(list(node.children.values()),key = lambda x: x.get_score())
				node 				= best_node
				score_mult			*= -1

		return node, score_mult
	
	def get_policy(self,search_iters,abbrev=True):
		
		return self.update_tree_nonrecursive_exp(iters=search_iters,abbrev=abbrev)

		# with torch.no_grad():
		# 	for _ in range(search_iters):
		# 		self.update_tree(self.root)
		# return {move:self.root.children[move].num_visited for move in self.root.children}


class Treert:

	
	def __init__(self,board,engine,context,stream,device_input,device_output,host_out,base_node=None,draw_thresh=250,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
		self.board 			= board 
		self.engine 		= engine 
		self.context 		= context 
		self.stream 		= stream 
		self.draw_thresh	= draw_thresh
		self.device	 		= device
		self.device_in		= device_input 
		self.device_out 	= device_output
		self.host_out 		= host_out
		if base_node: 
			self.root 			= base_node
			self.root.parent 	= None
		else:
			self.root 			= Node(board,0,None)
			self.root.parent	= None 

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
	
	def update_tree_nonrecursive_exp(self,x=.8,dirichlet_a=.3,rollout_p=.25,iters=300,abbrev=True): 
		
		#DEFINE FUNCTIONS IN LOCAL SCOPE 
		infer 					= self.model.forward
		create_repr				= Chess.fen_to_tensor_np
		noise_gen				= numpy.random.default_rng().dirichlet
		move_to_index 			= Chess.move_to_index
		index_to_move 			= Chess.index_to_move


		t_test 					=  0
		try:
			chess_moves 			= json.loads(open(os.path.join("steinpy","ml","res","chessmoves.txt"),"r").read())
		except FileNotFoundError:
			chess_moves 			= json.loads(open(os.path.join("/home/steinshark/code","steinpy","ml","res","chessmoves.txt"),"r").read())
		
		self.root.parent		= None 
		flag					= True
		debugging 				= False 
		for iter_i in range(iters):

			if iter_i * 10 == 0 and iter_i > 0:
				torch.cuda.empty_cache()
			node = self.root
			score_mult = 1 if node.board.turn == chess.WHITE else -1
			#print(f"turn was {node.board.turn} -> mult was {score_mult}")

			#Find best leaf node 
			node, score_mult = self.get_best_node_max(node,score_mult)
			#print(f"mult now {score_mult}")

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
				node.repr 				= create_repr(node.board,self.device)
				
				#	
				with torch.no_grad():
					model_in 					= node
					cuda_driver.memcpy_htod_async(self.device_in,node.repr,self.stream)
					self.context.execute_async(bindings=[int(self.device_in),int(self.device_out)],stream_handle=self.stream.handle)
					cuda_driver.memcpy_dtoh_async(self.host_out,self.device_out,self.stream)
					self.stream.synchronize()

					input(f"recieved output: {self.host_out}")

					prob,v 						= infer(model_in)
				prob_cpu					= prob[0].to(torch.device('cpu'),non_blocking=True).numpy()
				legal_moves 				= [move_to_index[m] for m in node.board.legal_moves]	#Get move numbers
				legal_probs 				= [prob_cpu[i] for i in legal_moves]

				noise 						= noise_gen([dirichlet_a for _ in range(len(legal_probs))],1)
				#input(f"noise yields: {noise}")
				legal_probs					= softmax_fn([x*p for p in legal_probs] + (1-x)*noise)[0]
				#input(f"model yields probs: {legal_probs}")
				#self.lookup_table[position_fen]	=	 (v,legal_moves,legal_probs)

				if debugging and abs(v) > .01:
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

	def get_best_node_max(self,node:Node,score_mult):
		score_mult *= -1
		while node.children:
				#drive down to leaf
				best_node 			= max(list(node.children.values()),key = lambda x: x.get_score())
				node 				= best_node
				score_mult			*= -1

		return node, score_mult
	
	def get_policy(self,search_iters,abbrev=True):
		
		return self.update_tree_nonrecursive_exp(iters=search_iters,abbrev=abbrev)

		# with torch.no_grad():
		# 	for _ in range(search_iters):
		# 		self.update_tree(self.root)
		# return {move:self.root.children[move].num_visited for move in self.root.children}


if __name__ == "__main__":
	board = chess.Board()
	board.push_san("e2e4")
	board.push_san("a7a6")
	board.push_san("f1c4")
	board.push_san("b7b6")


	for _ in range(100):
		command 	= f"test {board.fen().replace(' ', 'X')} 2"
		out = os.popen(command,shell=True).read()
	print(out)
	# model 	= ChessNet()
	
	# tree 		= Tree(board,model,draw_thresh=200)
	# moves 		= tree.get_policy(600)
	# print(f"in position\n{board}\nbuilt policy:")
	# print(f"chose moves\n{[(Chess.chess_moves[k],v) for k,v in moves.items()]}")

	# # next_move 		= random.choices(move_i,visits,k=1)[0]
	# # next_move		= Chess.chess_moves[next_move]
	# # print(f"chose to play {next_move}")



