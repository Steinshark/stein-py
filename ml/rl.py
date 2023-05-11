import torch 
import os 
import time 
import random 
import tkinter as tk
import chess
import chess.svg
import json 
from cairosvg import svg2png
from PIL import Image 
from io import BytesIO 
from torchvision.transforms import PILToTensor, ToPILImage
from math import sqrt 
import numpy 
import scipy 
from matplotlib import pyplot as plt 

os.environ['KMP_DUPLICATE_LIB_OK']='True'

inf_time = []
class QLearner:

	def __init__(	self,
		  		environment,
				model_fn,
				model_kwargs,
				loss_fn=torch.nn.MSELoss,
				optimizer_fn=torch.optim.Adam,
				optimizer_kwargs={"lr":1e-5},
				device=torch.device('cuda'),
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
				device=torch.device('cuda'),
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
	chess_moves 		= json.loads(open(os.path.join("C:/","gitrepos","steinpy","ml","res","chessmoves.txt"),"r").read())
	base_tensor 		= torch.zeros(size=(2,130,130),device=torch.device('cuda'),dtype=torch.half,requires_grad=False)
	created_base 		= False
	piece_tensors 		= {} 

	lookup_table 		= {}
	legal_move_table	= {}
	prob 				= None
	#Build 200 random noise vectors on the GPU
	noise				= numpy.random.default_rng()
	lookup 				= {} 
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



	def create_board_img(self,board_i):
		
		if self.move_num	% 2 == 1:
			turn 	= 1 
		else:
			turn 	= -1 

		#Build images from dataset if not available 
		if len(self.piece_tensors) == 0:
			for fname in os.listdir("C:/gitrepos/steinpy/ml/res"):
				if not "light" in fname and not "dark" in fname:
					continue
				self.piece_tensors[fname]	= torch.load(os.path.join("C:/gitrepos/steinpy/ml/res",fname))

		#Build base tensors for all games - will only have to update 1 piece 
		if len(self.state_imgs) == 0:
			png_bytes									= svg2png(chess.svg.board(chess.Board(),size=str(self.img_size)))	#Always 130
			img 										= Image.open(BytesIO(png_bytes)).convert("L")						#Convert to greyscale
			self.base_tensor							= PILToTensor()(img).type(torch.uint8)

			for _ in range(self.simul_games):
				self.state_imgs[chess.STARTING_FEN] 		= self.base_tensor.clone()

			return self.state_imgs[self.boards[board_i].fen()].clone().type(torch.float)


		#Check if we have this position already
		cur_fen 	= self.boards[board_i].fen()

		if cur_fen in self.state_imgs:
			return self.state_imgs[cur_fen].clone().type(torch.float)
		

		#Check if we have this boards last position

		if not len(self.boards[board_i].move_stack) == 0:
			prev_move   = self.boards[board_i].pop() 
			prev_fen 	= self.boards[board_i].fen() 
			self.boards[board_i].push_san(prev_move.uci())

			if prev_fen in self.state_imgs:

				base_tensor		= self.state_imgs[prev_fen]
				
				for rank_i, (cur_rank,prev_rank) in enumerate(zip(cur_fen.split(" ")[0].split("/"),prev_fen.split(" ")[0].split("/")),1):
					rank_i = 9 - rank_i
					if not cur_rank == prev_rank:
						cur_rank = cur_rank.replace("1","e").replace("2","ee").replace("3","eee").replace("4","eeee").replace("5","eeeee").replace("6","eeeeee").replace("7","eeeeeee").replace("8","eeeeeeee")
						#print(cur_rank)
						for file_x,square in enumerate(cur_rank,1):

							if ((rank_i % 2 == 0) and (file_x % 2 == 0)) or ((rank_i % 2 == 1) and (file_x % 2 == 1)):
								sq_color = "dark"
							else:
								sq_color = "light"

							if square in ["r","n","b","q","k","p"]:
								color 	= "b"
							elif square in ["R","N","B","Q","K","P"]:
								color 	= "w"

							elif square == 'e':
								color = "w"
								square 	= "e"
							
							tensor_key	= f"{color}_{square}_{sq_color}"
							tensor 		= self.piece_tensors[tensor_key].clone()

							y,x 	= self.coord_to_xy(rank_i,file_x)

							base_tensor[0,y:y+15,x:x+15]	= tensor
							#@from torchvision.transforms import ToPILImage
							#@img = ToPILImage(mode="L")(self.piece_tensors[f"w_P_light"].clone().type(torch.float)).show()
							#@input(self.piece_tensors[f"w_P_light"])

				self.state_imgs[cur_fen] = base_tensor
				del self.state_imgs[prev_fen]
				#from torchvision.transforms import ToPILImage
				#img = ToPILImage(mode="L")(self.state_imgs[cur_fen].clone().type(torch.float)).show()
				#input()
				return self.state_imgs[cur_fen].clone().type(torch.float)
		
		base_tensor		= self.base_tensor.clone()
		for rank_i, cur_rank in enumerate(cur_fen.split(" ")[0].split("/"),1):
			
			rank_i = 9 - rank_i
			cur_rank = cur_rank.replace("1","e").replace("2","ee").replace("3","eee").replace("4","eeee").replace("5","eeeee").replace("6","eeeeee").replace("7","eeeeeee").replace("8","eeeeeeee")
			#print(cur_rank)
			for file_x,square in enumerate(cur_rank,1):

				if ((rank_i % 2 == 0) and (file_x % 2 == 0)) or ((rank_i % 2 == 1) and (file_x % 2 == 1)):
					sq_color = "dark"
				else:
					sq_color = "light"

				if square in ["r","n","b","q","k","p"]:
					color 	= "b"
				elif square in ["R","N","B","Q","K","P"]:
					color 	= "w"

				elif square == 'e':
					color = "w"
					square 	= "e"
				
				tensor_key	= f"{color}_{square}_{sq_color}"
				tensor 		= self.piece_tensors[tensor_key].clone()

				y,x 	= self.coord_to_xy(rank_i,file_x)

				base_tensor[0,y:y+15,x:x+15]	= tensor
			return base_tensor.type(torch.float)

	@staticmethod
	def create_board_img_static(chess_board:chess.Board):

		if not Chess.piece_tensors:
			for fname in os.listdir("C:/gitrepos/steinpy/ml/res"):
				if not "light" in fname and not "dark" in fname:
					continue
				Chess.piece_tensors[fname]	= torch.load(os.path.join("C:/gitrepos/steinpy/ml/res",fname)).to(torch.device('cuda')).requires_grad_(False).type(torch.float)

		cur_fen	= chess_board.fen()
		for rank_i, cur_rank in enumerate(cur_fen.split(" ")[0].split("/"),1):
			
			rank_i = 9 - rank_i
			cur_rank = cur_rank.replace("1","e").replace("2","ee").replace("3","eee").replace("4","eeee").replace("5","eeeee").replace("6","eeeeee").replace("7","eeeeeee").replace("8","eeeeeeee")
			for file_x,square in enumerate(cur_rank,1):
				if ((rank_i % 2 == 0) and (file_x % 2 == 0)) or ((rank_i % 2 == 1) and (file_x % 2 == 1)):
					sq_color = "dark"
				else:
					sq_color = "light"

				if square in ["r","n","b","q","k","p"]:
					color 	= "b"
				elif square in ["R","N","B","Q","K","P"]:
					color 	= "w"

				elif square == 'e':
					color = "w"
					square 	= "e"
				
				tensor_key	= f"{color}_{square}_{sq_color}"
				tensor		= Chess.piece_tensors[tensor_key]

				y,x 	= Chess.coord_to_xy(rank_i,file_x)

				Chess.base_tensor[0,y:y+15,x:x+15]	= tensor
		Chess.base_tensor[1]						*= 1 if chess_board.turn == chess.WHITE else -1

		return Chess.base_tensor.clone()



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


	def __init__(self,board,p=.5,parent=None,c=4):

		self.board 			= board 
		self.parent 		= parent 
		self.children		= {}
		self.num_visited	= 0

		self.Q_val 			= 0 
		self.p				= p 
		self.c 				= c 

		self.score 			= 0


	def get_score(self):
		return self.Q_val + ((self.c * self.p) * (sqrt(sum([m.num_visited for m in self.parent.children.values()])) / (1 + self.num_visited)))
	

class Tree:
	
	def __init__(self,board,model,base_node=None,draw_thresh=250,lookups=0):
		self.board 			= board 
		self.model 			= model 
		self.draw_thresh	= draw_thresh
		self.lookups 		= lookups

		if base_node: 
			self.root 			= base_node
			self.root.parent 	= None
		else:
			self.root 			= Node(board,0,None)
			self.root.parent	= None 


	def update_tree(self,node:Node,x=.75,dirichlet_a=1.0,rollout_p=.1,rollout=False):
		
		#If gameover return game result 
		if node.board.outcome():
			if "1" in node.board.result():
				if node.board.result()[0] == "1":
					node.Q_val 			= (node.num_visited * node.Q_val + 1) / (node.num_visited + 1) 
					node.num_visited 	+= 1 
					return 1 
				else:
					node.Q_val 			= (node.num_visited * node.Q_val + -1) / (node.num_visited + 1) 
					node.num_visited 	+= 1 
					return -1 
			else:
				node.Q_val 			= (node.num_visited * node.Q_val + 0) / (node.num_visited + 1) 
				node.num_visited 	+= 1 
				return 0 
			
		#If not explored, expand out children and return v value
		if not node.children:
		
			#Initialize repr 
			node.repr 			= Chess.create_board_img_static(node.board)
			#Get prob and v
			
			with torch.no_grad():
				prob,v 			= self.model.forward(node.repr.unsqueeze(0))
			#NUMPY VERSION 	
			prob			= prob.cpu()
			legal_moves 	= [Chess.chess_moves.index(m.uci()) for m in node.board.legal_moves]
			legal_probs 	= [prob[0][i] for i in legal_moves]
			#Apply dirichlet noise 
			noise 			= Chess.noise.dirichlet([dirichlet_a for _ in range(len(legal_probs))],len(legal_probs))
			legal_probs		= scipy.special.softmax([x*p for p in legal_probs] + (1-x)*noise)[0]

			#Add to lookup 
			#Chess.lookup_table[node.board.fen()]	= (node.repr,legal_probs,legal_moves,v)
			#NUMPY VERSION

			for prob,move_i in zip(legal_probs,legal_moves):
				
				#clone board and push move 
				child_board										= node.board.copy()
				child_board.push_san(Chess.chess_moves[move_i])
				node.children[Chess.chess_moves[move_i]]		= Node(child_board,p=prob,parent=node)
			
			if random.random() < rollout_p:
				v = self.rollout(child_board.copy())
				node.Q_val 			= (node.num_visited * node.Q_val + v) / (node.num_visited + 1) 
				node.num_visited 	+= 1 
				return -v
			else:
				node.Q_val 			= (node.num_visited * node.Q_val + v) / (node.num_visited + 1) 
				node.num_visited 	+= 1 
				return -v 

		#best_score			= float('inf') * -1 
		best_node 			= None 


		best_node = max(list(node.children.values()),key = lambda x: x.get_score())

		v 					= self.update_tree(best_node)
		#Update Q val 
		node.Q_val 			= (node.num_visited * node.Q_val + v) / (node.num_visited + 1) 
		node.num_visited 	+= 1 

		#Return the negative val (for opposing player)
		return -v 

	def update_tree_memoize(self,node:Node,x=.75,dirichlet_a=1.0,rollout_p=.5,rollout=False):
		
		#If gameover return game result 
		if node.board.outcome():
			if "1" in node.board.result():
				if node.board.result()[0] == "1":
					node.Q_val 			= (node.num_visited * node.Q_val + 1) / (node.num_visited + 1) 
					node.num_visited 	+= 1 
					return 1 
				else:
					node.Q_val 			= (node.num_visited * node.Q_val + -1) / (node.num_visited + 1) 
					node.num_visited 	+= 1 
					return -1 
			else:
				node.Q_val 			= (node.num_visited * node.Q_val + 0) / (node.num_visited + 1) 
				node.num_visited 	+= 1 
				return 0 
			
		#If not explored, expand out children and return v value
		if not node.children:
			if node.board.fen() in Chess.lookup_table:
				node.repr,legal_probs,legal_moves,v = Chess.lookup_table[node.board.fen()]
				#print(f"used lookup")
			else:
				#Initialize repr 
				node.repr 			= Chess.create_board_img_static(node.board)
				#Get prob and v
				
				with torch.no_grad():
					prob,v 			= self.model(node.repr.view(1,2,130,130))
				#NUMPY VERSION 	
				prob			= prob.cpu()
				legal_moves 	= [Chess.chess_moves.index(m.uci()) for m in node.board.legal_moves]
				legal_probs 	= [prob[0][i] for i in legal_moves]
				#Apply dirichlet noise 
				noise 			= Chess.noise.dirichlet([dirichlet_a for _ in range(len(legal_probs))],len(legal_probs))
				legal_probs		= [x*p for p in legal_probs] + (1-x)*noise
				legal_probs		= scipy.special.softmax(legal_probs)[0]

				#Add to lookup 
				Chess.lookup_table[node.board.fen()]	= (node.repr,legal_probs,legal_moves,v)
				#NUMPY VERSION

			for prob,move_i in zip(legal_probs,legal_moves):
				
				#clone board and push move 
				child_board										= node.board.copy()
				child_board.push_san(Chess.chess_moves[move_i])
				node.children[Chess.chess_moves[move_i]]		= Node(child_board,p=prob,parent=node)
			
			if random.random() < rollout_p:
				v = self.rollout(child_board.copy())
				node.Q_val 			= (node.num_visited * node.Q_val + v) / (node.num_visited + 1) 
				node.num_visited 	+= 1 
				return -v
			else:
				node.Q_val 			= (node.num_visited * node.Q_val + v) / (node.num_visited + 1) 
				node.num_visited 	+= 1 
				return -v 

		#best_score			= float('inf') * -1 
		best_node 			= None 


		best_node = max(list(node.children.values()),key = lambda x: x.get_score())

		


		v 					= self.update_tree_memoize(best_node)
		#Update Q val 
		node.Q_val 			= (node.num_visited * node.Q_val + v) / (node.num_visited + 1) 
		node.num_visited 	+= 1 

		#Return the negative val (for opposing player)
		return -v 

	def rollout(self,board:chess.Board):
		started 	= board.turn
		
		#If gameover return game result 
		res 	= board.result()
		while res == "*":
			
			#If over draw thresh, return 0
			if board.ply() > self.draw_thresh:
				return 0
			
			#Else make move according to rollout policy (currently random)
			board.push_uci(random.choice(list(board.legal_moves)).uci())
			res 	= board.result()
		
		if res[0] == "1":
			v = 1 
		elif res[-1] == "1":
			v =  -1 
		else:
			v = 0

		return -v if started == board.turn else v

	def rollout_exp(self,board:chess.Board):
		started 	= board.turn
		
		#If gameover return game result 
		while not (board.is_checkmate() or board.is_stalemate() or board.is_seventyfive_moves() or board.is_fifty_moves()):
			
			#If over draw thresh, return 0
			if board.ply() > self.draw_thresh:
				return 0
			
			#Else make move according to rollout policy (currently random)
			board.push_uci(random.choice(list(board.legal_moves)).uci())
		
		res = board.result()
		if res[0] == "1":
			v = 1 
		elif res[-1] == "1":
			v =  -1 
		else:
			v = 0

		return -v if started == board.turn else v	  
	
	@torch.no_grad()
	def update_tree_nonrecursive(self,x=.9,dirichlet_a=1.0,rollout_p=.02,iters=300):
		
		#DEFINE FUNCTIONS IN LOCAL SCOPE 
		infer 					= self.model.forward
		create_repr				= Chess.create_board_img_static
		noise_gen				= numpy.random.default_rng().dirichlet
		softmax_fn				= scipy.special.softmax
		chessmoves_indexer 		= Chess.chess_moves.index
		chess_moves 			= json.loads(open(os.path.join("C:/","gitrepos","steinpy","ml","res","chessmoves.txt"),"r").read())
		lookups 				= 0
		lookup_table			= Chess.lookup
		
		for _ in range(iters):
			node = self.root
			score_mult = 1 if node.board.turn == chess.WHITE else -1 

			while node.children:

				#drive down to leaf
				best_node 			= max(list(node.children.values()),key = lambda x: x.get_score())
				node 				= best_node
				score_mult			*= -1

			#Check if game over
			if node.board.outcome():
				if "1" in node.board.result():
					if node.board.result()[0] == "1":
						v 	=   1 * score_mult
					else:
						v 	=  -1 * score_mult
				else:
					v 	=  0 
				
			
			#expand 
			else:
				node.repr 			= create_repr(node.board)
				 
				# if node.board.fen() in Chess.lookup:
				# 	v,legal_moves,legal_probs 				= lookup_table[node.board.fen()] 
				# 	self.lookups += 1

				prob_cpu:torch.Tensor
				prob,v 				= infer(node.repr.view(1,2,130,130))
				prob_cpu			= prob.to(torch.device('cpu'),non_blocking=True)
				legal_moves 		= [chessmoves_indexer(m.uci()) for m in node.board.legal_moves]
				legal_probs 		= [prob_cpu[0,i] for i in legal_moves]
				#lookup_table[node.board.fen()]		= (v,legal_moves,legal_probs)

				noise 				= noise_gen([dirichlet_a for _ in range(len(legal_probs))],len(legal_probs))
				legal_probs			= softmax_fn([x*p for p in legal_probs] + (1-x)*noise)[0]

				node.children 		= {chess_moves[move_i] : Node(node.board.copy(),p=p.item(),parent=node) for p,move_i in zip(legal_probs,legal_moves)} 
				
				#map(lambda key: node.children[key].board.push_san(key),node.children)
				for move in node.children:
					node.children[move].board.push_san(move)
				# 	node.children[Chess.chess_moves[move_i]]		= Node(child_board,p=prob,parent=node)
				
				if random.random() < rollout_p:
					v = self.rollout(node.board.copy()) * score_mult
			
			#input(f"node is {node}\nchild is {list(node.children.values())[0]}")
			
			while node.parent:
				node.Q_val 			= (node.num_visited * node.Q_val + v) / (node.num_visited + 1) 
				node.num_visited 	+= 1 
				v *= -1 
				node = node.parent

		return {move:self.root.children[move].num_visited for move in self.root.children}

	def update_tree_nonrecursive_exp(self,x=.9,dirichlet_a=1.0,rollout_p=.02,iters=300):
		
		#DEFINE FUNCTIONS IN LOCAL SCOPE 
		infer 					= self.model.forward
		create_repr				= Chess.create_board_img_static
		noise_gen				= numpy.random.default_rng().dirichlet
		softmax_fn				= scipy.special.softmax
		chessmoves_indexer 		= Chess.chess_moves.index
		chess_moves 			= json.loads(open(os.path.join("C:/","gitrepos","steinpy","ml","res","chessmoves.txt"),"r").read())
		lookups 				= 0
		lookup_table			= Chess.lookup
		legal_move_table		= Chess.legal_move_table
		
		for _ in range(iters):
			node = self.root
			score_mult = 1 if node.board.turn == chess.WHITE else -1 

			while node.children:

				#drive down to leaf
				best_node 			= max(list(node.children.values()),key = lambda x: x.get_score())
				node 				= best_node
				score_mult			*= -1

			#Check if game over
			if node.board.is_checkmate() or node.board.is_stalemate() or node.board.is_seventyfive_moves() or node.board.is_fifty_moves():
				
				if "1" in node.board.result():
					if node.board.result()[0] == "1":
						v 	=   1 * score_mult
					else:
						v 	=  -1 * score_mult
				else:
					v 	=  0 
				
			
			#expand 
			else:
				node.repr 			= create_repr(node.board)
				 
				# if node.board.fen() in Chess.lookup:
				# 	v,legal_moves,legal_probs 				= lookup_table[node.board.fen()] 
				# 	self.lookups += 1

				prob_cpu:torch.Tensor
				with torch.no_grad():
					prob,v 				= infer(node.repr.view(1,2,130,130))
				prob_cpu			= prob[0].to(torch.device('cpu'),non_blocking=True)
				legal_moves 		= [chessmoves_indexer(m.uci()) for m in node.board.legal_moves]
				legal_probs 		= [prob_cpu[i] for i in legal_moves]

				noise 				= noise_gen([dirichlet_a for _ in range(len(legal_probs))],len(legal_probs))
				legal_probs			= softmax_fn([x*p for p in legal_probs] + (1-x)*noise)[0]

				node.children 		= {chess_moves[move_i] : Node(node.board.copy(),p=p,parent=node) for p,move_i in zip(legal_probs,legal_moves)} 
				
				#map(lambda key: node.children[key].board.push_san(key),node.children)
				for move in node.children:
					node.children[move].board.push_san(move)
				# 	node.children[Chess.chess_moves[move_i]]		= Node(child_board,p=prob,parent=node)
				
				if random.random() < rollout_p:
					v = self.rollout_exp(random.choice(list(node.children.values())).board.copy()) * score_mult
				
			#input(f"node is {node}\nchild is {list(node.children.values())[0]}")
			
			while node.parent:
				node.Q_val 			= (node.num_visited * node.Q_val + v) / (node.num_visited + 1) 
				node.num_visited 	+= 1 

				v *= -1 
				node = node.parent

		return {move:self.root.children[move].num_visited for move in self.root.children}



	def get_policy(self,search_iters):

		#return self.update_tree(iters=search_iters)

		with torch.no_grad():
			for _ in range(search_iters):
				self.update_tree(self.root)
		return {move:self.root.children[move].num_visited for move in self.root.children}