#   Author:     Everett Stenbeg 
#   Github:     Steinshark


import torch 
from torch.utils.data import Dataset
from typing import OrderedDict

#   This type of network will contain all 
#   items necessary to train as-is 
#   Containerizes the network, optmizer,
#   Loss function, etc
class FullNet(torch.nn.Module):

	def __init__(self,
				 loss_fn=torch.nn.MSELoss,
				 optimizer=torch.optim.SGD,
				 optimizer_kwargs={"lr":1e-4,"weight_decay":1e-5,"momentum":.9},
				 device=torch.device('cpu')):

		#Init parent class 
		super(FullNet,self).__init__()

		#Set model variables 
		self.model              = None 
		self.device             = device 
		
		#Set training variables 
		self.loss_fn            = loss_fn
		self.optimizer          = optimizer
		self.optimizer_kwargs   = optimizer_kwargs

	
	def set_training_vars(self):

		#Ensure model exists 
		if not isinstance(self.model,torch.nn.Module):
			raise ValueError(f"'model' must be torch Module. Found {type(self.model)}")
		
		#Set training variables 
		self.loss               = self.loss_fn()
		self.optimizer          = self.optimizer(self.model.parameters(),**self.optimizer_kwargs)

	def forward(self):
		raise NotImplementedError(f"'forward' has not been implemented")


class LinearNet(FullNet):

	def __init__(self,
				 architecture,
				 activation_fn=torch.nn.ReLU,
				 loss_fn=torch.nn.MSELoss,
				 optimizer=torch.optim.SGD,
				 optimizer_kwargs={"lr":1e-4,"weight_decay":1e-5,"momentum":.9},
				 device=torch.device('cpu')
				 ):
		
		#Init parent class (FullNet)
		super().__init__(loss_fn=loss_fn,optimizer=optimizer,optimizer_kwargs=optimizer_kwargs,device=device)

		#Build the network 
		module_list         = OrderedDict()
		for i,layer in enumerate(architecture):
			module_list[str(i*2)]   = torch.nn.Linear(layer[0],layer[1])
			module_list[str(i*2+1)] = activation_fn()
		self.model          = torch.nn.Sequential(module_list).to(self.device)

		#Set training vars
		self.set_training_vars()

	
	def forward(self,x):

		#Ensure input is batched properly 
		if not len(x.shape) == 2:
			raise RuntimeError(f"Bad input shape. Requires 2D input, found {len(x.shape)}D")

		return self.model(x)
	

class Conv2dNet(FullNet):


	#Creates a standard convolutional network based off of 'architecure'
	def __init__(self,
				 architecture,
				 activation_fn=torch.nn.ReLU,
				 loss_fn=torch.nn.MSELoss,
				 optimizer=torch.optim.SGD,
				 optimizer_kwargs={"lr":1e-4,"weight_decay":1e-5,"momentum":.9},
				 device=torch.device('cpu')
				 ):
		
		#Init parent class (FullNet)
		super().__init__(loss_fn=loss_fn,optimizer=optimizer,optimizer_kwargs=optimizer_kwargs,device=device)

		#Build the network as per architecture
		module_list         = OrderedDict()

		#Add convolution portion of network
		for i,layer in enumerate(architecture['convlayers']):
			layer_len                               = len(layer.keys())
			#Add conv layer 
			clist                                   = layer['conv']
			kwargs                                  = {"in_channels":clist[0],"out_channels":clist[1],"kernel_size":clist[2],"stride":clist[3],"padding":clist[4]}
			module_list[str(len(module_list)+1)]    = torch.nn.Conv2d(**kwargs)

			#Add activation layer 
			activation                              = layer['act'] 
			module_list[str(len(module_list)+1)]    = activation

			#Add batchnorm layer 
			if 'bnorm' in layer:
				module_list[str(len(module_list)+1)]    = layer['bnorm']

			#Add maxpool layer 
			if 'mpool' in layer:
				module_list[str(len(module_list)+1)]    = layer['mpool']

		#Add flatten 
		module_list[str(len(module_list)+1)]        = torch.nn.Flatten()

		#Add linear portion of network
		for j,layer in enumerate(architecture['linlayers']):
			module_list[str(len(module_list)+1)]        = torch.nn.Linear(layer[0],layer[1])
			module_list[str(len(module_list)+1)]        = activation_fn()

		self.model                                  = torch.nn.Sequential(module_list).to(self.device)

		#Set training vars
		self.set_training_vars()

	
	def forward(self,x):

		#Ensure input is batched properly 
		if not len(x.shape) == 4:
			raise RuntimeError(f"Bad input shape. Requires 2D input, found {len(x.shape)}D")

		return self.model(x)
	

class ImgNet(FullNet):

	def __init__(self,
				 loss_fn=torch.nn.MSELoss,
				 optimizer=torch.optim.Adam,
				 optimizer_kwargs={"lr":1e-5,"weight_decay":1e-6},
				 device=torch.device('cpu'),
				 n_ch=1
				 ):
		
		super(ImgNet,self).__init__(loss_fn=loss_fn,optimizer=optimizer,optimizer_kwargs=optimizer_kwargs,device=device)


		self.conv_layers          = torch.nn.Sequential(
			torch.nn.Conv2d(n_ch,32,3,1,1,bias=False),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),

			torch.nn.Conv2d(32,64,3,1,1,bias=False),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),

			torch.nn.Conv2d(64,128,5,1,1,bias=False),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),

			torch.nn.Conv2d(128,128,5,1,1,bias=False),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			#torch.nn.MaxPool2d(2),
			torch.nn.AvgPool2d(2),

			torch.nn.Conv2d(128,128,5,1,1,bias=False),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			#torch.nn.MaxPool2d(2),
			torch.nn.AvgPool2d(2),

			torch.nn.Conv2d(128,128,5,1,1,bias=False),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			#torch.nn.MaxPool2d(2),
			torch.nn.AvgPool2d(2),      

			torch.nn.Conv2d(128,128,7,1,2,bias=False),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			#torch.nn.MaxPool2d(2),
			torch.nn.AvgPool2d(2)
			).to(device)

		self.probability_head = torch.nn.Sequential(
			torch.nn.Flatten(),
			torch.nn.Linear(4608,2048),
			torch.nn.LeakyReLU(negative_slope=.02),
			torch.nn.Dropout(.4),

			torch.nn.Linear(2048,2048),
			torch.nn.LeakyReLU(negative_slope=.02),
			torch.nn.Dropout(.2),

			torch.nn.Linear(2048,1968),
			torch.nn.Softmax(dim=1)

			).to(device)
		
		self.value_head = torch.nn.Sequential(
			
			torch.nn.Conv2d(128,128,7,1,2,bias=False),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			#torch.nn.MaxPool2d(2),
			torch.nn.AvgPool2d(2),

		
			torch.nn.Flatten(),
		
			torch.nn.Linear(1024,512),
			torch.nn.LeakyReLU(negative_slope=.02),
			torch.nn.Dropout(.25),

			torch.nn.Linear(512,128),
			torch.nn.LeakyReLU(negative_slope=.02),
			torch.nn.Dropout(.1),

			torch.nn.Linear(128,1),
			torch.nn.Tanh()
			).to(device)
		
		self.model  = torch.nn.ModuleList([self.conv_layers,self.probability_head,self.value_head])
		self.set_training_vars()


	def forward(self,x):

		conv_output         = self.conv_layers(x)

		probability_distr   = self.probability_head(conv_output)
		value_prediction    = self.value_head(conv_output)

		return probability_distr,value_prediction


class ImgNet2(FullNet):

	def __init__(self,
				 loss_fn=torch.nn.MSELoss,
				 optimizer=torch.optim.Adam,
				 optimizer_kwargs={"lr":1e-5,"weight_decay":1e-6},
				 device=torch.device('cuda'),
				 n_ch=1
				 ):
		
		super(ImgNet2,self).__init__(loss_fn=loss_fn,optimizer=optimizer,optimizer_kwargs=optimizer_kwargs,device=device)


		self.conv_layers          = torch.nn.Sequential(
			torch.nn.Conv2d(n_ch,32,3,1,1,bias=False),
			torch.nn.BatchNorm2d(32),
			torch.nn.ReLU(),

			torch.nn.Conv2d(32,64,3,1,1,bias=False),
			torch.nn.BatchNorm2d(64),
			torch.nn.ReLU(),

			torch.nn.Conv2d(64,128,3,1,1,bias=False),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			torch.nn.AvgPool2d(2),

			torch.nn.Conv2d(128,128,3,1,1,bias=False),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			#torch.nn.MaxPool2d(2),
			torch.nn.AvgPool2d(2),

			torch.nn.Conv2d(128,128,3,1,1,bias=False),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			#torch.nn.MaxPool2d(2),
			torch.nn.AvgPool2d(2),

			torch.nn.Conv2d(128,128,3,1,1,bias=False),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			#torch.nn.MaxPool2d(2),
			torch.nn.AvgPool2d(2),      

			torch.nn.Conv2d(128,128,3,1,2,bias=False),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			#torch.nn.MaxPool2d(2),
			torch.nn.AvgPool2d(2),

			torch.nn.Conv2d(128,128,3,1,2,bias=False),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),

			torch.nn.Conv2d(128,128,3,1,2,bias=False),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			#torch.nn.MaxPool2d(2),
			torch.nn.AvgPool2d(2),


			).to(device)

		self.probability_head = torch.nn.Sequential(
			torch.nn.Flatten(),
			torch.nn.Linear(4096,1968),
			# torch.nn.LeakyReLU(negative_slope=.02),
			# torch.nn.Dropout(.4),

			# torch.nn.Linear(2048,2048),
			# torch.nn.LeakyReLU(negative_slope=.02),
			# torch.nn.Dropout(.2),

			# torch.nn.Linear(2048,1968),
			torch.nn.Softmax(dim=1)

			).to(device)
		
		self.value_head = torch.nn.Sequential(
			
			torch.nn.Conv2d(128,128,3,1,2,bias=False),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			#torch.nn.MaxPool2d(2),
			torch.nn.AvgPool2d(2),

			torch.nn.Conv2d(128,128,3,1,2,bias=False),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			#torch.nn.MaxPool2d(2),
			torch.nn.AvgPool2d(2),

			torch.nn.Conv2d(128,128,3,1,2,bias=False),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			#torch.nn.MaxPool2d(2),
			torch.nn.AvgPool2d(2),

			torch.nn.Conv2d(128,128,3,1,2,bias=False),
			torch.nn.BatchNorm2d(128),
			torch.nn.ReLU(),
			#torch.nn.MaxPool2d(2),
			torch.nn.AvgPool2d(2),

		
			torch.nn.Flatten(),
		
			torch.nn.Linear(1024,128),
			torch.nn.LeakyReLU(negative_slope=.02),
			torch.nn.Dropout(.25),

			torch.nn.Linear(128,64),
			torch.nn.LeakyReLU(negative_slope=.02),
			torch.nn.Dropout(.1),

			torch.nn.Linear(64,1),
			torch.nn.Tanh()
			).to(device)
		
		self.model  = torch.nn.ModuleList([self.conv_layers,self.probability_head,self.value_head])
		self.set_training_vars()


	def forward(self,x):

		conv_output         = self.conv_layers(x)

		probability_distr   = self.probability_head(conv_output)
		value_prediction    = self.value_head(conv_output)

		return probability_distr,value_prediction


class ChessNet(FullNet):
	def __init__(self,
				 loss_fn=torch.nn.MSELoss,
				 optimizer=torch.optim.Adam,
				 optimizer_kwargs={"lr":1e-5,"weight_decay":1e-6},
				 device=torch.device('cuda'),
				 n_ch=19,
				 n_layers=18,
				 act_fn=torch.nn.ReLU
				 ):
		
		super(ChessNet,self).__init__(loss_fn=loss_fn,optimizer=optimizer,optimizer_kwargs=optimizer_kwargs,device=device)
		self.act_fn					= act_fn
		self.conv_layers_res     	= [] 
		layers          			=    {i:128 for i in range(n_layers)}
		self.n_layers   			= n_layers
		for i in range(len(layers)):
			if i == 0:
				prev_ch  = n_ch 
			else:
				prev_ch = layers[i-1]
			
			cur_ch = layers[i]

			self.conv_layers_res.append(
				torch.nn.Sequential(
					torch.nn.Conv2d(prev_ch,cur_ch,3,1,1,bias=False),
					torch.nn.BatchNorm2d(cur_ch),
				
					act_fn(),

					torch.nn.Conv2d(cur_ch,cur_ch,3,1,1,bias=False),
					torch.nn.BatchNorm2d(cur_ch),
		).to(device))
		


		self.prob_net   = torch.nn.Sequential(  
			torch.nn.Conv2d(cur_ch,4,1,1,1,bias=False),
			torch.nn.BatchNorm2d(4),
			torch.nn.Flatten(),

			torch.nn.Linear(400,1968),
			act_fn(),
			torch.nn.Softmax(dim=1)
		).to(device)

		self.value_net  = torch.nn.Sequential( 
			torch.nn.Conv2d(cur_ch,1,1,1,1,bias=False), 
			torch.nn.BatchNorm2d(1),
			act_fn(),
			torch.nn.Flatten(),

			torch.nn.Linear(100,64), 
			act_fn(),
			
			torch.nn.Linear(64,1),
			torch.nn.Tanh()
		).to(device)
		
		self.model  = torch.nn.ModuleList(self.conv_layers_res+[self.prob_net]+[self.value_net]).to(device)
		self.set_training_vars()

	def forward_old(self,x):

		conv_forward_pass   = self.conv_layers(x) 

		return self.prob_net(conv_forward_pass), self.value_net(conv_forward_pass)

	def forward(self,x):

		conv_layer_1:torch.nn.Sequential = self.conv_layers_res[0]  
		prev_out        = conv_layer_1(x)
		reluer          = self.act_fn()

		for layer in self.conv_layers_res[1:]:
			cur_out         = layer(prev_out)
			prev_out        = reluer(cur_out+prev_out)
			
		return self.prob_net(prev_out), self.value_net(prev_out)


class ChessNetCompat(FullNet):
	def __init__(self,
				 loss_fn=torch.nn.MSELoss,
				 optimizer=torch.optim.Adam,
				 optimizer_kwargs={"lr":1e-5,"weight_decay":1e-6},
				 device=torch.device('cuda'),
				 n_ch=13,
				 n_layers=0
				 ):
		
		super(ChessNetCompat,self).__init__(loss_fn=loss_fn,optimizer=optimizer,optimizer_kwargs=optimizer_kwargs,device=device)
		
		self.l1                     = torch.nn.Sequential(torch.nn.Conv2d(n_ch,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128)).to(device)
		self.l2                     = torch.nn.Sequential(torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128)).to(device)
		self.l3                     = torch.nn.Sequential(torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128)).to(device)
		self.l4                     = torch.nn.Sequential(torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128)).to(device)
		self.l5                     = torch.nn.Sequential(torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128)).to(device)
		self.l6                     = torch.nn.Sequential(torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128)).to(device)
		self.l7                     = torch.nn.Sequential(torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128)).to(device)
		self.l8                     = torch.nn.Sequential(torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128)).to(device)
		self.l9                     = torch.nn.Sequential(torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128)).to(device)
		self.l10                    = torch.nn.Sequential(torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128)).to(device)
		self.l11                    = torch.nn.Sequential(torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128)).to(device)
		self.l12                    = torch.nn.Sequential(torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128)).to(device)
		self.l13                    = torch.nn.Sequential(torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128)).to(device)
		self.l14                    = torch.nn.Sequential(torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128)).to(device)
		#self.l15                    = torch.nn.Sequential(torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128)).to(device)
		#self.l16                    = torch.nn.Sequential(torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128)).to(device)
		#self.l17                    = torch.nn.Sequential(torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128)).to(device)
		#self.l18                    = torch.nn.Sequential(torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128),torch.nn.ReLU(),torch.nn.Conv2d(128,128,3,1,1,bias=False),torch.nn.BatchNorm2d(128)).to(device)
	   

		self.prob_net   = torch.nn.Sequential(  
			torch.nn.Conv2d(128,10,1,1,1,bias=False),
			torch.nn.BatchNorm2d(10),
			torch.nn.Flatten(),

			torch.nn.Linear(1000,1968),

			torch.nn.ReLU(),
			torch.nn.Softmax(dim=1)
		).to(device)

		self.value_net  = torch.nn.Sequential( 
			torch.nn.Conv2d(128,1,1,1,1,bias=False), 
			torch.nn.BatchNorm2d(1),
			torch.nn.ReLU(),
			torch.nn.Flatten(),

			torch.nn.Linear(100,64), 
			torch.nn.ReLU(),
			
			torch.nn.Linear(64,1),
			torch.nn.Tanh()
		).to(device)
		
		self.model  = torch.nn.ModuleList([self.l1,self.l2,self.l3,self.l4,self.l5,self.l6,self.l7,self.l8,self.value_net,self.prob_net]).to(device)#,self.l13,self.l14,self.l15,self.l16,self.l17,self.l18]).to(device)
		self.set_training_vars()
		self.loaded     = False

	def forward(self,x):
		reLU        = torch.nn.functional.relu

		x1          = reLU(self.l1(x))
		x2          = reLU(self.l2(x1)+x1)
		x3          = reLU(self.l3(x2)+x2)
		x4          = reLU(self.l4(x3)+x3)
		x5          = reLU(self.l5(x4)+x4)
		x6          = reLU(self.l6(x5)+x5)
		x7          = reLU(self.l7(x6)+x6)
		x8          = reLU(self.l8(x7)+x7)
		x9          = reLU(self.l9(x8)+x8)
		x10         = reLU(self.l10(x9)+x9)
		x11         = reLU(self.l11(x10)+x10)
		x12         = reLU(self.l12(x11)+x11)
		x13         = reLU(self.l13(x12)+x12)
		x14         = reLU(self.l2(x13)+x13)
		# x15         = reLU(self.l2(x14)+x14)
		# x16         = reLU(self.l2(x15)+x15)
		# x17         = reLU(self.l2(x16)+x16)
		# x18         = reLU(self.l2(x17)+x17)
		# x19         = reLU(self.l2(x18)+x18)

		return self.prob_net(x14), self.value_net(x14)

	
class ChessConvNet(FullNet):

	def __init__(self,act_fn=torch.nn.LeakyReLU,device=torch.device('cuda'if torch.cuda.is_available() else 'cpu'),n_ch=19):
		super(ChessConvNet,self).__init__()

		self.base_model     = torch.nn.Sequential( 
			torch.nn.Conv2d(n_ch,64,5,1,2,bias=False),
			torch.nn.BatchNorm2d(64),
			act_fn(), 

			torch.nn.Conv2d(64,128,5,1,2,bias=False),
			torch.nn.BatchNorm2d(128),
			act_fn(), 

			torch.nn.Conv2d(128,256,5,1,2,bias=False),
			torch.nn.BatchNorm2d(256),
			act_fn(), 

			torch.nn.Conv2d(256,256,5,1,1,bias=False),
			torch.nn.BatchNorm2d(256),
			act_fn(), 

			torch.nn.Conv2d(256,256,5,1,1,bias=False),
			torch.nn.BatchNorm2d(256),
			act_fn()
		).to(device)

		self.prob_net       = torch.nn.Sequential(  
			torch.nn.Conv2d(256,32,1,1,1,bias=True),
			act_fn(),
			torch.nn.Flatten(),

			torch.nn.Linear(576*2,1968),
			act_fn(),
			torch.nn.Softmax(dim=1)
		).to(device)

		self.value_net      = torch.nn.Sequential( 
			torch.nn.Conv2d(256,4,1,1,1,bias=True), 
			act_fn(),
			torch.nn.Flatten(),

			torch.nn.Linear(36*4,64), 
			act_fn(),
			
			torch.nn.Linear(64,1)
			#torch.nn.Tanh()
		).to(device)


		self.module_list    = torch.nn.ModuleList([self.base_model,self.prob_net,self.value_net])


	def forward(self,x):
		base_out            = self.base_model(x)

		return self.prob_net(base_out),self.value_net(base_out)

class ChessConvNetLG(FullNet):

	def __init__(self,act_fn=torch.nn.LeakyReLU,device=torch.device('cuda'if torch.cuda.is_available() else 'cpu'),n_ch=19):
		super(ChessConvNetLG,self).__init__()

		self.base_model     = torch.nn.Sequential( 
			torch.nn.Conv2d(n_ch,64,5,1,2,bias=False),
			torch.nn.BatchNorm2d(64),
			act_fn(), 

			torch.nn.Conv2d(64,128,5,1,2,bias=False),
			torch.nn.BatchNorm2d(128),
			act_fn(), 

			torch.nn.Conv2d(128,256,5,1,2,bias=False),
			torch.nn.BatchNorm2d(256),
			act_fn(), 

			torch.nn.Conv2d(256,512,5,1,1,bias=False),
			torch.nn.BatchNorm2d(512),
			act_fn(), 

			torch.nn.Conv2d(512,1024,5,1,1,bias=False),
			torch.nn.BatchNorm2d(1024),
			act_fn(),

			torch.nn.Conv2d(1024,2048,5,1,1,bias=False),
			torch.nn.BatchNorm2d(2048),
			act_fn(),


		).to(device)

		self.prob_net       = torch.nn.Sequential(  
			torch.nn.Conv2d(2048,64,1,1,1,bias=True),
			act_fn(),
			torch.nn.Flatten(),

			torch.nn.Linear(1024,1968),
			act_fn(),
			torch.nn.Softmax(dim=1)
		).to(device)

		self.value_net      = torch.nn.Sequential( 
			torch.nn.Conv2d(2048,16,1,1,1,bias=True), 
			act_fn(),
			torch.nn.Flatten(),

			torch.nn.Linear(256,256), 
			act_fn(),

			torch.nn.Linear(256,64), 
			act_fn(),
			
			torch.nn.Linear(64,1)
			#torch.nn.Tanh()
		).to(device)


		self.module_list    = torch.nn.ModuleList([self.base_model,self.prob_net,self.value_net])


	def forward(self,x):
		base_out            = self.base_model(x)

		return self.prob_net(base_out),self.value_net(base_out)

class ChessDataset(Dataset):

	def __init__(self,experience_set):
		self.data   = experience_set

	
	def __getitem__(self, i):
		return self.data[i]

	def __len__(self):
		return len(self.data)
	

class Model1(FullNet):

	def __init__(self,optimizer=torch.optim.Adam,act_fn=torch.nn.ReLU,optimizer_kwargs={"lr":1e-3,"weight_decay":2.5e-4,},device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')):
		super(Model1,self).__init__()


		self.convlayers	= torch.nn.Sequential(
			torch.nn.Conv2d(1,8,3,1,1),
			act_fn(),

			torch.nn.Conv2d(8,16,3,1,1),
			act_fn(),

			torch.nn.Conv2d(16,32,3,1,1),
			act_fn(),

			torch.nn.Conv2d(32,64,3,1,1),
			act_fn(),
		)

		self.policy_head	= torch.nn.Sequential(
			torch.nn.Flatten(),

			torch.nn.Linear(3072,1024),
			act_fn(),

			torch.nn.Linear(1024,128),
			act_fn(),

			torch.nn.Linear(128,8),
			torch.nn.Softmax(dim=1)
		)

		self.value_head	= torch.nn.Sequential(
			torch.nn.Flatten(),

			torch.nn.Linear(3072,1024),
			act_fn(),

			torch.nn.Linear(1024,128),
			act_fn(),

			torch.nn.Linear(128,1),
			torch.nn.Tanh(),
		)

		self.model 	= torch.nn.ModuleList([self.convlayers,self.policy_head,self.value_head]).to(torch.device('cuda' if torch.cuda.is_available() else "cpu"))

		self.set_training_vars()

	def forward(self,x:torch.Tensor):
		if len(x.shape) == 3:
			x = x.unsqueeze(dim=1)
		conv_out 	= self.convlayers(x)
		return self.policy_head(conv_out),self.value_head(conv_out)


class ChessSmall(FullNet):

	def __init__(self,optimizer=torch.optim.Adam,act_fn=torch.nn.ReLU,optimizer_kwargs={"lr":1e-3,"weight_decay":2.5e-4,},device=torch.device('cuda'if torch.cuda.is_available() else 'cpu'),n_ch=6):
		super(ChessSmall,self).__init__()


		#HYPERPARAMETERS 

		kernel_size 		= 3 

		self.conv_layers	= torch.nn.Sequential(
			torch.nn.Conv2d(n_ch,128,kernel_size,1,1),
			torch.nn.ReLU(),

			torch.nn.Conv2d(128,128,kernel_size,1,1),
			torch.nn.ReLU(),

			torch.nn.Conv2d(128,128,kernel_size,1,1),
			torch.nn.ReLU(),

			torch.nn.Conv2d(128,256,kernel_size,1,1),
			torch.nn.ReLU()
		).to(device)

		self.policy_head       = torch.nn.Sequential(  
			torch.nn.Conv2d(256,32,1,1,1,bias=True),
			act_fn(),
			torch.nn.Flatten(),

			torch.nn.Linear(3200,1968),
			act_fn(),
			torch.nn.Softmax(dim=1)
		).to(device)

		self.value_head      = torch.nn.Sequential( 
			torch.nn.Conv2d(256,16,1,1,1,bias=True), 
			act_fn(),
			torch.nn.Flatten(),

			torch.nn.Linear(1600,256), 
			act_fn(),

			torch.nn.Linear(256,64), 
			act_fn(),
			
			torch.nn.Linear(64,1)
			#torch.nn.Tanh()
		).to(device)


		self.model 			= torch.nn.ModuleList([self.conv_layers,self.policy_head,self.value_head]) 
		self.set_training_vars()
	def forward(self,x:torch.Tensor):
		if len(x.shape) == 3:
			x = x.unsqueeze_(0)
		conv_out 	= self.conv_layers(x)
		return self.policy_head(conv_out),self.value_head(conv_out)