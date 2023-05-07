#   Author:     Everett Stenbeg 
#   Github:     Steinshark


import torch 
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
                 optimizer_kwargs={"lr":1e-5,"weight_decay":1e-5},
                 device=torch.device('cuda'),
                 n_ch=1
                 ):
        
        super(ImgNet,self).__init__(loss_fn=loss_fn,optimizer=optimizer,optimizer_kwargs=optimizer_kwargs,device=device)


        self.conv_layers          = torch.nn.Sequential(
            torch.nn.Conv2d(n_ch,32,3,1,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32,64,3,1,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),

            torch.nn.Conv2d(64,128,5,1,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),

            torch.nn.Conv2d(128,128,5,1,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(2),
            torch.nn.AvgPool2d(2),

            torch.nn.Conv2d(128,128,5,1,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(2),
            torch.nn.AvgPool2d(2),

            torch.nn.Conv2d(128,128,5,1,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(2),
            torch.nn.AvgPool2d(2),      

            torch.nn.Conv2d(128,128,7,1,2),
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
            
            torch.nn.Conv2d(128,256,7,1,2),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            #torch.nn.MaxPool2d(2),
            torch.nn.AvgPool2d(2),

        
            torch.nn.Flatten(),
        
            torch.nn.Linear(1024,512),
            torch.nn.LeakyReLU(negative_slope=.02),
            torch.nn.Dropout(.4),

            torch.nn.Linear(512,128),
            torch.nn.LeakyReLU(negative_slope=.02),
            torch.nn.Dropout(.2),

            torch.nn.Linear(128,1),
            torch.nn.Tanh()
            ).to(device)
        
        self.model  = torch.nn.ModuleList([self.conv_layers,self.probability_head,self.value_head])
        self.set_training_vars()


    def forward(self,x):

        #Ensure input is batched properly 
        if not len(x.shape) == 4:
            raise RuntimeError(f"Bad input shape. Requires 4D input, found {len(x.shape)}D")

        conv_output         = self.conv_layers(x)

        probability_distr   = self.probability_head(conv_output)
        value_prediction    = self.value_head(conv_output)

        return probability_distr,value_prediction
