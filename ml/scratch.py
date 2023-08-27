import games 
import time 
from sklearn.utils import extmath 
import numpy 
import random
import torch 
import itertools 
DATASET_ROOT  	=	 r"//FILESERVER/S Drive/Data/chess"
import os 

import networks
from torch.utils.data import DataLoader
#from steinpy.ml.rl_notorch import Tree, pre_network_call,post_network_call
import multiprocessing
import sys 
import numba 
import chess 




if __name__ == "__main__":
	
	sorted_list 	= [(0,None)]
	total_time 		= 0 
	for line in open("out2.txt","r").readlines():
		c 				= [l for l in line.split(" ") if not l == ""]

		try:
			c[1] 		= float(c[1])
			total_time 	+= c[1]

			#place in sorted list 
			inserted 	= False 
			for i in range(len(sorted_list)):
				pair 		= sorted_list[i]
				cost,item 	= pair 
				if c[1] < cost:
					pass 
				else:
					sorted_list.insert(i,(c[1],"".join(c[5:])))
					inserted 	= True
					break 
			if not inserted:
				sorted_list.append((c[1],"".join(c[5:])))

		except ValueError:
			pass
		except IndexError:
			pass

	sorted_list 	= [(f"{c[0]:.3f}",c[1]) for c in sorted_list]
	import pprint 
	pprint.pp(sorted_list[:100])