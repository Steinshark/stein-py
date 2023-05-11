import numpy 
import math 
import json 
import os 

#   General function to interpolate an array
#   To a smaller size  
def reduce_arr(arr,newlen):

    #Find GCF of len(arr) and len(newlen)
    gcf         = math.gcd(len(arr),newlen)
    mult_fact   = int(newlen / gcf) 
    div_fact    = int(len(arr) / gcf) 

    new_arr     = numpy.repeat(arr,mult_fact)

    return [sum(list(new_arr[n*div_fact:(n+1)*div_fact]))/div_fact for n in range(newlen)]


#Responsible for writing objects to the file server
def db_writer(fpath,fdest):

    folders     = fpath.split(os.sep)

    
    
