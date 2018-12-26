import json
import os
import numpy as np
from numpy.random import uniform,choice
#==================INPUT DATA=================================#
'''
Input any dictionary with kwargs knowing that:
    tuples - will be treated as a uniform distribution ranging from the first to the second element.
    lists - random choice of one of the list elements
    int,floats - Use this value to all configs
'''


class RandomModels:
    def __init__(self,**kwargs):
            for k,v in kwargs.items():
                print(v,type(v))
                setattr(self,k,v)
            
            self.kwargs = kwargs
            self.filenames = []
            self.configs = []


    def build_configs(self,number_of_models=1,path='tmp'):
        for i in range(number_of_models):
            config = {}
            for k,v in self.kwargs.items():
                if type(v)==tuple:
                    config[k]=float(10**uniform(v[0],v[1]))
                elif type(v)==list:
                    if type(v[0])==str:
                        config[k]=choice(v)
                    elif type(v[0])==int:
                        config[k]=int(choice(v))
                    else:
                        config[k]=float(choice(v))
                else:
                    config[k]=v

            self.configs.append(config)
            filename = path+'/params'+str(i).zfill(4)+'.json'

            if path not in os.listdir():os.system('mkdir '+path)

            json.dump(config,open(filename,'w'))
            self.filenames.append(filename)
    
    def showfile(self,name=None):
        if name==None:
            print('Choose one file:')
            print('verify filenames as self.filenames')

            return self.filenames
        else:
            f = open(name,"r")
            config = json.loads(f.read())
            for k,v in config.items():
                print(k,v)
            f.close()
            pass
    

