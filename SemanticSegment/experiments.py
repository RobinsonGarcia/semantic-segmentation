import argparse
import pickle
import os
from search_hyperparams.RandomSearch import *

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir',help='Set the relative location to save the results from experiments.py.')
parser.add_argument('--number_experiments',help='Set a number of random experiments to run')
parser.add_argument('--random_params',help='Set the relative location to the pickle file containing the random search specification.')


args = vars(parser.parse_args())

if __name__=="__main__":
    worker =os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train.py')


    hm = os.getcwd()

    try:
        root = os.path.join(hm,args['model_dir']) #sys.argv[1]
    except:
        root = input("Enter project's root path:")

    try:
        n_models = int(args['number_experiments']) #sys.argv[2]
    except:
        n_models =int(input("Enter a number of random experiments to run:"))

    try:
        params_path =os.path.join(hm, args['random_params'])
    except:
        params_path = input('Enter path to random_params.json')

    folders = os.listdir()
    if root not in folders: os.system('mkdir '+root)
    if root+'/tmp' not in folders: os.system('mkdir '+root+'/tmp')
    if root+'/experiments'not in folders: os.system('mkdir '+root+'/experiments')
    if root+'/experiments/config'not in folders: os.system('mkdir '+root+'experimentss/config')

    #================================Generate Random Models================================#

    f = open(params_path,"rb")
    args = pickle.load(f)
    f.close()

    rmodels = RandomModels(**args)
    rmodels.build_configs(n_models,path=root+'/experiments/params')

    pipeline_text = open('pipeline.sh','w')

    pipeline_text.write('#!/bin/bash \n\n')

    pipeline = ''
    tensorboard = 'tensorboard --logdir '
    for i in range(n_models):
        os.system('mkdir '+root+'/experiments/experiment'+str(i).zfill(4))
        pipeline+='python '+worker+' '+root+'/experiments/params/params'+str(i).zfill(4)+'.json '+root+'/experiments/experiment'+str(i).zfill(4)+'| tee -a '+root+'/experiments/experiment'+str(i).zfill(4)+'/pipeline_log.txt -a pipeline_log.txt;\n'
    pipeline_text.write(pipeline)

    pipeline_text.close()

    os.system('sudo chmod +x pipeline.sh')
    os.system('tmux new -s hyper_tunning "sh pipeline.sh"')
