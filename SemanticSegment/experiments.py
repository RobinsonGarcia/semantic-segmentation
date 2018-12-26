import os
from search_hyperparams.RandomSearch import *
import sys
worker = 'train.py'

try:
    root = sys.argv[1]
except:
    root = input("Enter project's root path:")

try:
    n_models = sys.argv[2]
except:
    n_models =int(input("Enter a number of random experiments to run:"))

folders = os.listdir()
if root not in folders: os.system('mkdir '+root)
if root+'/tmp' not in folders: os.system('mkdir '+root+'/tmp')
if root+'/experiments'not in folders: os.system('mkdir '+root+'/experiments')
if root+'/experiments/config'not in folders: os.system('mkdir '+root+'experimentss/config')

#================================Generate Random Models================================#
args = {
        'data_repeats':5,
        'save_summary_steps':5,
        'num_parallel_calls':8,
        'n_classes':2,
        'epochs':30,
        'loss_type':['dice','crossentropy'],
        'optimizer':['adam','sgd'],
        'momentum':[0.9,0.99],
        'learning_rate':(-7,-2),
        'train_batch_size':5,
        'val_test_batch_size':5,
        'img_size':[256,512],
        'alpha':(-30,-1),
        'lr_schedule_type':['step'],
        'reg_rate':(-20,1),
        'imgs_dir':'dataset/imgs',
        'masks_dir':'dataset/masks'
        }

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

