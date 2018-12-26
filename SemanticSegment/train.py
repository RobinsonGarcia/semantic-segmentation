import sys
import os
import tensorflow as tf

from model.input_fn import input_fn
from model.model_fn import model_fn
from model.training import train_and_evaluate
from model.utils import set_logger

import json
import logging

if __name__=="__main__":
    
    tf.set_random_seed(230)

    param_path = sys.argv[1]

    model_dir = sys.argv[2]

    try:
        restore_from = sys.argv[3]
    except:
        restore_from = None

    set_logger(os.path.join(model_dir,'train.log'))

    logging.info("reading parameters from:  {}".format(param_path))

    params = json.loads(open(param_path,'r').read())

    train_inputs = input_fn(True,params['imgs_dir']+'/train',params['masks_dir']+'/train',params)

    val_inputs = input_fn(False,params['imgs_dir']+'/val',params['masks_dir']+'/val',params)

    train_model_spec = model_fn(mode='train',inputs=train_inputs,params=params)

    val_model_spec = model_fn(mode='eval',inputs=val_inputs,params=params,reuse=True)

    train_and_evaluate(train_model_spec,val_model_spec,model_dir,params,restore_from)

