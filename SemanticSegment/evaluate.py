"""Evaluate the model"""

import argparse
import logging
import os

import tensorflow as tf

import json
from model.input_fn import input_fn
from model.model_fn import model_fn
from model.evaluation import evaluate
from model.utils import Params
from model.utils import set_logger


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', help='Model dir eg: Shiriu or Seya')
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--model_number', default='0000',
        help="Model number eg: 0001")


if __name__ == '__main__':
    # Set the random seed for the whole graph
    tf.set_random_seed(230)

    # Load the parameters
    args = parser.parse_args()

    #=> define model_number
    path2param = 'params/params'+args.model_number+'.json'
    json_path = os.path.join(args.model_dir,'experiments',path2param)
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = json.loads(open(json_path,'r').read())

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")


    #==>DEFINE: img_dir, mask_dir
    img_dir = os.path.join(args.data_dir,'imgs/test')
    mask_dir = os.path.join(args.data_dir,'masks/test')

    # create the iterator over the dataset
    test_inputs = input_fn(False, img_dir,mask_dir,params)

    # Define the model
    logging.info("Creating the model...")
    model_spec = model_fn('eval', test_inputs, params, reuse=False)
    

    restore_from = os.path.join('experiments/experiment'+args.model_number,'best_weights')

    logging.info("Starting evaluation")
    evaluate(model_spec, args.model_dir, params, restore_from)
