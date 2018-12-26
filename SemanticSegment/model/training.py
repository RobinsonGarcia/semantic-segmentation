
# coding: utf-8

# In[ ]:
import pickle
import os
import math
import argparse
import numpy as np
import tensorflow as tf
import json
import sys
import pprint
from PIL import Image
from tqdm import trange

from model.evaluation import evaluate_sess,get_masks_t
from model.utils import save_dict_to_json
import logging
#logging.basicConfig(level=logging.DEBUG)

def train_sess(sess,model_specs,num_steps,writer,params):
    
    loss = model_specs['loss']
    train_op = model_specs['train_op']
    update_metrics = model_specs['update_metrics']
    metrics = model_specs['metrics']
    summary_op = model_specs['summary_op']
    lr = model_specs['lr']
    global_step = tf.train.get_global_step()
    
    sess.run(model_specs['iterator_init_op'])
    sess.run(model_specs['metrics_init_op'])
   

    t = trange(num_steps-1)

    for i in t:
        if i%params['save_summary_steps'] == 0:
            _,_,loss_val,summ,global_step_val = sess.run([train_op,update_metrics,loss,summary_op,global_step],feed_dict={lr:params['learning_rate']})
            writer.add_summary(summ,global_step_val)
        else:
            _,_,loss_val = sess.run([train_op,update_metrics,loss],feed_dict={lr:params['learning_rate']})
        t.set_postfix(loss='{:05.3f}'.format(loss_val))

    
    
    metrics_values = {k:v[0] for k, v in metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k,v in metrics_val.items())
    logging.info('- Train metrics: '+metrics_string)
    

def train_and_evaluate(train_model_spec,eval_model_spec,model_dir,params,restore_from=None):
    best_saver = tf.train.Saver(max_to_keep=1)
    last_saver = tf.train.Saver()
    img_pairs = {}

    begin_at_epoch = 0
    with tf.Session() as sess:
        sess.run([train_model_spec['global_var_init'],train_model_spec['local_var_init']])

        #===> restore model

        if restore_from is not None:
            logging.info('Restoring parameter from {}'.format(restore_from))
            if os.path.isdir(restore_from):
                restore_from = tf.train.latest_checkpoint(restore_from)
                begin_at_epoch = int(restore_from.split('-')[-1])
            last_saver.restore(sess, restore_from)

        #==> writers
        train_writer = tf.summary.FileWriter(os.path.join(model_dir,'train_summaries'),sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(model_dir,'eval_summaries'),sess.graph)
        learning_evolu_writer = tf.summary.FileWriter(os.path.join(model_dir,'learning_evolu_summaries'),sess.graph)


        best_eval_acc = 0.0
        for epoch in range(begin_at_epoch,begin_at_epoch+params['epochs']):
            logging.info("Epoch {}/{}".format(epoch+1,params['epochs']))
            train_sess(sess,train_model_spec,train_model_spec['n_iters'],train_writer,params)
        
            last_save_path = os.path.join(model_dir,'last_weights','after-epoch')
            last_saver.save(sess,last_save_path,global_step=epoch+1)

            metrics = evaluate_sess(sess,eval_model_spec,eval_model_spec['n_iters'],eval_writer)


            eval_acc = metrics['accuracy']
            if eval_acc>best_eval_acc:
                best_eval_acc = eval_acc
                best_save_path = os.path.join(model_dir,'best_weights','after-epoch')
                best_save_path = best_saver.save(sess,best_save_path,global_step=epoch+1)
                logging.info('- Found new best accuracy, saving in {}'.format(best_save_path))
                best_json_path = os.path.join(model_dir,"metrics_eval_best_weights.json")
                save_dict_to_json(metrics,best_json_path)

            last_json_path = os.path.join(model_dir,"metrics_eval_last_weights.json")
            save_dict_to_json(metrics,last_json_path)
                
            img_pairs[epoch]=get_masks_t(sess,eval_model_spec,eval_model_spec['n_iters'],learning_evolu_writer,epoch)
        with open(model_dir+'/img_pairs.pickle','wb') as f:
            pickle.dump(img_pairs,f)



