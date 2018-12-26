import logging
import os
from model.utils import save_dict_to_json
from tqdm import trange
import tensorflow as tf

def get_masks_t(sess,model_spec,num_steps,writer=None,params=None,epoch=None):
    output = model_spec['output']
    sess.run(model_spec['iterator_init_op'])

    imgs = []
    for _ in range(num_steps):
        output_pairs = sess.run(output)
        imgs.append(output_pairs)
    return imgs


def evaluate_sess(sess,model_spec,num_steps,writer=None,params=None):
    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']
    global_step = tf.train.get_global_step()
    summ_op = model_spec['summary_op']
    #==> Initialize dataset and metrics
    sess.run(model_spec['iterator_init_op'])
    sess.run(model_spec['metrics_init_op'])


    #==> Update metrics over the eval dataset
    for _ in range(num_steps):
       sess.run(update_metrics)

    metrics_values = {k: v[0] for k,v in eval_metrics.items()}
    metrics_val = sess.run(metrics_values)
    metrics_string = " ; ".join("{}: {:05.3f}".format(k,v) for k,v in metrics_val.items())
    logging.info("- Eval metrics: "+metrics_string)

    if writer is not None:
        global_step_val = sess.run(global_step)
        for tag,val in metrics_val.items():
            summ = tf.Summary(value=[tf.Summary.Value(tag=tag,simple_value=val)])
            writer.add_summary(summ,global_step_val)
    return metrics_val


def evaluate(model_spec,model_dir,params,restore_from): 
    saver = tf.train.Saver()

    with tf.Session() as sess:
        
        sess.run([model_spec['global_var_init'],model_spec['local_var_init']])

        save_path = os.path.join(model_dir, restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess,save_path)

        num_steps = model_spec['n_iters']
        metrics = evaluate_sess(sess,model_spec,num_steps)
        metrics_name = '_'.join(restore_from.split('/'))
        save_path = os.path.join(model_dir,"metrics_test_{}.json".format(metrics_name))
        save_dict_to_json(metrics,save_path)

        




