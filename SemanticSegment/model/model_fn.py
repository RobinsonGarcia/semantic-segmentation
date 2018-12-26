
# coding: utf-8

# In[ ]:

import os
import math
import argparse
import numpy as np
import tensorflow as tf
import json
import sys
import pprint
from PIL import Image
import logging
#logging.basicConfig(level=logging.DEBUG)
#===================================== HELPER FUNCTIONS ======================================================#
def lr_schedule(e,i,l_rate0,k,type_='step'):
    if type_=='exp':
        return l_rate0*math.e**(-i*k)
    if type_=='step':
        return l_rate0*k**e

#====================================== LAYERS/BLOCKS =========================================================#

#==> LAYERS
def relu(x,n): return tf.nn.leaky_relu(x,alpha=lalpha,name=n)

def maxpool(x,n): return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='max'+str(n))

def conv11(x,din,dout):
    w0 = tf.get_variable(name="conv11",shape=[1,1,din,dout],initializer=tf.contrib.layers.xavier_initializer())
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,w0)
    b0 = tf.get_variable("b0", dout,initializer=tf.constant_initializer(0.0))
    return tf.nn.conv2d(x,w0,strides=[1,1,1,1],padding='SAME')+b0

#==> BLOCKS
class conv_block:
    def __init__(self,ksize,din,dout,n):
        self.n = n
        with tf.variable_scope("conv_block"+str(n)):
            self.w0 = tf.get_variable(name="conv1",shape=[ksize,ksize,din,dout],initializer=tf.contrib.layers.xavier_initializer())
            self.b0 = tf.get_variable("b0", dout,initializer=tf.constant_initializer(0.0))
            self.w1 = tf.get_variable(name="conv2",shape=[ksize,ksize,dout,dout],initializer=tf.contrib.layers.xavier_initializer())
            self.b1 = tf.get_variable("b1", dout,initializer=tf.constant_initializer(0.0))
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,self.w0)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,self.w1)

            if save_hist:
                tf.summary.histogram('w_conv1',self.w0)
                tf.summary.histogram('w_conv2',self.w1)

    def forward(self,x,stride=1,dilate=1):
        self.x0 = x
        self.x1 = tf.nn.conv2d(self.x0,self.w0,strides=[1,stride,stride,1],padding='SAME')
        self.x1+=self.b0
        with tf.name_scope("conv_block"+str(self.n)):
            self.x2 = relu(self.x1,'relu'+str(self.n))
        
        self.x3 = tf.nn.conv2d(self.x2,self.w1,strides=[1,stride,stride,1],padding='SAME')
        self.x3+=self.b1
        with tf.name_scope("conv_block"+str(self.n)):
            self.x4 = relu(self.x3,'relu'+str(self.n))

        return self.x4

class upconv:
    def __init__(self,ksize,din,dout,n):
        self.n=n
        with tf.variable_scope("upconv"+str(n)):
            self.w0 =  tf.get_variable(name="upconv1",shape=[ksize,ksize,dout,din],initializer=tf.contrib.layers.xavier_initializer())
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,self.w0)
            if save_hist:
                tf.histogram('w_conv1',self.w0)

    def crop_concat(self,x0,x):
        with tf.name_scope('crop_concat'+str(self.n)):
            shape0 = tf.shape(x0)
            shape1 = tf.shape(x)
            offsets = [0,(shape0[1]-shape1[1])//2,(shape0[2]-shape1[2])//2,0]
            size = [-1,shape1[1],shape1[2],-1]
            x0 =  tf.slice(x0,offsets,size)
            return tf.concat([x0,x],axis=3)

    def forward(self,x0,x,stride=1,dilate=1):
        shape = x.shape
        self.x = x
        self.x =tf.nn.conv2d_transpose(self.x,self.w0,output_shape=[bsize]+
                                       [int(shape[1].value*2),
                              int(shape[2].value*2),int(shape[3].value/2)],
                                      strides=[1,2,2,1],padding='SAME')
        with tf.name_scope("upconv"+str(self.n)):
            self.x = relu(self.x,'relu'+str(self.n))
        self.x = self.crop_concat(x0,self.x)
        return self.x


#=Model Loss
def dice_loss(y_pred,y_target):
    num = 2*tf.reduce_sum(y_pred*y_target,axis=0)
    den = tf.reduce_sum(tf.add(y_pred**2,y_target**2),axis=0) + 1e-6
    return 1-tf.reduce_mean(num/den)

def build_model(is_training,inputs):
    x0 = inputs['images']
    
    with tf.variable_scope('block1'):
        conv_block1 = conv_block(3,3,64,1)
    with tf.variable_scope('block2'):
        conv_block2 = conv_block(3,64,128,2)
    with tf.variable_scope('block3'):
        conv_block3 = conv_block(3,128,256,3)
    with tf.variable_scope('block4'):
        conv_block4 = conv_block(3,256,512,4)
    with tf.variable_scope('block5'):
        conv_block5 = conv_block(3,512,1024,5)

    with tf.variable_scope('up_block6'):
        upconv1 = upconv(2,1024,512,6)
        conv_block6 = conv_block(3,1024,512,7)
    with tf.variable_scope('up_block7'):
        upconv2 = upconv(2,512,256,8)
        conv_block7 = conv_block(3,512,256,9)
    with tf.variable_scope('up_block8'):
        upconv3 = upconv(2,256,128,10)
        conv_block8 = conv_block(3,256,128,11)
    with tf.variable_scope('up_block9'):
        upconv4 = upconv(2,128,64,12)
        conv_block9 = conv_block(3,128,64,13)

    with tf.variable_scope('block1'):
        h1 = x = conv_block1.forward(x0)
        x = maxpool(x,1)
    with tf.variable_scope('block2'):
        h2 = x = conv_block2.forward(x)
        x = maxpool(x,2)
    with tf.variable_scope('block3'):
        h3 = x = conv_block3.forward(x)
        x = maxpool(x,3)
    with tf.variable_scope('block4'):
        h4 = x = conv_block4.forward(x)
        x = maxpool(x,4)
    with tf.variable_scope('block5'):
        x = conv_block5.forward(x)

    with tf.variable_scope('up_block6'):
        x = upconv1.forward(h4,x)
        x = conv_block6.forward(x)
    with tf.variable_scope('up_block7'):
        x = upconv2.forward(h3,x)
        x = conv_block7.forward(x)
    with tf.variable_scope('up_block8'):
        x = upconv3.forward(h2,x)
        x = conv_block8.forward(x)
    with tf.variable_scope('up_block9'):
        x = upconv4.forward(h1,x)
        x = conv_block9.forward(x)
    with tf.variable_scope('fcn'):
        x = conv11(x,64,n_classes)

    with tf.variable_scope('scores'):
        scores = tf.reshape(tensor=x, shape=(-1, n_classes),name='logits')
    
    return scores,x


def model_fn(mode,inputs,params,reuse=False):
    #==============================Build Graph========================================#

    model_specs = inputs
   
    is_training = (mode == 'train')

    global size
    size = params['img_size']

    global lalpha
    lalpha = params['alpha']

    global n_classes
    n_classes = params['n_classes']

    global save_hist
    try:
        save_hist = params['save_hist']
    except:
        save_hist = False
        logging.warning('Save histograms set to False (add save_hist:True to params otherwise')

    #=== Input batch
    #x0 = tf.placeholder(tf.float32,shape=(None,size,size,3),name='x0')

    #=== Parameters
    #= learning rate
    lr = tf.placeholder(tf.float32,shape=(),name="learning_rate")

    #==> Image data iterator
    x0 = inputs['images']
        
    #= batch size
    global bsize
    bsize = tf.shape(x0)[0]

    #==> BUILD THE MODEL
    with tf.variable_scope('model',reuse=reuse):
        scores,out_img = build_model(is_training,inputs)
        predictions = tf.argmax(scores,1)

    out_img = tf.expand_dims(tf.argmax(out_img,axis=3),axis=3)
    model_specs['output'] = (inputs['images'],out_img)

    ''' 
    with tf.variable_scope('summary'):
        #for ix in range(params['batch_size']):
        image = tf.argmax(x,axis=3,name='mask_image')
        image = tf.expand_dims(image,axis=3)
        get_image = tf.summary.image('masks_itraining_'+str(is_training),tf.cast(image,tf.float32),max_outputs = 100)
    '''
    
    #= Mask/y
    with tf.variable_scope('masks'):
         #=== Mask data iterator
        #y = tf.placeholder(tf.float32,shape=(None,size,size,1),name='y')
        y = inputs['labels']
        mask_ = tf.concat(values=[y,-y+1],axis=3)
        mask  = tf.reshape(tensor=mask_, shape=(-1, n_classes))
 
    labels = tf.argmax(mask,1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels,predictions),tf.float32))
 
    #= Regularization
    with tf.variable_scope('regularization'):
        lamb = params['reg_rate']
        regularizer = tf.contrib.layers.l2_regularizer(scale=lamb)
        reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg = tf.contrib.layers.apply_regularization(regularizer,reg_var)


    with tf.variable_scope('loss'):
        if params['loss_type']=='dice':
            loss_ = dice_loss(tf.nn.softmax(scores),mask)+reg
        else:
            loss_ = tf.nn.softmax_cross_entropy_with_logits_v2(logits=scores,labels=mask)
        loss = tf.reduce_mean(loss_)+reg

    
    
    
    #==> OPTIMIZATION (is_training)
    if is_training:
        #= Optimization
        with tf.variable_scope('Optimizer'):
            global_step = tf.train.get_or_create_global_step()
            if params['optimizer']=='adam':
                opt = tf.train.AdamOptimizer(learning_rate=lr)
            else:
                opt = tf.train.MomentumOptimizer(learning_rate=lr,momentum=params['momentum'])
            train_op = opt.minimize(loss,global_step=global_step)
    else:
        with tf.variable_scope('Optimizer'):
            train_op = None

    #==> METRICS
    with tf.variable_scope("metrics"):
        metrics = {
                'accuracy':tf.metrics.accuracy(labels=labels,predictions=predictions),'loss':tf.metrics.mean(loss_)}

    update_metrics_op = tf.group(*[op for op in metrics.values()])
        
    metrics_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metrics_variables)

   
    with tf.variable_scope('summary'):
        tf.summary.scalar('loss',loss)
        tf.summary.scalar('accuracy',accuracy)
        #tf.summary.image('train_image',inputs['images'])
    
   


    #======================= Initialize Variables=============#
    model_specs['global_var_init'] = tf.global_variables_initializer()
    model_specs['local_var_init'] = tf.local_variables_initializer()
    model_specs['train_op'] = train_op
    model_specs['loss'] = loss
    model_specs['logits'] = scores
    model_specs['accuracy'] = accuracy
    model_specs['summary_op'] = tf.summary.merge_all()
    model_specs['lr']=lr
    model_specs['metrics'] = metrics
    model_specs['update_metrics'] = update_metrics_op
    model_specs['metrics_init_op'] = metrics_init_op
    #model_specs['get_image'] = get_image
    return model_specs 


