import tensorflow as tf
import os
import math
import numpy as np
#==> TRANSFORMATION FUNCTIONS, USED 4 DATA AUGMENTATION

def shift_img(image,label,interpolation='NEAREST'):
    shift = np.random.randint(0, 1)
    if shift:
        tx = np.random.randn()*5
        ty = np.random.randn()*5
        transforms = [1, 0, -tx, 0, 1, -ty, 0, 0]
        image = tf.contrib.image.transform(image, transforms, interpolation)
        label = tf.contrib.image.transform(label, transforms, interpolation)
    return image,label
                
def flip(image,label):
    flip = np.random.randint(0, 1)
    if flip:
        seed = np.random.randint(1234)
        image = tf.image.random_flip_left_right(image,seed=seed)
        label = tf.image.random_flip_left_right(label,seed=seed)
    return image,label

def rotate(image,label,max_ang=math.pi*30/180,interpolation='NEAREST'):
    rot = np.random.randint(0, 1)
    if rot:
        angles = max_ang*np.random.rand(1)
        image = tf.contrib.image.rotate(image,angles,interpolation)
        label = tf.contrib.image.rotate(label,angles,interpolation)
        #image = tf.image.rot90(image)
        #label = tf.image.rot90(label)
    return image,label

def crop(image,label):
    crop_img = np.random.randint(0, 1)
    if crop_img:
        f = np.random.randint(2,4)
        H,W,C = image.shape
        seed = np.random.randint(1234)
        image = tf.random_crop(image, size = [int(H//f),int(W//f),int(C)], seed = seed)
        image = tf.image.resize_images(image,[int(H),int(W)])
        label = tf.random_crop(label, size = [int(H//f),int(W//f),1], seed = seed)
        label = tf.image.resize_images(label,[int(H),int(W)])
    return image,label


#==>PARSE FUNCTION: READ FILE, DECODE USING JPEG, CONVERT FLOATS TO [0,1] AND RESIZE

def parse_function(filename,masks,img_size):
    image_string = tf.read_file(filename)
    
    #==> read's the image
    image = tf.image.decode_jpeg(image_string,channels=3)
    
    #==> convert to float values in [0,1]
    image = tf.image.convert_image_dtype(image,tf.float32)

    image = tf.image.resize_images(image,[img_size,img_size])

    mask_string = tf.read_file(masks)
    
    #==> read's the image
    mask = tf.image.decode_jpeg(mask_string,channels=1)
    
    #==> convert to float values in [0,1]
    mask = tf.image.convert_image_dtype(mask,tf.float32)

    mask = tf.image.resize_images(mask,[img_size,img_size])
 
    return image,mask


#===> PREPROCESSING FOR TRAINING/DATA AUGMENTATION
def train_preprocess(image,mask):
    
    image,mask = flip(image,mask)
    image,mask = shift_img(image,mask)
    image,mask = rotate(image,mask)
    image,mask = crop(image,mask)
    bright = np.random.randint(0,1)
    sat = np.random.randint(0,1)
    if bright:
        image = tf.image.random_brightness(image,max_delta=32.0/255.0)
    if sat:
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)

    image = tf.clip_by_value(image,0.0,1.0)
    return image,mask

def roundup(num,den):
    rem = num%den
    if rem>0:
        return num//den+1
    else:
        return num//den

#==> dataset is a node on the tensorflow graph that contains intructions to read file. Need a Session to read it, its a part of the graph thus doesn't requires feeding.
def input_fn(is_training,img_dir,mask_dir,params):
    with tf.device('/cpu:0'):
        files = os.listdir(img_dir)
        masks = os.listdir(mask_dir)
        filenames = []
        masknames = []
        for file_ in files:
            filenames.append(os.path.join(img_dir,file_))
        for mask in masks:
            masknames.append(os.path.join(mask_dir,mask))

        num_samples = len(filenames)
        
        #global img_size
        #img_size = params['img_size']

        parse_fn = lambda f,m:parse_function(f,m,params['img_size'])
        train_fn = lambda f,m:train_preprocess(f,m)

        if is_training:
            dataset = tf.data.Dataset.from_tensor_slices((tf.constant(filenames),tf.constant(masknames)))
            dataset = dataset.repeat(params['data_repeats'])
            dataset = dataset.shuffle(num_samples)
            dataset = dataset.map(parse_fn,num_parallel_calls = params['num_parallel_calls'])
            dataset = dataset.map(train_fn,num_parallel_calls = params['num_parallel_calls'])
            dataset = dataset.batch(params['train_batch_size'])
            dataset = dataset.prefetch(1)
                    
        else:
            dataset =  tf.data.Dataset.from_tensor_slices((tf.constant(filenames),tf.constant(masknames)))
            dataset = dataset.map(parse_fn,num_parallel_calls = params['num_parallel_calls'])
            dataset = dataset.batch(params['val_test_batch_size'])
            dataset = dataset.prefetch(1)
                    
        iterator = dataset.make_initializable_iterator()
        images,masks = iterator.get_next()
        iterator_init_op = iterator.initializer
        
        if is_training:
            num_steps = roundup(num_samples*params['data_repeats'],params['train_batch_size'])
        else:
            num_steps = roundup(num_samples,params['val_test_batch_size'])


        inputs = {'images':images,'labels':masks,'iterator_init_op':iterator_init_op,'n_iters':num_steps}
        
        return inputs
    

