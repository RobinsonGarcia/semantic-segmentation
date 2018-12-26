import os
import sys
import shutil
import numpy as np
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--orig',help='Origin folder, containing the files to be split into train, val, test.')
parser.add_argument('--dest',help='Destination folder, files from Origin will be split into train, val, test.')
parser.add_argument('--test_cut',help='Percentage to be allocated to the test set. eg 30%',type=float,default=0.3)
parser.add_argument('--val_cut',help='Percentage to be allocated to the validation set. eg 30%',type=float,default=0.3)


if __name__=="__main__":
    args = vars(parser.parse_args())

    if args['test_cut']==None:
        logging.warning('30% of the entire set will be allocated to the test folder (default value used)')
    if args['val_cut']==None:
        logging.warning('30% of the train set (already discounted the test percentage) will be allocated to the val folder (default value used)')


    origins = os.listdir(args['orig'])

    orig = os.path.join(args['orig'],origins[0])

    train_cut = 1 - args['test_cut']
    val_cut = 1 - args['val_cut']

    files = os.listdir(orig)

    num_files = len(files)
    logging.info('A total of %d where retrieved from %s'%(num_files,orig))


    #===> SHUFFLE
    idx = np.arange(num_files)
    idx = np.random.permutation(idx)

    train_split = int(train_cut*num_files)

    train_set_idx = idx[:train_split]
    test_set_idx = idx[train_split:]

    val_split = int(len(train_set_idx)*val_cut)
    val_set_idx = train_set_idx[val_split:]
    train_set_idx = train_set_idx[:val_split]

    logging.info('Final disribution - Train set: %d, Test set: %d, Val set: %d'%(len(train_set_idx),len(val_set_idx),len(test_set_idx)))


    dest = args['dest']
    try: 
        os.mkdir(dest)
    except:
        logging.info('%s folder already exists'%(dest))


    for og in origins:
        orig = args['orig']+'/'+og
        dest = args['dest']+'/'+og

        try: 
            os.mkdir(dest)
        except:
            logging.info('%s folder already exists'%(dest))

        try:
            os.mkdir(os.path.join(dest,'train'))
        except:
            logging.info('%s folder already exists'%(dest+'/train'))
        
        for i in train_set_idx:
            o = os.path.join(orig,files[i])
            d = os.path.join(dest,'train',files[i])
            shutil.copy(o,d)
            logging.info('Copied %s to %s'%(o,d))

        try:
            os.mkdir(os.path.join(dest,'val'))
        except:
            logging.info('%s folder already exists'%(dest+'/val'))
        
        for i in val_set_idx:
            o = os.path.join(orig,files[i])
            d = os.path.join(dest,'val',files[i])
            shutil.copy(o,d)
            logging.info('Copied %s to %s'%(o,d))

        try:
            os.mkdir(os.path.join(dest,'test'))
        except:
            logging.info('%s folder already exists'%(dest+'/test'))
           
        for i in test_set_idx:
            o = os.path.join(orig,files[i])
            d = os.path.join(dest,'test',files[i])
            shutil.copy(o,d)
            logging.info('Copied %s to %s'%(o,d))
