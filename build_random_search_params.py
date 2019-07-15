import pickle

  


args = {
'data_repeats':3,
'save_summary_steps':5,
"keep_prob":0.5,
'num_parallel_calls':8,
'n_classes':2,
'epochs':70,
'loss_type':'crossentropy',
'optimizer':'adam',
'momentum':0.9,
'learning_rate':(-7,-3),
'train_batch_size':1,
'val_test_batch_size':1,
'img_size':512,
'alpha':(-30,-1),
'lr_schedule_type':['step'],
'reg_rate':(-20,1),
'imgs_dir':'/home/u-net/dataset/images',
'masks_dir':'/home/u-net/dataset/masks'
}



f = open('random_search_params.pickle','wb')
pickle.dump(args,f)
f.close()
