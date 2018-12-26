import pickle


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



f = open('random_search_params.pickle','wb')
pickle.dump(args,f)
f.close()
