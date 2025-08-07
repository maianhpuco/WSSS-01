SILLYMODEL_CKPT=path/to/checkpoint_folder/s_1_checkpoint.pt 
# For dataset only 
quantizing_luad: 
	main_quantizing.py --configs configs/quantizing_luad.yaml


# model + dataset: training
train_sillymodel_luad:
	main_train_silly_model.py --configs configs/sillymodel_luad.yaml --k_start 0 --k_end 1 --epoch 100 
# model + dataset: testing 
test_sillymodel_luad:
	main_train_silly_model.py --configs configs/sillymodel_luad.yaml --check_point $(SILLYMODEL_CKPT) 