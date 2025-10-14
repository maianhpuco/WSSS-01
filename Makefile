PYTHON ?= python3

SILLYMODEL_CKPT=path/to/checkpoint_folder/s_1_checkpoint.pt 
PBIPMODEL_CKPT=./model_weight/PBIP.pth


# For dataset only 
quantizing_luad: 
	main_quantizing.py --configs configs/quantizing_luad.yaml


# model + dataset: trainig
train_sillymodel_luad:
	main_train_silly_model.py --configs configs/sillymodel_luad.yaml --k_start 0 --k_end 1 --epoch 100 
# model + dataset: testing 
test_sillymodel_luad:
	main_train_silly_model.py --configs configs/sillymodel_luad.yaml --check_point $(SILLYMODEL_CKPT) 

train_pbip_bcss:
	$(PYTHON) main_train_pbip.py --config configs/pbip_bcss.yaml --gpu 0

test_pbip_bcss:
	$(PYTHON) main_test_pbip.py --config configs/pbip_bcss.yaml --check_point $(PBIPMODEL_CKPT)

train_tpro_bcss_wsss:
	$(PYTHON) main_train_tpro.py --config configs_maui/tpro_bcss_wsss.yaml 

train_tpro_bcss_wsss_cls:
	$(PYTHON) main_train_tpro_cls.py --config configs_maui/tpro_bcss_wsss.yaml  
train_tpro_bcss_wsss_pseudo: 
	$(PYTHON) main_train_tpro_pseudo.py --config configs_maui/tpro_bcss_wsss.yaml  

train_tpro_bcss_wsss_seg:
	$(PYTHON) main_train_tpro_seg.py --config configs_maui/tpro_bcss_wsss.yaml
