

# 🔬 WSSS-01: Weakly Supervised Segmentation Framework

## 📁 1. Using External Libraries

To use an external repository (e.g., `taming-transformers`) as a submodule:

### ➕ Add Submodule
```bash
git submodule add https://github.com/CompVis/taming-transformers.git src/externals/taming-transformers
```
This command adds the repository under src/externals/taming-transformers.


### 🔧 Update .gitmodules
The .gitmodules file will automatically be updated with: 
```bash 
[submodule "src/externals/taming-transformers"]
    path = src/externals/taming-transformers
    url = https://github.com/CompVis/taming-transformers.git
```
### Commit Changes 

```bash 
git add .gitmodules src/externals/taming-transformers
git commit -m "Add taming-transformers as a submodule under src/externals/"
git push origin main
```
### 📥 Clone & Initialize Submodules (for others) 
After cloning the repo, other users must run:

```bash 
git submodule update --init --recursive
```
 
## ⚙️ 2. Configuration Files
`configs/datasetname.yaml`: Dataset-specific settings (e.g., file paths, image size).
`configs/model_datasetname.yaml`: Combined model and dataset config (e.g., architecture, training parameters). 

## 🛠 3. Makefile Targets 
### a. Data processing: For example we want to run VQGAN on LUAD and save the result: 
```bash 
quantizing_luad:
	python main_quantizing.py --configs configs/luad.yaml
``` 
```bash 
make quantizing_luad 
``` 
### b. Training
Train a model (e.g., SillyModel) on LUAD: 
📌 Note:
k_start=1, k_end=1 → run only fold 1
k_start=1, k_end=5 → run  fold 1, 2, 3, 4, 5 
You can adjust --epoch to control the number of training epochs
```bash 
sillymodel_luad:
	python main_train_silly_model.py --configs configs/sillymodel_luad.yaml --k_start 0 --k_end 1 --epoch 100
``` 
```bash 
make sillymodel_luad
``` 
 
### b. Test
To evaluate a trained model checkpoint:

```bash 
test_sillymodel_luad:
	python main_test_silly_model.py \
		--configs configs/sillymodel_luad.yaml \
		--k_start 0 --k_end 1 \
		--checkpoint results/sillymodel_luad/fold_0/best_model.pt 
``` 
```bash 
make test_sillymodel_luad
```  

### Repo Structure: 
```
.
├── src/
│   ├── externals/
│   │   └── taming-transformers/     # Submodule
│   └── models/                      # Your model definitions
├── configs/
│   ├── luad.yaml                    # Dataset-only config
│   ├── sillymodel_luad.yaml         # Model + dataset config
├── main_quantizing.py               # For feature quantization
├── main_train_silly_model.py       # Training script
├── main_test_silly_model.py        # Evaluation script
└── Makefile                         # Shortcuts for running tasks
 
```