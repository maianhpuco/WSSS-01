# WSSS-01

1. Use External library: 
Submodule: 
For example: I want to run the code from taming-transformer: 
```bash 
git submodule add https://github.com/CompVis/taming-transformers.git src/externals/taming-transformers
``` 
Check that the following are updated: .gitmodules will be created or updated with: 
```bash 
[submodule "src/externals/taming-transformers"]
    path = src/externals/taming-transformers
    url = https://github.com/CompVis/taming-transformers.git
``` 
Then add and comnit as normal:
```bash 
git add .gitmodules src/externals/taming-transformers
git commit -m "Add taming-transformers as a submodule under src/externals/"
git push origin main 
``` 
Then other people will also have the code update with that submodule and can use them after running: 
```git submodule update --init --recursive``` 
2. File Structure 
a. Make file: 
- main + config 

b. Make file for train: 
- main + config + k_start + k_end + epoch

c. make file for test: 
- check point defined as variable ($)
- main + config + k_start + k_end + checkpoint 

--> adding example for the command and also for the file (main)
3. configs:
For dataset: (in wsi-seg-data repo)

For model + dataset: 

