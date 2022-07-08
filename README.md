# Stable Tuple Embeddings for Dynamic Databases

#### Installation:
```
conda create --name env python=3.8
conda activate env
conda install pytorch=1.7.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
```
Note that we use cuda 10.2 by default. 


#### Experiments
We provide scripts to reproduce our experiemnts with FoRWaRD and Node2Vec, respectively:
```
bash run_forward_experiments.sh
bash run_n2v_experiments.sh
```

