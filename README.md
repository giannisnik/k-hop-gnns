## k-hop Graph Neural Networks
Code for the paper [k-hop Graph Neural Networks](https://www.sciencedirect.com/science/article/abs/pii/S0893608020302495?dgcid=rss_sd_all).

### Requirements
Code is written in Python 3.6 and requires:
* PyTorch 1.1
* NetworkX 2
* scikit-learn 0.21

### Datasets
Use the following link to download datasets: 
```
https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
```
Extract the datasets into the `datasets` folder.

### Run the model
First, specify the dataset and the hyperparameters in the `main.py` file. Then, use the following command:

```
$ python main.py
```

### Cite
Please cite our paper if you use this code:
```
@article{nikolentzos2020k,
  title={k-hop graph neural networks},
  author={Nikolentzos, Giannis and Dasoulas, George and Vazirgiannis, Michalis},
  journal={Neural Networks},
  volume={130},
  pages={195--205},
  year={2020},
  publisher={Elsevier}
}
```

-----------

Provided for academic use only
