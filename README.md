## k-hop Graph Neural Networks
Code for the paper [k-hop Graph Neural Networks](https://arxiv.org/pdf/1907.06051.pdf).

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
@article{nikolentzos2019k,
  title={k-hop Graph Neural Networks},
  author={Nikolentzos, Giannis and Dasoulas, George and Vazirgiannis, Michalis},
  journal={arXiv preprint arXiv:1907.06051},
  year={2019}
}

```

-----------

Provided for academic use only