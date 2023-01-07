# MBIEP

MBIEP is a deep learning framework for identifying essential proteins by integrating Multiple Biological Information, namely subcellular localization, gene expression, and PPI network.This framework is published in BMC Bioinformatics.
## CITE US:
Yi Yue,Chen Ye,Pei‑Yun Peng,Hui‑Xin Zhai,Iftikhar Ahmad,Chuan Xia,Yun‑Zhi Wu and You‑Hua Zhang.A deep learning framework for identifying essential proteins based on multiple biological information.BMC Bioinformatics (2022) 23:318.https://doi.org/10.1186/s12859-022-04868-8

# Requirement

-   Python-3.8.0
-   Tensorflow-2.4.0
-   Numpy-1.19.4
-   Networkx-2.4
-   scikit-learn-0.23.1

# Usage
## Source and preprocessing
1.  `PPI network` is downloaded from BioGRID (https://thebiogrid.org/), we use the physical interactions only, remove the self-loops,  and duplicate edges.
2. `Subcellular localization` is downloaded from the integrated channel of COMPARTMENTS (https://compartments.jensenlab.org/Search), then the data is normalized.
3. `Gene expression` is downloaded from the GEO database (Accession number: GSE7645), then the data is normalized.
4. `Essential proteins`  are downloaded from the OGEE database (https://v3.ogee.info) and DEG database (http://origin.tubic.org/deg/public/index.php), we removed duplicate records from two datasets to obtain the union set.

## Dataset
We provide a zip file of the dataset, which includes the training set, validation set, and test set, divided in the ratio of 6:2:2, respectively. Among them：
1. `_emb.npy` file represents the features extracted from the PPI network, we use the node2vec ([https://github.com/aditya-grover/node2vec](https://github.com/aditya-grover/node2vec)) technique to obtain feature representations from the PPI network. Parameters are set as follows:

```python
python main.py --input Gnodes.txt --output _emb.txt --dimensions 64 --walk-length 20 --num-walks 10 --window-size 10
```

2. `_gse.npy` file represents features extracted from gene expression profiles. According to the time step, the control group and the replication samples, we reshape it as (5988, 8, 3, 2).
3. `_sub.npy` file represents features extracted from subcellular localization.
4. `_label.npy` file represents the label of the protein, where 0 represents the non-essential protein and 1 represents the essential protein.

## Model parameter

In the file `model.h5`, we also provide the parameters of the model obtained from training, which you can use to predict essential proteins, or to test our dataset.

## License

This project is licensed under the MIT License.
