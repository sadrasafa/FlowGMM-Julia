# Comp541-Project

This repository contains the Julia implementation of the [Semi-Supervised Learning with Normalizing Flows](https://arxiv.org/abs/1912.13025) paper.

(PyTorch implementation is [here](https://github.com/izmailovpavel/flowgmm/))

# Downloading the Datasets
Downlaod the files in the following Google Drive links:


[Toy Datasets](https://drive.google.com/open?id=10ykNO7XgYA9B1PqVLjq7_9XSOEIbpSoz)

[UCI Datasets](https://drive.google.com/open?id=1-FLjRxw7uAeA0H-_kT-d10ciEtOXzfth)
(Original UCI datasets: [MiniBooNE](http://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification) and [HEPMASS](http://archive.ics.uci.edu/ml/datasets/HEPMASS). Also the preprocessing from [Masked Autoregressive Flow for Density Estimation](https://github.com/gpapamak/maf) has been used where sensible )

[NLP Datasets](https://drive.google.com/open?id=113qI9K3MESs528M4rpM3aKWsdqbg8r-x) 
(This data has been obtained using [this script](https://github.com/izmailovpavel/flowgmm/blob/public/data/nlp_datasets/get_text_classification_data.sh) from the original PyTorch repository. BERT Embeddings of the data has been computed afterwards.)

After downloading the datasets, make three directories named `toy_datasets`, `uci_datasets`,and `nlp_datasets` and move the downloaded datasets to the appropriate folders without any subdirectories.

# Results
The experiments are implemented in [this notebook](https://github.com/sadrasafa/Comp541-Project/blob/master/FlowGMM.ipynb).
