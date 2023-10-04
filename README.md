[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

# Explainable Multilayer Graph Neural Network for Cancer Gene Prediction
 
The preprint paper associated with this work can be accessed at the following link: https://arxiv.org/abs/2301.08831

This repository contains the scripts for training and explaining the predictions of the EMGNN model.

![EMGNN Architecture](Fig1-1.png)

## Requirements
- Python 3
- PyTorch
- torch-scatter, torch-sparse, torch-cluster, torch-spline-conv, torch-geometric
- networkx
- captum
- pandas
- sklearn

## How to Run


### Data Preperation 

To have a fair comparison with EMOGI, we used the same data preprocessing as their official implementation https://github.com/schulter/EMOGI. Follow the instruction to download the PPI networks and the labels.

### Training

To train the model, run the following command:

    python train.py --gcn 1 

This will train the EMGNN model using GCN as the graph neural network with the default settings and the datasets specified in the script. You can also specify different settings and datasets by passing in command-line arguments. For example, to train the model using the GAT architecture on the IREF_2015, PCNET and STRING PPI networks, and test it on STRING you can run:

    python train.py --gat 1 --dataset IREF_2015 PCNET STRING

Notice that the last PPI network will always be used as the test set.
The script also includes additional functionalities such as loading a pretrained model, adding random features, adding identical features, adding structural noise, and running a multi-layer perceptron (MLP) as a baseline instead of the EMGNN. The functionality of these options can be found in the code.

### Explaining Predictions of EMGNN

The following script allows you to explain predictions made by an EMGNN (Edge-enhanced Meta Graph Neural Network) model for gene prediction. It uses the Captum library to perform gradient-based attribution methods, focusing on the Integrated Gradients method. You can use this script to gain insights into why the model made specific predictions for cancer genes.

    python explain.py --model_dir <path_to_trained_model> --gene_label <gene_type>

Replace <path_to_trained_model> with the path to your trained EMGNN model and <gene_type> with one of the following options:

- cancer: To explain cancer genes.

- non-cancer: To explain non-cancer genes.

- top_predicted: To explain the top predicted genes.

You can also use the --visualize flag to save network explanation visualizations if desired.

Note: Make sure the path to the trained model is the correct path.

The script generates explanation outputs for both edge and node explainability. The explanations are saved as pickle files in the explain directory within your model directory.

- Edge Explainability: Edge attributions are saved as ``` edge_mask_explain_<idx>_<label>.pkl ```.
- Node Explainability: Node feature attributions are saved as ``` node_feat_mask_explain_<idx>_<label>.pkl ```.

For a comprehensive explanation of the results and detailed code to generate the explainability plots of the paper, please refer to the `analysis.ipynb` Jupyter Notebook in this repository. 
The notebook provides step-by-step instructions and code snippets to perform an in-depth analysis of the EMGNN model's predictions and explanations.

### Predictions for Unlabelled Genes

We provide the predictions for the unlabelled genes of our EMGNN model in the following [link](https://michailchatzianastasis.github.io/csv_to_html/).

