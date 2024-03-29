[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

# Explainable Multilayer Graph Neural Network for Cancer Gene Prediction 
<strong>Published at UOC Bioinformatics</strong> [Paper link](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btad643/7325352).

The identification of cancer genes is a critical yet challenging problem in cancer genomics research. Existing
computational methods, including deep graph neural networks, fail to exploit the multilayered gene-gene interactions or
provide limited explanations for their predictions. These methods are restricted to a single biological network, which cannot
capture the full complexity of tumorigenesis. Models trained on different biological networks often yield different and even
opposite cancer gene predictions, hindering their trustworthy adaptation. Here, we introduce an <strong>Explainable Multilayer
Graph Neural Network (EMGNN)</strong> approach to identify cancer genes by leveraging multiple gene-gene interaction networks
and pan-cancer multi-omics data. Unlike conventional graph learning on a single biological network, EMGNN uses a
multilayered graph neural network to learn from multiple biological networks for accurate cancer gene prediction.


![EMGNN Architecture](Fig1-1.png)

## Requirements
- Python 3
- PyTorch
- torch-scatter, torch-sparse, torch-cluster, torch-spline-conv, torch-geometric
- networkx
- captum
- pandas
- sklearn

This repository contains the scripts for training and explaining the predictions of the EMGNN model.

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

### Citation
    @article{10.1093/bioinformatics/btad643,
        author = {Chatzianastasis, Michail and Vazirgiannis, Michalis and Zhang, Zijun},
        title = "{Explainable Multilayer Graph Neural Network for Cancer Gene Prediction}",
        journal = {Bioinformatics},
        pages = {btad643},
        year = {2023},
        month = {10},
        abstract = "{The identification of cancer genes is a critical yet challenging problem in cancer genomics research. Existing computational methods, including deep graph neural networks, fail to exploit the multilayered gene-gene interactions or provide limited explanations for their predictions. These methods are restricted to a single biological network, which cannot capture the full complexity of tumorigenesis. Models trained on different biological networks often yield different and even opposite cancer gene predictions, hindering their trustworthy adaptation. Here, we introduce an Explainable Multilayer Graph Neural Network (EMGNN) approach to identify cancer genes by leveraging multiple gene-gene interaction networks and pan-cancer multi-omics data. Unlike conventional graph learning on a single biological network, EMGNN uses a multilayered graph neural network to learn from multiple biological networks for accurate cancer gene prediction.Our method consistently outperforms all existing methods, with an average 7.15\\% improvement in area under the precision-recall curve (AUPR) over the current state-of-the-art method. Importantly, EMGNN integrated multiple graphs to prioritize newly predicted cancer genes with conflicting predictions from single biological networks. For each prediction, EMGNN provided valuable biological insights via both model-level feature importance explanations and molecular-level gene set enrichment analysis. Overall, EMGNN offers a powerful new paradigm of graph learning through modeling the multilayered topological gene relationships and provides a valuable tool for cancer genomics research.Our code is publicly available at https://github.com/zhanglab-aim/EMGNN.}",
        issn = {1367-4811},
        doi = {10.1093/bioinformatics/btad643},
        url = {https://doi.org/10.1093/bioinformatics/btad643},
        eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btad643/52306228/btad643.pdf},
    }





