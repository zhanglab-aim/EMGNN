import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from model import EMGNN
from gnn_explainer import GNNExplainer
from captum.attr import Saliency, IntegratedGradients
from captum_custom import to_captum,Explainer
import networkx as nx
import os
import random
import argparse
import numpy as np
import torch
import pickle
from sklearn.metrics import f1_score
import pandas as pd 

parser = argparse.ArgumentParser()
parser.add_argument('--visualize', type=int, default=0, help='Save network explantion visualization')
parser.add_argument('--gene_label', type=str, default="cancer", help='Genes to explain: cancer,non-cancer,top_predicted')
parser.add_argument('--model_dir', type=str, default="./results/my_models/GCN_['IREF_2015', 'PCNET', 'IREF', 'STRING', 'MULTINET', 'CPDB']_2023_01_02_16_59_31", help='Path to trained model to explain')

args_2 = parser.parse_args()


features_order = ['MF: UCEC', 'MF: BLCA', 'MF: THCA', 'MF: KIRC', 'MF: READ', 'MF: LUAD', 'MF: ESCA', 'MF: LUSC', 'MF: BRCA', 'MF: COAD', 'MF: HNSC', 'MF: KIRP', 'MF: PRAD', 'MF: LIHC', 'MF: STAD', 'MF: CESC', 
'METH: UCEC', 'METH: BLCA', 'METH: THCA', 'METH: KIRC', 'METH: READ', 'METH: LUAD', 'METH: ESCA', 'METH: LUSC', 'METH: BRCA', 'METH: COAD', 'METH: HNSC', 'METH: KIRP', 'METH: PRAD', 'METH: LIHC', 'METH: STAD', 'METH: CESC', 
'GE: UCEC', 'GE: BLCA', 'GE: THCA', 'GE: KIRC', 'GE: READ', 'GE: LUAD', 'GE: ESCA', 'GE: LUSC', 'GE: BRCA', 'GE: COAD', 'GE: HNSC', 'GE: KIRP', 'GE: PRAD', 'GE: LIHC', 'GE: STAD', 'GE: CESC', 
'CNA: UCEC', 'CNA: BLCA', 'CNA: THCA', 'CNA: KIRC', 'CNA: READ', 'CNA: LUAD', 'CNA: ESCA', 'CNA: LUSC', 'CNA: BRCA', 'CNA: COAD', 'CNA: HNSC', 'CNA: KIRP', 'CNA: PRAD', 'CNA: LIHC', 'CNA: STAD', 'CNA: CESC']
features_order = np.array(features_order)

#model_dir = "./results/my_models/GCN_['CPDB', 'IREF_2015', 'PCNET', 'STRING', 'MULTINET', 'IREF']_2022_10_05_05_19_17" 
#model_dir = "./results/my_models/GIN_['IREF_2015', 'PCNET', 'STRING', 'MULTINET', 'IREF', 'CPDB']_2022_12_20_09_21_44"
#model_dir = "./results/my_models/GIN_['IREF_2015', 'PCNET', 'IREF', 'STRING', 'MULTINET', 'CPDB']_2023_01_02_15_02_41"
#model_dir = "./results/my_models/GCN_['IREF_2015', 'PCNET', 'IREF', 'STRING', 'MULTINET', 'CPDB']_2023_01_02_16_59_31" #path to trained model to explain. 

model_dir = args_2.model_dir

#read input data
with open(f"{model_dir}/edge_index.pkl","rb") as handle:
    edge_index = pickle.load(handle)
with open(f"{model_dir}/final_edge_index.pkl","rb") as handle:
    final_edge_index = pickle.load(handle)
with open(f"{model_dir}/all_node_names.pkl","rb") as handle:
    all_node_names = pickle.load(handle)
with open(f"{model_dir}/args.pkl","rb") as handle:
    args = pickle.load(handle)
with open(f"{model_dir}/meta_edge_index.pkl","rb") as handle:
    meta_edge_index = pickle.load(handle)
with open(f"{model_dir}/node2idx.pkl","rb") as handle:
    node2idx = pickle.load(handle)
with open(f"{model_dir}/meta_x.pkl","rb") as handle:
    meta_x = pickle.load(handle)
with open(f"{model_dir}/final_y.pkl","rb") as handle:
    final_y = pickle.load(handle)
with open(f"{model_dir}/batch.pkl","rb") as handle:
    batch = pickle.load(handle)
with open(f"{model_dir}/idx_test.pkl","rb") as handle:
    idx_test = pickle.load(handle)

if args.cuda:
    final_y = final_y.squeeze().cuda()
    batch = batch.cuda()
    meta_x = meta_x.cuda()
    edge_index = edge_index.cuda()
    meta_edge_index = meta_edge_index.cuda()

model = EMGNN(batch.x.shape[1],
                    args.hidden,
                    args.n_layers,
                    nclass=2,
                    args=args,
                    data=batch,
                    meta_x=meta_x,
                    node2idx=node2idx).cuda()


model.load_state_dict(torch.load(f'{model_dir}/model.pkl'))
if args.cuda:
    model = model.cuda()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#explain


#explain top predictions
number_of_input_nodes = batch.x.shape[0]

node_name_to_idx = {}
for idx,name in enumerate(all_node_names):
    node_name_to_idx[name] = idx

df = pd.read_table(f"{model_dir}/predictions.tsv")
sorted_df = df.sort_values(by="Prob_pos",ascending=False)
nodes_to_explain = list(sorted_df["Name"])[:100]
predicted_idx = []
for node in nodes_to_explain:
    n = node+"_Meta_Node"
    predicted_idx.append(node_name_to_idx[n])

predicted_idx = [x - number_of_input_nodes for x in predicted_idx]

cancer_idx = []
for idx,label in enumerate(final_y[number_of_input_nodes:]):
    if(label==1):
        cancer_idx.append(idx)

non_cancer_idx = []
for idx,label in enumerate(final_y[number_of_input_nodes:]):
    if(label==0):
        non_cancer_idx.append(idx)


if(args_2.gene_label=="cancer"):
    idx_to_explain = cancer_idx
    label_name = "cancer"
elif(args_2.gene_label=="non_cancer"):
    idx_to_explain = non_cancer_idx
    label_name = "non_cancer"
else: 
    idx_to_explain = predicted_idx
    label_name = "top_predicted"

for idx in idx_to_explain[:2]:  
    output_idx = number_of_input_nodes + idx #TARGET NODE GENE to explain

    target = int(final_y[output_idx])

    explainer = Explainer(model)


    #with open(f"./{model_dir}/explainer.pkl","wb") as handle:
    #    pickle.dump(explainer.cpu(),handle)
    # Edge explainability
    # ===================
    explainer = explainer.cuda()
    
    if not os.path.exists(f'{model_dir}/explain'):
        os.mkdir(f"{model_dir}/explain")

    if(args.edge_explain):
        # Captum assumes that for all given input tensors, dimension 0 is
        # equal to the number of samples. Therefore, we use unsqueeze(0).
        captum_model = to_captum(model, mask_type='edge', output_idx=output_idx)
        edge_mask = torch.ones(len(meta_edge_index[0]), requires_grad=True, device=device)

        ig = IntegratedGradients(captum_model)
        ig_attr_edge = ig.attribute(edge_mask.unsqueeze(0), target=target,
                                    additional_forward_args=(batch.x.cuda(), batch.edge_index.cuda(),batch, meta_edge_index,None, True),
                                    internal_batch_size=1)

        
        with open(f"{model_dir}/explain/edge_mask_explain_{idx}_{label_name}.pkl","wb") as handle:
            pickle.dump(ig_attr_edge.cpu(),handle)

        # Scale attributions to [0, 1]:

        ig_attr_edge = ig_attr_edge.squeeze(0).abs()
        ig_attr_edge /= ig_attr_edge.max()
        print(ig_attr_edge)

        # Visualize absolute values of attributions:
        #explainer = Explainer(model)
        #ax, G = explainer.visualize_subgraph(output_idx, meta_edge_index, ig_attr_edge,y=y)
        #plt.show()
        #plt.savefig(f"{model_dir}/explain/captum_edge_{idx}.pdf")
        #plt.clf()
    # Node explainability
    # ===================
    if(args.node_explain):
   
        captum_model = to_captum(model, mask_type='node', output_idx=output_idx)
        ig = IntegratedGradients(captum_model)
        ig_attr_node, approximation_error = ig.attribute(torch.concat((batch.x.cuda(),meta_x.cuda()),dim=0).float().cuda().unsqueeze(0),
                                    target=target,
                                    additional_forward_args=(batch.edge_index.cuda(), batch, None, batch.x.cuda(),True),
                                    internal_batch_size=1,
                                    return_convergence_delta=True)

        print(approximation_error)

        # Scale attributions to [0, 1]:
        print(ig_attr_node.shape)
        with open(f"{model_dir}/explain/node_feat_mask_explain_{idx}_{label_name}.pkl","wb") as handle:
            pickle.dump(ig_attr_node.cpu(),handle)

        ig_attr_node = ig_attr_node.squeeze(0).abs().sum(dim=1)
        ig_attr_node /= ig_attr_node.max()
        print(ig_attr_node)
        # Visualize absolute values of attributions:
        if(args_2.visualize == 1):
            node_dict = {i:n for i,n in enumerate(all_node_names)}
            plt.figure(figsize=(14,8))
            ax, G = explainer.visualize_subgraph(output_idx, meta_edge_index, ig_attr_edge,y=final_y,labels=node_dict)
            plt.show()
            plt.savefig(f"{model_dir}/explain/captum_node_{idx}_{label_name}.pdf",bbox_inches="tight")
            plt.clf()

     